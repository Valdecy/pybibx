"""Cross-database bibliographic source comparison for pybibx.

This module implements :func:`compare_sources`, a lightweight, dependency-minimal
comparison layer for Scopus, Web of Science, PubMed, and OpenAlex exports.  It is
intended to answer questions that normally come *before* the full bibliometric
pipeline: Which database contributed what? Where do records overlap? Which source
has better metadata coverage? What is unique to each source?

The implementation deliberately avoids calling the heavy ``pbx_probe`` analysis
workflow.  It parses only the metadata needed for source-coverage diagnostics and
therefore remains fast for exploratory source-audit work.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import unicodedata
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple, Union

import pandas as pd


_UNKNOWN_VALUES = {"", "unknown", "unkn", "nan", "none", "null", "na", "n/a", "not available"}
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
_YEAR_RE = re.compile(r"(?<!\d)(18\d{2}|19\d{2}|20\d{2}|21\d{2})(?!\d)")
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")
_SPACES_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Small utilities


def _read_text(path: Union[str, os.PathLike[str]], encodings: Optional[Sequence[str]] = None) -> str:
    encodings = encodings or ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    path = Path(path)
    last_error: Optional[Exception] = None
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc
    raw = path.read_bytes()
    try:
        import chardet  # pybibx already depends on chardet

        guess = chardet.detect(raw).get("encoding") or "latin-1"
        return raw.decode(guess, errors="replace")
    except Exception:
        if last_error is not None:
            return raw.decode("latin-1", errors="replace")
        raise


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text.lower() in _UNKNOWN_VALUES


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = "; ".join(str(v) for v in value if not _is_missing(v))
    text = str(value)
    text = text.replace("\ufeff", "")
    text = text.replace("\xa0", " ")
    text = text.strip().strip(",")
    # Remove common BibTeX wrapping leftovers without destroying mathematical text.
    while len(text) >= 2 and ((text[0] == "{" and text[-1] == "}") or (text[0] == '"' and text[-1] == '"')):
        text = text[1:-1].strip()
    text = text.replace("{{", "{").replace("}}", "}")
    text = text.replace("\n", " ").replace("\r", " ")
    text = _SPACES_RE.sub(" ", text).strip()
    return "" if _is_missing(text) else text


def _strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def _normalize_text(text: Any) -> str:
    text = _clean_value(text).lower()
    text = _strip_accents(text)
    text = text.replace("&", " and ")
    text = _NON_WORD_RE.sub(" ", text)
    return _SPACES_RE.sub(" ", text).strip()


def _normalize_title(title: Any) -> str:
    text = _normalize_text(title)
    # Remove uninformative leading articles only; do not remove scientific terms.
    tokens = [tok for tok in text.split() if tok not in {"a", "an", "the"}]
    return " ".join(tokens)


def _title_fingerprint(title: Any) -> str:
    norm = _normalize_title(title)
    if len(norm) < 12:
        return ""
    return norm


def _extract_doi(value: Any) -> str:
    if _is_missing(value):
        return ""
    text = str(value)
    text = text.replace("https://doi.org/", " ").replace("http://doi.org/", " ")
    text = text.replace("doi:", " ").replace("DOI:", " ")
    text = text.replace("[doi]", " ")
    match = _DOI_RE.search(text)
    if not match:
        return ""
    doi = match.group(0).strip().lower()
    # DOI strings exported from reference lists sometimes end with punctuation.
    doi = doi.rstrip(".,;:) ]}")
    return doi


def _extract_year(value: Any) -> str:
    if _is_missing(value):
        return ""
    match = _YEAR_RE.search(str(value))
    return match.group(1) if match else ""


def _split_people(authors: Any) -> List[str]:
    text = _clean_value(authors)
    if not text:
        return []
    text = text.replace("|", ";")
    parts = re.split(r"\s+and\s+|\s*;\s*", text)
    return [_clean_value(p) for p in parts if _clean_value(p)]


def _first_author_key(authors: Any) -> str:
    people = _split_people(authors)
    if not people:
        return ""
    first = _normalize_text(people[0])
    toks = first.split()
    # Prefer surname-like last token. Handles both "Smith J" and "J. Smith" reasonably.
    return toks[-1] if toks else ""


def _count_items(value: Any, sep_regex: str = r"\s*;\s*|\s*\|\s*") -> int:
    text = _clean_value(value)
    if not text:
        return 0
    return len([x for x in re.split(sep_regex, text) if _clean_value(x)])


def _first_nonempty(mapping: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        if key in mapping and not _is_missing(mapping[key]):
            val = _clean_value(mapping[key])
            if val:
                return val
    return ""


def _canonical_field_name(name: str) -> str:
    name = _strip_accents(name).strip().lower()
    name = name.replace("-", "_").replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]+", "", name)
    return name


# ---------------------------------------------------------------------------
# Parsers


def _split_bib_entries(text: str) -> List[str]:
    starts = [m.start() for m in re.finditer(r"(?m)^\s*@", text)]
    if not starts:
        return []
    entries = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            entries.append(chunk)
    return entries


def _parse_bib_entry(entry: str) -> Dict[str, str]:
    """Parse a single BibTeX-like record using brace-aware scanning."""
    out: Dict[str, str] = {}
    header = re.match(r"\s*@\s*([A-Za-z]+)\s*\{\s*([^,]*)\s*,", entry, re.S)
    if header:
        out["entry_type"] = header.group(1).strip()
        out["entry_key"] = header.group(2).strip()
        i = header.end()
    else:
        i = 0
    n = len(entry)
    while i < n:
        # Skip separators and whitespace.
        while i < n and entry[i] in " \t\r\n,}":
            i += 1
        if i >= n:
            break
        name_start = i
        while i < n and re.match(r"[A-Za-z0-9_\- ]", entry[i]):
            i += 1
        name = entry[name_start:i].strip()
        if not name:
            i += 1
            continue
        while i < n and entry[i].isspace():
            i += 1
        if i >= n or entry[i] != "=":
            # Not a field; advance to avoid infinite loops.
            i += 1
            continue
        i += 1
        while i < n and entry[i].isspace():
            i += 1
        if i >= n:
            break
        if entry[i] == "{":
            depth = 0
            value_start = i + 1
            while i < n:
                ch = entry[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        value = entry[value_start:i]
                        i += 1
                        break
                i += 1
            else:
                value = entry[value_start:]
        elif entry[i] == '"':
            i += 1
            value_start = i
            escaped = False
            while i < n:
                ch = entry[i]
                if ch == '"' and not escaped:
                    value = entry[value_start:i]
                    i += 1
                    break
                escaped = (ch == "\\" and not escaped)
                if ch != "\\":
                    escaped = False
                i += 1
            else:
                value = entry[value_start:]
        else:
            value_start = i
            while i < n and entry[i] not in ",\n}":
                i += 1
            value = entry[value_start:i]
        out[_canonical_field_name(name)] = _clean_value(value)
    return out


def _parse_bib_file(path: Union[str, os.PathLike[str]], source: str) -> List[Dict[str, Any]]:
    text = _read_text(path)
    records: List[Dict[str, Any]] = []
    for idx, entry in enumerate(_split_bib_entries(text)):
        raw = _parse_bib_entry(entry)
        records.append(_standardize_record(raw, source=source, row_id=idx + 1))
    return records


def _parse_scopus_csv(path: Union[str, os.PathLike[str]]) -> List[Dict[str, Any]]:
    df = pd.read_csv(path, dtype=str, encoding="utf-8", keep_default_na=False)
    df.columns = [_canonical_field_name(c) for c in df.columns]
    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        records.append(_standardize_record(row.to_dict(), source="scopus", row_id=idx + 1))
    return records



def _decode_text_bytes(raw: bytes, encodings: Optional[Sequence[str]] = None) -> str:
    """Decode bytes from normal files or ZIP members using tolerant fallbacks."""
    encodings = encodings or ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    last_error: Optional[Exception] = None
    for enc in encodings:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError as exc:
            last_error = exc
    try:
        import chardet  # pybibx already depends on chardet

        guess = chardet.detect(raw).get("encoding") or "latin-1"
        return raw.decode(guess, errors="replace")
    except Exception:
        if last_error is not None:
            return raw.decode("latin-1", errors="replace")
        raise


def _openalex_payload_to_items(payload: Any) -> List[Dict[str, Any]]:
    """Normalize the common OpenAlex API payload shapes into a list of works."""
    if isinstance(payload, dict):
        for key in ("results", "works", "items", "data"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _parse_json_or_jsonl_items(text: str) -> List[Dict[str, Any]]:
    """Parse OpenAlex JSON, JSON Lines, or NDJSON content.

    Some GitHub/OpenAlex ZIP exports are named with a CSV-like extension but
    contain one JSON object per line.  This helper detects that situation before
    pandas tries to split raw JSON on commas.
    """
    stripped = text.strip()
    if not stripped:
        return []

    # Standard JSON: API response dict, list of works, or a single work object.
    try:
        payload = json.loads(stripped)
        return _openalex_payload_to_items(payload)
    except Exception:
        pass

    # JSON Lines / NDJSON: one JSON object per line.  Tolerate trailing commas
    # from manually concatenated exports.
    items: List[Dict[str, Any]] = []
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line or line in {"[", "]", "[,]"}:
            continue
        if line.endswith(","):
            line = line[:-1].rstrip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            items.append(obj)
    return items


def _records_from_openalex_items(items: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAlex work dictionaries to pybibx comparison records."""
    records: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, Mapping):
            continue
        authors = []
        institutions = []
        countries = []
        for au in item.get("authorships", []) or []:
            if not isinstance(au, Mapping):
                continue
            author = (au.get("author") or {}).get("display_name") if isinstance(au.get("author"), Mapping) else None
            if author:
                authors.append(author)
            for inst in au.get("institutions", []) or []:
                if not isinstance(inst, Mapping):
                    continue
                if inst.get("display_name"):
                    institutions.append(inst["display_name"])
                if inst.get("country_code"):
                    countries.append(inst["country_code"])
        source_obj: Mapping[str, Any] = {}
        primary_location = item.get("primary_location") or {}
        if isinstance(primary_location, Mapping):
            source_obj = primary_location.get("source") or {}
            if not isinstance(source_obj, Mapping):
                source_obj = {}
        host_venue = item.get("host_venue") or {}
        if not isinstance(host_venue, Mapping):
            host_venue = {}
        abstract = item.get("abstract") or _invert_openalex_abstract(item.get("abstract_inverted_index"))
        issn_value = source_obj.get("issn") or host_venue.get("issn") or []
        if isinstance(issn_value, str):
            issn_text = issn_value
        else:
            issn_text = "; ".join(str(x) for x in issn_value if not _is_missing(x))
        referenced_works = item.get("referenced_works") or item.get("references") or []
        if isinstance(referenced_works, str):
            references_text = referenced_works
        else:
            references_text = "; ".join(str(x) for x in referenced_works if not _is_missing(x))
        raw = {
            "openalex_id": item.get("id") or item.get("openalex_id") or item.get("work_id"),
            "doi": item.get("doi"),
            "title": item.get("title") or item.get("display_name"),
            "year": item.get("publication_year") or item.get("year"),
            "author": "; ".join(authors),
            "affiliation": "; ".join(dict.fromkeys(institutions)),
            "country": "; ".join(dict.fromkeys(countries)),
            "journal": source_obj.get("display_name") or host_venue.get("display_name"),
            "issn": issn_text,
            "document_type": item.get("type") or item.get("type_crossref"),
            "language": item.get("language"),
            "abstract": abstract,
            "references": references_text,
            "cited_by_count": item.get("cited_by_count"),
        }
        records.append(_standardize_record(raw, source="openalex", row_id=idx + 1))
    return records


def _parse_openalex_json(path: Union[str, os.PathLike[str]]) -> List[Dict[str, Any]]:
    text = _read_text(path)
    items = _parse_json_or_jsonl_items(text)
    return _records_from_openalex_items(items)


def _invert_openalex_abstract(inverted_index: Any) -> str:
    if not isinstance(inverted_index, dict):
        return ""
    positions: List[Tuple[int, str]] = []
    for word, inds in inverted_index.items():
        if isinstance(inds, list):
            for pos in inds:
                try:
                    positions.append((int(pos), str(word)))
                except Exception:
                    pass
    return " ".join(word for _, word in sorted(positions))


def _read_openalex_delimited_text(text: str, member_name: str = "<memory>") -> pd.DataFrame:
    """Read OpenAlex website CSV/TSV text with delimiter and engine fallbacks."""
    from io import StringIO

    suffix = Path(member_name).suffix.lower()
    if suffix == ".tsv":
        attempts: List[Tuple[Any, str]] = [("\t", "python"), (None, "python"), (",", "python"), (";", "python")]
    else:
        attempts = [(None, "python"), (",", "python"), ("\t", "python"), (";", "python")]

    errors: List[str] = []
    best_df: Optional[pd.DataFrame] = None
    best_score = -1
    recognized = {
        "id", "openalex_id", "doi", "title", "display_name", "publication_year", "year",
        "authorships", "authors", "author", "source", "primary_location", "host_venue",
        "cited_by_count", "referenced_works", "abstract", "abstract_inverted_index",
    }

    for sep, engine in attempts:
        try:
            kwargs: Dict[str, Any] = {
                "dtype": str,
                "keep_default_na": False,
                "engine": engine,
                "on_bad_lines": "skip",
            }
            if sep is None:
                kwargs["sep"] = None
            else:
                kwargs["sep"] = sep
            df = pd.read_csv(StringIO(text), **kwargs)
        except Exception as exc:
            errors.append(f"sep={sep!r}, engine={engine}: {exc}")
            continue
        if df.empty:
            continue
        canonical_cols = [_canonical_field_name(str(c)) for c in df.columns]
        score = sum(1 for c in canonical_cols if c in recognized)
        # Penalize obvious wrong parses such as one giant JSON line split into
        # hundreds of anonymous comma columns.
        if len(canonical_cols) > 150 and score == 0:
            score = -1
        if score > best_score:
            best_score = score
            best_df = df
        if score > 0:
            best_df = df
            break

    if best_df is None:
        raise ValueError(f"Could not parse OpenAlex delimited file {member_name}. Attempts: {' | '.join(errors)}")
    best_df.columns = [_canonical_field_name(str(c)) for c in best_df.columns]
    return best_df


def _parse_openalex_csv(path: Union[str, os.PathLike[str]]) -> List[Dict[str, Any]]:
    text = _read_text(path)
    # Robustly support JSONL/NDJSON files accidentally saved with .csv.
    items = _parse_json_or_jsonl_items(text)
    if items:
        return _records_from_openalex_items(items)
    df = _read_openalex_delimited_text(text, member_name=str(path))
    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        records.append(_standardize_record(row.to_dict(), source="openalex", row_id=idx + 1))
    return records


def _parse_openalex_zip(path: Union[str, os.PathLike[str]]) -> List[Dict[str, Any]]:
    """Parse the first usable OpenAlex JSON/JSONL/CSV/TSV file in a ZIP archive."""
    path = Path(path)
    allowed_suffixes = {".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".txt"}
    errors: List[str] = []
    with zipfile.ZipFile(path, "r") as zf:
        candidates = [
            name
            for name in zf.namelist()
            if not name.endswith("/")
            and not Path(name).name.startswith(".")
            and Path(name).suffix.lower() in allowed_suffixes
        ]
        if not candidates:
            raise ValueError(f"OpenAlex ZIP archive does not contain a supported file (.json, .jsonl, .ndjson, .csv, .tsv, .txt): {path}")
        # Prefer structured JSON-like files, then delimited files.  Still inspect
        # every candidate because some exports have misleading extensions.
        priority = {".json": 0, ".jsonl": 1, ".ndjson": 1, ".txt": 2, ".csv": 3, ".tsv": 3}
        candidates = sorted(candidates, key=lambda n: (priority.get(Path(n).suffix.lower(), 9), n))
        for member in candidates:
            suffix = Path(member).suffix.lower()
            try:
                text = _decode_text_bytes(zf.read(member))
            except Exception as exc:
                errors.append(f"{member}: decode failed: {exc}")
                continue

            # Try JSON/JSONL/NDJSON first for all member types, including .csv,
            # because some OpenAlex exports are JSONL with a misleading extension.
            try:
                items = _parse_json_or_jsonl_items(text)
                if items:
                    return _records_from_openalex_items(items)
            except Exception as exc:
                errors.append(f"{member}: JSON/JSONL parse failed: {exc}")

            if suffix in {".csv", ".tsv", ".txt"}:
                try:
                    df = _read_openalex_delimited_text(text, member_name=member)
                    records: List[Dict[str, Any]] = []
                    for idx, row in df.iterrows():
                        records.append(_standardize_record(row.to_dict(), source="openalex", row_id=idx + 1))
                    if records:
                        return records
                except Exception as exc:
                    errors.append(f"{member}: delimited parse failed: {exc}")
                    continue

    detail = " | ".join(errors[-8:]) if errors else "No usable candidate could be parsed."
    raise ValueError(f"Could not parse any supported OpenAlex file inside ZIP archive {path}. {detail}")


def _parse_pubmed_txt(path: Union[str, os.PathLike[str]]) -> List[Dict[str, Any]]:
    text = _read_text(path)
    # PubMed records are separated by blank lines; keep records beginning with PMID.
    chunks = re.split(r"\n\s*\n", text.strip())
    records: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        if "PMID" not in chunk[:20] and not re.search(r"(?m)^PMID-", chunk):
            continue
        fields: Dict[str, List[str]] = defaultdict(list)
        current_tag: Optional[str] = None
        for raw_line in chunk.splitlines():
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            if len(line) >= 6 and line[:4].strip() and line[4:6] in {"- ", "  "}:
                tag = line[:4].strip().lower()
                value = line[6:].strip() if len(line) > 6 else ""
                current_tag = tag
                fields[tag].append(value)
            elif line.startswith("      ") and current_tag:
                continuation = line[6:].strip()
                if fields[current_tag]:
                    fields[current_tag][-1] = (fields[current_tag][-1] + " " + continuation).strip()
            else:
                # Fallback for compact forms such as PMID- 123.
                m = re.match(r"^([A-Z0-9]{2,4})\s*-\s*(.*)$", line)
                if m:
                    current_tag = m.group(1).lower()
                    fields[current_tag].append(m.group(2).strip())
        doi = ""
        for tag in ("lid", "aid", "doi"):
            for val in fields.get(tag, []):
                doi = _extract_doi(val)
                if doi:
                    break
            if doi:
                break
        journal = _first_nonempty({k: "; ".join(v) for k, v in fields.items()}, ["jt", "ta", "so"])
        raw = {
            "pmid": _first_nonempty({k: "; ".join(v) for k, v in fields.items()}, ["pmid"]),
            "doi": doi,
            "title": " ".join(fields.get("ti", [])),
            "year": _extract_year(_first_nonempty({k: "; ".join(v) for k, v in fields.items()}, ["dp", "da", "edat", "crdt"])),
            "author": " and ".join(fields.get("au", []) or fields.get("fau", [])),
            "affiliation": "; ".join(fields.get("ad", [])),
            "journal": journal,
            "issn": "; ".join(fields.get("is", [])),
            "document_type": "; ".join(fields.get("pt", [])),
            "language": "; ".join(fields.get("la", [])),
            "abstract": " ".join(fields.get("ab", [])),
            "references": "",  # PubMed MEDLINE exports normally do not include full reference lists.
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{_first_nonempty({k: '; '.join(v) for k, v in fields.items()}, ['pmid'])}/" if fields.get("pmid") else "",
        }
        records.append(_standardize_record(raw, source="pubmed", row_id=idx + 1))
    return records


def _standardize_record(raw: Mapping[str, Any], source: str, row_id: int) -> Dict[str, Any]:
    # Canonicalize keys so the same function can consume BibTeX, CSV, JSON, and MEDLINE records.
    raw_c = {_canonical_field_name(str(k)): v for k, v in raw.items()}

    title = _first_nonempty(raw_c, ["title", "ti", "display_name"])
    year = _extract_year(_first_nonempty(raw_c, ["year", "publication_year", "py", "dp", "date", "published_date"] ))
    doi = _extract_doi(_first_nonempty(raw_c, ["doi", "di", "lid", "aid", "url"] ))

    authors = _first_nonempty(raw_c, ["author", "authors", "au", "fau"])
    journal = _first_nonempty(
        raw_c,
        [
            "journal",
            "source_title",
            "abbrev_source_title",
            "publication_name",
            "booktitle",
            "journal_iso",
            "jt",
            "ta",
            "so",
            "container_title",
        ],
    )
    abstract = _first_nonempty(raw_c, ["abstract", "ab"])
    references = _first_nonempty(raw_c, ["references", "cited_references", "cr", "cr_openalex", "referenced_works"])
    document_type = _first_nonempty(raw_c, ["document_type", "type", "dt", "entry_type", "publication_type"])
    language = _first_nonempty(raw_c, ["language", "la", "languages"])
    keywords = _first_nonempty(raw_c, ["author_keywords", "keywords", "keywords_plus", "mesh_terms", "mh"])
    affiliation = _first_nonempty(raw_c, ["affiliation", "affiliations", "c1", "ad", "institutions"])
    pmid = _first_nonempty(raw_c, ["pmid", "pubmed_id"])
    openalex_id = _first_nonempty(raw_c, ["openalex_id", "id", "ut"])
    issn = _first_nonempty(raw_c, ["issn", "sn", "eissn"])
    url = _first_nonempty(raw_c, ["url", "link"])
    citations = _first_nonempty(raw_c, ["cited_by_count", "times_cited", "tc", "note", "cited_by"])

    title_norm = _normalize_title(title)
    record = {
        "source": source.lower(),
        "record_id": f"{source.lower()}_{row_id:06d}",
        "source_row": row_id,
        "entry_key": _first_nonempty(raw_c, ["entry_key", "ut", "id"]),
        "doi": doi,
        "pmid": _clean_value(pmid),
        "openalex_id": _clean_value(openalex_id),
        "title": _clean_value(title),
        "title_norm": title_norm,
        "title_fingerprint": _title_fingerprint(title),
        "year": year,
        "author": _clean_value(authors),
        "first_author_key": _first_author_key(authors),
        "journal": _clean_value(journal),
        "issn": _clean_value(issn),
        "document_type": _clean_value(document_type),
        "language": _clean_value(language),
        "keywords": _clean_value(keywords),
        "affiliation": _clean_value(affiliation),
        "abstract": _clean_value(abstract),
        "references": _clean_value(references),
        "url": _clean_value(url),
        "citations_raw": _clean_value(citations),
    }
    return record


def _parse_source(path: Union[str, os.PathLike[str]], source: str) -> List[Dict[str, Any]]:
    source = source.lower()
    suffix = Path(path).suffix.lower()
    if source == "pubmed":
        return _parse_pubmed_txt(path)
    if source == "openalex":
        if suffix == ".json":
            return _parse_openalex_json(path)
        if suffix == ".csv":
            return _parse_openalex_csv(path)
        if suffix == ".zip":
            return _parse_openalex_zip(path)
        raise ValueError("OpenAlex input must be a .json, .csv, or .zip file.")
    if source == "scopus" and suffix == ".csv":
        return _parse_scopus_csv(path)
    if suffix in {".bib", ".bibtex", ".txt"}:
        return _parse_bib_file(path, source=source)
    raise ValueError(f"Unsupported input format for {source}: {suffix}")


# ---------------------------------------------------------------------------
# Matching and result object


def _match_confidence(method: str, score: float) -> Tuple[float, str, str]:
    """Map the matching evidence to a conservative confidence score and label."""
    method = (method or "").lower()
    if method in {"doi", "pmid", "openalex_id"}:
        return 1.0, "very_high", "exact persistent identifier"
    if method == "title_exact":
        return 0.96, "high", "exact normalized title"
    if method == "title_fuzzy":
        confidence = max(0.0, min(0.95, float(score) * 0.95))
        if confidence >= 0.92:
            label = "high"
        elif confidence >= 0.88:
            label = "medium"
        else:
            label = "low"
        return round(confidence, 4), label, "fuzzy title similarity within year window"
    return round(float(score), 4), "unknown", "unspecified evidence"


def _best_match_method(methods: Iterable[str]) -> str:
    priority = {"doi": 5, "pmid": 5, "openalex_id": 5, "title_exact": 4, "title_fuzzy": 3}
    methods = [m for m in methods if m]
    if not methods:
        return ""
    return sorted(methods, key=lambda m: (priority.get(m, 0), m), reverse=True)[0]


def _confidence_label_from_score(score: float) -> str:
    if score >= 0.99:
        return "very_high"
    if score >= 0.92:
        return "high"
    if score >= 0.88:
        return "medium"
    if score > 0:
        return "low"
    return "unmatched"


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, x: str) -> str:
        parent = self.parent[x]
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass
class SourceComparisonResult:
    """Container returned by :func:`compare_sources`.

    The object exposes every table as a pandas DataFrame and can export the
    complete audit to a folder.  It intentionally behaves like a light report
    object rather than a black-box visualization.
    """

    records: pd.DataFrame
    documents: pd.DataFrame
    source_summary: pd.DataFrame
    data_quality: pd.DataFrame
    overlap_counts: pd.DataFrame
    overlap_jaccard: pd.DataFrame
    overlap_coverage: pd.DataFrame
    source_combinations: pd.DataFrame
    database_contribution_score: pd.DataFrame
    matching_diagnostics: pd.DataFrame
    matching_confidence_report: pd.DataFrame
    pairwise_matches: pd.DataFrame
    unique_documents: pd.DataFrame
    year_distribution: pd.DataFrame
    document_type_distribution: pd.DataFrame
    language_distribution: pd.DataFrame
    journal_distribution: pd.DataFrame
    warnings: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def matched_documents(self) -> pd.DataFrame:
        """Alias for ``matching_confidence_report`` for convenient access."""
        return self.matching_confidence_report

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary of all report tables."""
        return {
            "records": self.records,
            "documents": self.documents,
            "source_summary": self.source_summary,
            "data_quality": self.data_quality,
            "overlap_counts": self.overlap_counts,
            "overlap_jaccard": self.overlap_jaccard,
            "overlap_coverage": self.overlap_coverage,
            "source_combinations": self.source_combinations,
            "database_contribution_score": self.database_contribution_score,
            "matching_diagnostics": self.matching_diagnostics,
            "matching_confidence_report": self.matching_confidence_report,
            "pairwise_matches": self.pairwise_matches,
            "unique_documents": self.unique_documents,
            "year_distribution": self.year_distribution,
            "document_type_distribution": self.document_type_distribution,
            "language_distribution": self.language_distribution,
            "journal_distribution": self.journal_distribution,
            "warnings": self.warnings,
            "parameters": self.parameters,
        }

    def export(self, out: Union[str, os.PathLike[str]] = "pybibx_source_comparison", html: bool = True) -> str:
        """Export all comparison tables as CSV and optionally an HTML report.

        Parameters
        ----------
        out:
            Output directory.
        html:
            If ``True``, write ``source_comparison_report.html``.

        Returns
        -------
        str
            The output directory path.
        """
        out_path = Path(out)
        out_path.mkdir(parents=True, exist_ok=True)
        for name, table in self.to_dict().items():
            if isinstance(table, pd.DataFrame):
                table.to_csv(out_path / f"{name}.csv", index=True)
        if html:
            (out_path / "source_comparison_report.html").write_text(self._html_report(), encoding="utf-8")
        return str(out_path)

    def _html_report(self) -> str:
        def _table(df: pd.DataFrame, max_rows: int = 50) -> str:
            if df.empty:
                return "<p><em>No data.</em></p>"
            return df.head(max_rows).to_html(classes="table", border=0, escape=False)

        warning_html = "".join(f"<li>{w}</li>" for w in self.warnings) or "<li>No warnings.</li>"
        sections = [
            ("Source Summary", self.source_summary),
            ("Data Quality", self.data_quality),
            ("Overlap Counts", self.overlap_counts),
            ("Overlap Jaccard", self.overlap_jaccard),
            ("Overlap Coverage", self.overlap_coverage),
            ("Source Combinations", self.source_combinations),
            ("Database Contribution Score", self.database_contribution_score),
            ("Matching Diagnostics", self.matching_diagnostics),
            ("Matching Confidence Report", self.matching_confidence_report),
            ("Unique Documents", self.unique_documents),
            ("Pairwise Matches", self.pairwise_matches),
            ("Year Distribution", self.year_distribution),
            ("Document Type Distribution", self.document_type_distribution),
            ("Language Distribution", self.language_distribution),
            ("Top Journals/Sources", self.journal_distribution),
        ]
        body = "\n".join(f"<h2>{title}</h2>{_table(df)}" for title, df in sections)
        return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>pybibx Source Comparison Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
h1 {{ margin-bottom: 0; }}
.subtitle {{ color: #555; margin-top: 6px; }}
.table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 13px; }}
.table th, .table td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
.table th {{ background: #f2f2f2; }}
code {{ background: #f7f7f7; padding: 2px 4px; }}
</style>
</head>
<body>
<h1>pybibx Source Comparison Report</h1>
<p class="subtitle">Generated by <code>pybibx.compare_sources</code>.</p>
<h2>Warnings</h2><ul>{warning_html}</ul>
{body}
</body>
</html>
"""

    def plot_overlap_heatmap(self, path: Union[str, os.PathLike[str]] = "overlap_heatmap.png", metric: str = "jaccard") -> str:
        """Save a heatmap for the overlap matrix.

        Parameters
        ----------
        path:
            Output image path.
        metric:
            ``"jaccard"``, ``"counts"`` or ``"coverage"``.
        """
        metric = metric.lower()
        if metric == "jaccard":
            df = self.overlap_jaccard
            title = "Source overlap - Jaccard index"
        elif metric == "coverage":
            df = self.overlap_coverage
            title = "Source overlap - row coverage"
        elif metric == "counts":
            df = self.overlap_counts
            title = "Source overlap - shared unique documents"
        else:
            raise ValueError("metric must be 'jaccard', 'coverage', or 'counts'.")
        try:
            import matplotlib
            try:
                matplotlib.use("Agg")
            except Exception:
                pass
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - optional plotting dependency edge case
            raise ImportError("Install matplotlib to use plot_overlap_heatmap.") from exc
        fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(df.columns) + 2), max(4, 0.7 * len(df.index) + 2)))
        values = df.astype(float).values if not df.empty else []
        im = ax.imshow(values, aspect="auto")
        ax.set_xticks(range(len(df.columns)))
        ax.set_yticks(range(len(df.index)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right")
        ax.set_yticklabels(df.index)
        ax.set_title(title)
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                val = df.iloc[i, j]
                label = f"{val:.2f}" if metric != "counts" else str(int(val))
                ax.text(j, i, label, ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def print_summary(self, n_unique: int = 10) -> None:
        """Print a compact textual summary to stdout."""
        print("pybibx source comparison")
        print("=" * 28)
        print(self.source_summary.to_string(index=False))
        print("\nOverlap counts:")
        print(self.overlap_counts.to_string())
        if not self.database_contribution_score.empty:
            print("\nDatabase contribution score:")
            print(self.database_contribution_score.to_string(index=False))
        if not self.matching_diagnostics.empty:
            print("\nMatching diagnostics:")
            print(self.matching_diagnostics.to_string(index=False))
        if not self.unique_documents.empty:
            print(f"\nFirst {n_unique} unique documents:")
            cols = [c for c in ["global_id", "source", "title", "year", "doi"] if c in self.unique_documents.columns]
            print(self.unique_documents[cols].head(n_unique).to_string(index=False))


# ---------------------------------------------------------------------------
# Core comparison functions


def compare_sources(
    scopus: Optional[Union[str, os.PathLike[str]]] = None,
    wos: Optional[Union[str, os.PathLike[str]]] = None,
    openalex: Optional[Union[str, os.PathLike[str]]] = None,
    pubmed: Optional[Union[str, os.PathLike[str]]] = None,
    sources: Optional[Mapping[str, Union[str, os.PathLike[str]]]] = None,
    out: Optional[Union[str, os.PathLike[str]]] = None,
    export: bool = False,
    make_plots: bool = False,
    min_title_similarity: float = 0.94,
    year_window: int = 1,
    missing: str = "warn",
    deduplicate_within_source: bool = True,
    top_n: int = 25,
) -> SourceComparisonResult:
    """Compare bibliographic coverage across Scopus, WoS, OpenAlex and PubMed.

    Parameters
    ----------
    scopus, wos, openalex, pubmed:
        Paths to source exports.  Scopus accepts ``.bib`` and ``.csv``.  WoS
        accepts BibTeX-style ``.bib`` files.  OpenAlex accepts API ``.json`` or
        website ``.csv`` exports.  PubMed accepts MEDLINE ``.txt`` exports.
    sources:
        Optional mapping for additional/custom names, e.g.
        ``{"scopus": "scopus.bib", "wos": "wos.bib"}``. Explicit keyword
        arguments and this mapping are merged; explicit arguments take priority.
    out:
        Output directory used when ``export=True`` or ``make_plots=True``.
    export:
        If ``True``, write CSV tables and an HTML report.
    make_plots:
        If ``True``, save heatmaps for counts, Jaccard, and coverage.
    min_title_similarity:
        Minimum normalized title similarity for DOI-free fuzzy matching.
    year_window:
        Maximum publication-year distance for fuzzy title matching.
    missing:
        How to handle missing paths: ``"warn"`` skips the missing source and
        records a warning; ``"raise"`` raises ``FileNotFoundError``; ``"ignore"``
        silently skips.
    deduplicate_within_source:
        If ``True``, documents duplicated inside the same database are clustered
        before source-level coverage is computed.
    top_n:
        Number of top journals/sources retained in the journal distribution table.

    Returns
    -------
    SourceComparisonResult
        A report object with pandas DataFrames and export/plot helpers.

    Examples
    --------
    >>> import pybibx
    >>> result = pybibx.compare_sources(scopus="scopus.bib", wos="wos.bib", pubmed="pubmed.txt")
    >>> result.source_summary
    >>> result.overlap_counts
    >>> result.export("comparison_report")
    """
    if not (0 <= min_title_similarity <= 1):
        raise ValueError("min_title_similarity must be between 0 and 1.")
    if year_window < 0:
        raise ValueError("year_window must be non-negative.")
    missing = missing.lower()
    if missing not in {"warn", "raise", "ignore"}:
        raise ValueError("missing must be 'warn', 'raise', or 'ignore'.")

    source_paths: Dict[str, Optional[Union[str, os.PathLike[str]]]] = {}
    if sources:
        source_paths.update({str(k).lower(): v for k, v in sources.items()})
    explicit = {"scopus": scopus, "wos": wos, "openalex": openalex, "pubmed": pubmed}
    for key, value in explicit.items():
        if value is not None:
            source_paths[key] = value

    warnings_list: List[str] = []
    all_records: List[Dict[str, Any]] = []
    loaded_sources: List[str] = []

    for source, path in source_paths.items():
        if path is None:
            continue
        path_obj = Path(path)
        if not path_obj.exists():
            msg = f"Source '{source}' skipped because file was not found: {path_obj}"
            if missing == "raise":
                raise FileNotFoundError(msg)
            if missing == "warn":
                warnings.warn(msg)
                warnings_list.append(msg)
            continue
        parsed = _parse_source(path_obj, source=source)
        all_records.extend(parsed)
        loaded_sources.append(source.lower())

    if not all_records:
        raise ValueError("No source files were loaded. Check the supplied paths.")

    records = pd.DataFrame(all_records)
    records = _ensure_columns(records)

    records, pairwise_matches = _cluster_records(
        records,
        min_title_similarity=min_title_similarity,
        year_window=year_window,
        deduplicate_within_source=deduplicate_within_source,
    )

    documents = _build_documents_table(records)
    source_summary = _build_source_summary(records, documents, loaded_sources)
    data_quality = _build_data_quality(records)
    overlap_counts, overlap_jaccard, overlap_coverage = _build_overlap_tables(records)
    source_combinations = _build_source_combinations(records)
    database_contribution_score = _build_database_contribution_score(records, documents)
    matching_diagnostics = _build_matching_diagnostics(records, documents, pairwise_matches)
    matching_confidence_report = _build_matching_confidence_report(records, documents, pairwise_matches)
    unique_documents = _build_unique_documents(records, documents)
    year_distribution = _distribution_table(records, "year")
    document_type_distribution = _distribution_table(records, "document_type")
    language_distribution = _distribution_table(records, "language")
    journal_distribution = _top_journal_table(records, top_n=top_n)

    result = SourceComparisonResult(
        records=records,
        documents=documents,
        source_summary=source_summary,
        data_quality=data_quality,
        overlap_counts=overlap_counts,
        overlap_jaccard=overlap_jaccard,
        overlap_coverage=overlap_coverage,
        source_combinations=source_combinations,
        database_contribution_score=database_contribution_score,
        matching_diagnostics=matching_diagnostics,
        matching_confidence_report=matching_confidence_report,
        pairwise_matches=pairwise_matches,
        unique_documents=unique_documents,
        year_distribution=year_distribution,
        document_type_distribution=document_type_distribution,
        language_distribution=language_distribution,
        journal_distribution=journal_distribution,
        warnings=warnings_list,
        parameters={
            "min_title_similarity": min_title_similarity,
            "year_window": year_window,
            "missing": missing,
            "deduplicate_within_source": deduplicate_within_source,
            "loaded_sources": sorted(set(loaded_sources)),
        },
    )

    if export or make_plots:
        out_dir = Path(out or "pybibx_source_comparison")
        if export:
            result.export(out_dir, html=True)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
        if make_plots:
            result.plot_overlap_heatmap(out_dir / "overlap_counts_heatmap.png", metric="counts")
            result.plot_overlap_heatmap(out_dir / "overlap_jaccard_heatmap.png", metric="jaccard")
            result.plot_overlap_heatmap(out_dir / "overlap_coverage_heatmap.png", metric="coverage")
    return result


def _ensure_columns(records: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source",
        "record_id",
        "source_row",
        "entry_key",
        "doi",
        "pmid",
        "openalex_id",
        "title",
        "title_norm",
        "title_fingerprint",
        "year",
        "author",
        "first_author_key",
        "journal",
        "issn",
        "document_type",
        "language",
        "keywords",
        "affiliation",
        "abstract",
        "references",
        "url",
        "citations_raw",
    ]
    for col in columns:
        if col not in records.columns:
            records[col] = ""
    records = records[columns].copy()
    for col in columns:
        records[col] = records[col].fillna("").astype(str)
    return records


def _cluster_records(
    records: pd.DataFrame,
    min_title_similarity: float,
    year_window: int,
    deduplicate_within_source: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ids = records["record_id"].tolist()
    uf = _UnionFind(ids)
    matches: List[Dict[str, Any]] = []

    def add_match(a_idx: int, b_idx: int, method: str, score: float, key: str) -> None:
        a = records.iloc[a_idx]
        b = records.iloc[b_idx]
        if (not deduplicate_within_source) and a["source"] == b["source"]:
            return
        uf.union(a["record_id"], b["record_id"])
        confidence_score, confidence_label, evidence = _match_confidence(method, score)
        match_scope = "cross_source" if a["source"] != b["source"] else "within_source"
        matches.append(
            {
                "match_scope": match_scope,
                "source_a": a["source"],
                "record_id_a": a["record_id"],
                "title_a": a["title"],
                "doi_a": a["doi"],
                "year_a": a["year"],
                "source_b": b["source"],
                "record_id_b": b["record_id"],
                "title_b": b["title"],
                "doi_b": b["doi"],
                "year_b": b["year"],
                "match_method": method,
                "match_key": key,
                "score": round(float(score), 4),
                "confidence_score": confidence_score,
                "confidence_label": confidence_label,
                "evidence": evidence,
            }
        )

    # Deterministic exact identifier matches.
    for col, method in (("doi", "doi"), ("pmid", "pmid"), ("openalex_id", "openalex_id")):
        groups: Dict[str, List[int]] = defaultdict(list)
        for idx, value in records[col].items():
            value = str(value).strip().lower()
            if value:
                groups[value].append(int(idx))
        for key, idxs in groups.items():
            if len(idxs) > 1:
                for a_idx, b_idx in combinations(idxs, 2):
                    add_match(a_idx, b_idx, method=method, score=1.0, key=key)

    # Exact normalized-title matches. Useful when DOI is absent or malformed.
    title_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, value in records["title_fingerprint"].items():
        if value and len(value) >= 18:
            title_groups[value].append(int(idx))
    for key, idxs in title_groups.items():
        if len(idxs) > 1:
            for a_idx, b_idx in combinations(idxs, 2):
                add_match(a_idx, b_idx, method="title_exact", score=1.0, key=key[:120])

    # Fuzzy title matching for residual cross-source records. Block by first title token and year.
    # This protects speed and reduces false positives.
    rows = records.reset_index().to_dict("records")
    blocks: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        title_norm = row.get("title_norm", "")
        if len(title_norm) < 24:
            continue
        first_token = title_norm.split()[0]
        year = row.get("year", "") or ""
        year_keys: List[str]
        if year.isdigit():
            y = int(year)
            year_keys = [str(k) for k in range(y - year_window, y + year_window + 1)]
        else:
            year_keys = [""]
        for ykey in year_keys:
            blocks[(first_token, ykey)].append(row)

    seen_pairs: Set[Tuple[str, str]] = set()
    for block_rows in blocks.values():
        if len(block_rows) < 2:
            continue
        for a, b in combinations(block_rows, 2):
            ida, idb = a["record_id"], b["record_id"]
            pair = tuple(sorted((ida, idb)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if a["source"] == b["source"] and not deduplicate_within_source:
                continue
            # Avoid wasting work where strong identifiers disagree.
            if a.get("doi") and b.get("doi") and a["doi"] != b["doi"]:
                continue
            score = SequenceMatcher(None, a["title_norm"], b["title_norm"]).ratio()
            if score >= min_title_similarity:
                add_match(a["index"], b["index"], method="title_fuzzy", score=score, key=a["title_norm"][:120])

    # Assign stable global document identifiers.
    roots = [uf.find(rid) for rid in records["record_id"]]
    unique_roots = {root: i + 1 for i, root in enumerate(sorted(set(roots)))}
    records = records.copy()
    records["global_id"] = [f"DOC_{unique_roots[root]:06d}" for root in roots]

    pairwise = pd.DataFrame(matches)
    if not pairwise.empty:
        pairwise = pairwise.drop_duplicates(subset=["record_id_a", "record_id_b", "match_method", "match_key"]).reset_index(drop=True)
        # Attach global IDs after clustering.
        id_to_global = dict(zip(records["record_id"], records["global_id"]))
        pairwise["global_id"] = pairwise["record_id_a"].map(id_to_global)
        cols = ["global_id"] + [c for c in pairwise.columns if c != "global_id"]
        pairwise = pairwise[cols]
    else:
        pairwise = pd.DataFrame(
            columns=[
                "global_id",
                "match_scope",
                "source_a",
                "record_id_a",
                "title_a",
                "doi_a",
                "year_a",
                "source_b",
                "record_id_b",
                "title_b",
                "doi_b",
                "year_b",
                "match_method",
                "match_key",
                "score",
                "confidence_score",
                "confidence_label",
                "evidence",
            ]
        )
    return records, pairwise


def _best_value(group: pd.DataFrame, column: str) -> str:
    vals = [_clean_value(v) for v in group[column].tolist() if not _is_missing(v)]
    if not vals:
        return ""
    # Prefer the most frequent non-empty value; break ties by longest string.
    counts = Counter(vals)
    return sorted(counts.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)[0][0]


def _build_documents_table(records: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for gid, group in records.groupby("global_id", sort=True):
        sources = sorted(group["source"].unique().tolist())
        row = {
            "global_id": gid,
            "sources": "; ".join(sources),
            "n_sources": len(sources),
            "n_records": group.shape[0],
            "doi": _best_value(group, "doi"),
            "pmid": _best_value(group, "pmid"),
            "openalex_id": _best_value(group, "openalex_id"),
            "title": _best_value(group, "title"),
            "year": _best_value(group, "year"),
            "author": _best_value(group, "author"),
            "journal": _best_value(group, "journal"),
            "document_type": _best_value(group, "document_type"),
            "language": _best_value(group, "language"),
            "has_abstract": any(not _is_missing(v) for v in group["abstract"]),
            "has_references": any(not _is_missing(v) for v in group["references"]),
            "source_record_ids": "; ".join(group["record_id"].tolist()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n_sources", "year", "title"], ascending=[False, False, True]).reset_index(drop=True)


def _build_source_summary(records: pd.DataFrame, documents: pd.DataFrame, loaded_sources: Sequence[str]) -> pd.DataFrame:
    global_docs = set(records["global_id"].unique())
    rows = []
    for source in sorted(set(loaded_sources)):
        src = records[records["source"] == source]
        src_docs = set(src["global_id"].unique())
        unique_only = [gid for gid in src_docs if records.loc[records["global_id"] == gid, "source"].nunique() == 1]
        rows.append(
            {
                "source": source,
                "records_loaded": int(src.shape[0]),
                "unique_documents_after_matching": int(len(src_docs)),
                "duplicated_records_inside_source": int(src.shape[0] - len(src_docs)),
                "documents_unique_to_source": int(len(unique_only)),
                "documents_shared_with_any_other_source": int(len(src_docs) - len(unique_only)),
                "share_of_global_unique_documents": round(len(src_docs) / len(global_docs), 4) if global_docs else 0,
                "doi_coverage": _coverage(src, "doi"),
                "title_coverage": _coverage(src, "title"),
                "abstract_coverage": _coverage(src, "abstract"),
                "reference_coverage": _coverage(src, "references"),
                "author_coverage": _coverage(src, "author"),
                "affiliation_coverage": _coverage(src, "affiliation"),
                "journal_coverage": _coverage(src, "journal"),
                "year_coverage": _coverage(src, "year"),
            }
        )
    rows.append(
        {
            "source": "GLOBAL",
            "records_loaded": int(records.shape[0]),
            "unique_documents_after_matching": int(documents.shape[0]),
            "duplicated_records_inside_source": int(records.shape[0] - records.drop_duplicates(["source", "global_id"]).shape[0]),
            "documents_unique_to_source": int((documents["n_sources"] == 1).sum()) if not documents.empty else 0,
            "documents_shared_with_any_other_source": int((documents["n_sources"] > 1).sum()) if not documents.empty else 0,
            "share_of_global_unique_documents": 1.0,
            "doi_coverage": _coverage(records, "doi"),
            "title_coverage": _coverage(records, "title"),
            "abstract_coverage": _coverage(records, "abstract"),
            "reference_coverage": _coverage(records, "references"),
            "author_coverage": _coverage(records, "author"),
            "affiliation_coverage": _coverage(records, "affiliation"),
            "journal_coverage": _coverage(records, "journal"),
            "year_coverage": _coverage(records, "year"),
        }
    )
    return pd.DataFrame(rows)


def _coverage(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    return round(float((~df[column].map(_is_missing)).mean()), 4)


def _build_data_quality(records: pd.DataFrame) -> pd.DataFrame:
    fields = {
        "doi": "DOI",
        "title": "Title",
        "abstract": "Abstract",
        "references": "References",
        "author": "Authors",
        "affiliation": "Affiliations",
        "journal": "Journal/Source",
        "year": "Year",
        "document_type": "Document type",
        "language": "Language",
        "keywords": "Keywords",
    }
    rows = []
    for source, group in records.groupby("source", sort=True):
        n = group.shape[0]
        for col, label in fields.items():
            non_missing = int((~group[col].map(_is_missing)).sum()) if col in group.columns else 0
            rows.append(
                {
                    "source": source,
                    "field": label,
                    "available_records": non_missing,
                    "total_records": n,
                    "coverage": round(non_missing / n, 4) if n else 0,
                }
            )
    return pd.DataFrame(rows)


def _source_sets(records: pd.DataFrame) -> Dict[str, Set[str]]:
    return {source: set(group["global_id"].unique()) for source, group in records.groupby("source", sort=True)}


def _build_overlap_tables(records: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sets = _source_sets(records)
    sources = sorted(sets.keys())
    counts = pd.DataFrame(0, index=sources, columns=sources, dtype=int)
    jaccard = pd.DataFrame(0.0, index=sources, columns=sources, dtype=float)
    coverage = pd.DataFrame(0.0, index=sources, columns=sources, dtype=float)
    for a in sources:
        for b in sources:
            inter = len(sets[a] & sets[b])
            union = len(sets[a] | sets[b])
            counts.loc[a, b] = inter
            jaccard.loc[a, b] = round(inter / union, 4) if union else 0.0
            coverage.loc[a, b] = round(inter / len(sets[a]), 4) if sets[a] else 0.0
    return counts, jaccard, coverage


def _build_source_combinations(records: pd.DataFrame) -> pd.DataFrame:
    doc_sources = records.groupby("global_id")["source"].apply(lambda x: tuple(sorted(set(x)))).reset_index()
    counter = Counter(doc_sources["source"])
    rows = []
    total = len(doc_sources)
    for combo, count in sorted(counter.items(), key=lambda kv: (len(kv[0]), kv[0])):
        rows.append(
            {
                "source_combination": " + ".join(combo),
                "n_sources": len(combo),
                "unique_documents": int(count),
                "share_of_global_unique_documents": round(count / total, 4) if total else 0,
            }
        )
    return pd.DataFrame(rows)


def _build_database_contribution_score(records: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    """Quantify how much each database contributes to the merged corpus."""
    if records.empty or documents.empty:
        return pd.DataFrame()
    global_unique = documents.shape[0]
    rows: List[Dict[str, Any]] = []
    for source, group in records.groupby("source", sort=True):
        src_docs = set(group["global_id"].unique())
        exclusive_docs = {
            gid
            for gid in src_docs
            if records.loc[records["global_id"] == gid, "source"].nunique() == 1
        }
        shared_docs = src_docs - exclusive_docs
        internal_duplicates = group.shape[0] - len(src_docs)
        exclusive_share_inside_source = len(exclusive_docs) / len(src_docs) if src_docs else 0.0
        shared_share_inside_source = len(shared_docs) / len(src_docs) if src_docs else 0.0
        redundancy_rate = 1.0 - exclusive_share_inside_source
        # The score is intentionally interpretable: unique contribution to the
        # global corpus, slightly rewarded for metadata/reference usability.
        quality_bonus = (
            _coverage(group, "doi")
            + _coverage(group, "title")
            + _coverage(group, "abstract")
            + _coverage(group, "references")
            + _coverage(group, "year")
        ) / 5.0
        unique_contribution_share = len(exclusive_docs) / global_unique if global_unique else 0.0
        coverage_share = len(src_docs) / global_unique if global_unique else 0.0
        contribution_score_0_100 = 100.0 * (0.65 * unique_contribution_share + 0.20 * coverage_share + 0.15 * quality_bonus)
        rows.append(
            {
                "source": source,
                "records_loaded": int(group.shape[0]),
                "unique_documents_after_matching": int(len(src_docs)),
                "exclusive_documents": int(len(exclusive_docs)),
                "shared_documents": int(len(shared_docs)),
                "internal_duplicate_records": int(internal_duplicates),
                "unique_contribution_share_global": round(unique_contribution_share, 4),
                "coverage_share_global": round(coverage_share, 4),
                "exclusive_share_inside_source": round(exclusive_share_inside_source, 4),
                "shared_share_inside_source": round(shared_share_inside_source, 4),
                "redundancy_rate": round(redundancy_rate, 4),
                "metadata_quality_bonus": round(quality_bonus, 4),
                "contribution_score_0_100": round(contribution_score_0_100, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("contribution_score_0_100", ascending=False).reset_index(drop=True)


def _build_matching_diagnostics(records: pd.DataFrame, documents: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    """Summarize the evidence used to match records across and within databases."""
    if pairwise.empty:
        return pd.DataFrame(
            [{
                "match_scope": "none",
                "match_method": "none",
                "confidence_label": "unmatched",
                "pairwise_match_pairs": 0,
                "unique_matched_documents_involved": 0,
                "share_of_all_documents_with_match_evidence": 0.0,
                "mean_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
            }]
        )
    if "match_scope" not in pairwise.columns:
        pairwise = pairwise.copy()
        pairwise["match_scope"] = pairwise.apply(lambda r: "cross_source" if r.get("source_a") != r.get("source_b") else "within_source", axis=1)
    all_docs_with_evidence = set(pairwise["global_id"].dropna().astype(str))
    denominator = max(1, len(all_docs_with_evidence))
    rows: List[Dict[str, Any]] = []
    grouped = pairwise.groupby(["match_scope", "match_method", "confidence_label"], dropna=False)
    for (scope, method, label), group in grouped:
        docs = set(group["global_id"].dropna().astype(str))
        scores = pd.to_numeric(group["score"], errors="coerce")
        rows.append(
            {
                "match_scope": scope,
                "match_method": method,
                "confidence_label": label,
                "pairwise_match_pairs": int(group.shape[0]),
                "unique_matched_documents_involved": int(len(docs)),
                "share_of_all_documents_with_match_evidence": round(len(docs) / denominator, 4),
                "mean_score": round(float(scores.mean()), 4),
                "min_score": round(float(scores.min()), 4),
                "max_score": round(float(scores.max()), 4),
            }
        )
    for scope, group in pairwise.groupby("match_scope", sort=True):
        docs = set(group["global_id"].dropna().astype(str))
        scores = pd.to_numeric(group["score"], errors="coerce")
        rows.append(
            {
                "match_scope": scope,
                "match_method": "TOTAL",
                "confidence_label": "all",
                "pairwise_match_pairs": int(group.shape[0]),
                "unique_matched_documents_involved": int(len(docs)),
                "share_of_all_documents_with_match_evidence": round(len(docs) / denominator, 4),
                "mean_score": round(float(scores.mean()), 4),
                "min_score": round(float(scores.min()), 4),
                "max_score": round(float(scores.max()), 4),
            }
        )
    scores = pd.to_numeric(pairwise["score"], errors="coerce")
    rows.append(
        {
            "match_scope": "ALL",
            "match_method": "TOTAL",
            "confidence_label": "all",
            "pairwise_match_pairs": int(pairwise.shape[0]),
            "unique_matched_documents_involved": int(len(all_docs_with_evidence)),
            "share_of_all_documents_with_match_evidence": 1.0,
            "mean_score": round(float(scores.mean()), 4),
            "min_score": round(float(scores.min()), 4),
            "max_score": round(float(scores.max()), 4),
        }
    )
    return pd.DataFrame(rows)


def _build_matching_confidence_report(records: pd.DataFrame, documents: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    """List every multi-source document and the evidence supporting the match."""
    if records.empty or documents.empty:
        return pd.DataFrame()
    matched_docs = documents[documents["n_sources"] > 1].copy()
    if matched_docs.empty:
        return pd.DataFrame(
            columns=[
                "global_id", "sources", "n_sources", "title", "year", "doi", "pmid", "openalex_id",
                "best_match_method", "match_methods", "best_confidence_score", "confidence_label",
                "pairwise_match_pairs", "record_ids_by_source", "titles_by_source"
            ]
        )
    rows: List[Dict[str, Any]] = []
    pairwise_by_gid = {gid: g for gid, g in pairwise.groupby("global_id")} if not pairwise.empty else {}
    for _, doc in matched_docs.iterrows():
        gid = doc["global_id"]
        group = records[records["global_id"] == gid].copy()
        pg = pairwise_by_gid.get(gid, pd.DataFrame())
        methods = sorted(set(pg["match_method"].dropna().astype(str))) if not pg.empty else []
        best_method = _best_match_method(methods)
        if not pg.empty and "confidence_score" in pg.columns:
            scores = pd.to_numeric(pg["confidence_score"], errors="coerce").dropna()
            best_conf = float(scores.max()) if not scores.empty else 0.0
        else:
            best_conf = 0.0
        # Be conservative at document level: if any pair was only fuzzy, show all evidence.
        confidence_label = _confidence_label_from_score(best_conf)
        record_ids_by_source = []
        titles_by_source = []
        for source, sg in group.groupby("source", sort=True):
            record_ids_by_source.append(f"{source}: " + ", ".join(sg["record_id"].tolist()))
            titles = sorted(set(_clean_value(t) for t in sg["title"].tolist() if _clean_value(t)))
            titles_by_source.append(f"{source}: " + " | ".join(titles[:3]))
        rows.append(
            {
                "global_id": gid,
                "sources": doc.get("sources", ""),
                "n_sources": int(doc.get("n_sources", 0)),
                "title": doc.get("title", ""),
                "year": doc.get("year", ""),
                "doi": doc.get("doi", ""),
                "pmid": doc.get("pmid", ""),
                "openalex_id": doc.get("openalex_id", ""),
                "best_match_method": best_method,
                "match_methods": "; ".join(methods),
                "best_confidence_score": round(best_conf, 4),
                "confidence_label": confidence_label,
                "pairwise_match_pairs": int(pg.shape[0]) if not pg.empty else 0,
                "record_ids_by_source": "; ".join(record_ids_by_source),
                "titles_by_source": "; ".join(titles_by_source),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_sources", "best_confidence_score", "year", "title"], ascending=[False, False, False, True]).reset_index(drop=True)


def _build_unique_documents(records: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    if documents.empty:
        return pd.DataFrame()
    unique = documents[documents["n_sources"] == 1].copy()
    if unique.empty:
        return unique
    unique["source"] = unique["sources"]
    cols = [
        "global_id",
        "source",
        "title",
        "year",
        "doi",
        "author",
        "journal",
        "document_type",
        "language",
        "source_record_ids",
    ]
    return unique[[c for c in cols if c in unique.columns]].sort_values(["source", "year", "title"], ascending=[True, False, True]).reset_index(drop=True)


def _distribution_table(records: pd.DataFrame, field: str) -> pd.DataFrame:
    if field not in records.columns:
        return pd.DataFrame(columns=["source", field, "records"])
    tmp = records.copy()
    tmp[field] = tmp[field].map(lambda x: _clean_value(x) or "UNKNOWN")
    table = tmp.groupby(["source", field]).size().reset_index(name="records")
    totals = tmp.groupby("source").size().rename("total")
    table = table.merge(totals, on="source", how="left")
    table["share"] = (table["records"] / table["total"]).round(4)
    return table.sort_values(["source", "records", field], ascending=[True, False, True]).reset_index(drop=True)


def _top_journal_table(records: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    tmp = records.copy()
    tmp["journal"] = tmp["journal"].map(lambda x: _clean_value(x) or "UNKNOWN")
    rows = []
    for source, group in tmp.groupby("source", sort=True):
        counts = group["journal"].value_counts().head(top_n)
        total = group.shape[0]
        for journal, count in counts.items():
            rows.append({"source": source, "journal": journal, "records": int(count), "share": round(count / total, 4) if total else 0})
    return pd.DataFrame(rows)


__all__ = ["compare_sources", "SourceComparisonResult"]
