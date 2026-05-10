"""OpenAlex integration helpers for pyBibX.

This module keeps OpenAlex-specific API access and normalization outside
``pbx.py``.  OpenAlex references are graph identifiers, not textual cited
references, so the default normalization expands cited works into readable
references while preserving the original OpenAlex IDs in ``CR_OPENALEX``.
"""

import json
import re
import time
import urllib.parse
import urllib.request
from collections import OrderedDict

import pandas as pd

OPENALEX_WORKS_API = "https://api.openalex.org/works"
OPENALEX_HOST = "https://openalex.org/"
OPENALEX_API_HOST = "https://api.openalex.org/"
OPENALEX_URL_RE = re.compile(r"(?:https?://)?(?:api\.)?openalex\.org/(?:works/)?(W\d+)", re.IGNORECASE)

OPENALEX_COLUMNS = [
    "AU", "TI", "SO", "PY", "DT", "DI", "AB", "DE", "ID", "C1", "CR",
    "CR_OPENALEX", "TC", "LA", "DB", "UT", "URL", "OA_URL",
]

PYBIBX_REQUIRED_COLUMNS = [
    'abbrev_source_title', 'abstract', 'address', 'affiliation', 'art_number',
    'author', 'author_keywords', 'chemicals_cas', 'coden', 'country', 'institution',
    'correspondence_address1', 'document_type', 'doi', 'editor', 'funding_details',
    'funding_text\xa01', 'funding_text\xa02', 'funding_text\xa03', 'isbn', 'issn',
    'journal', 'keywords', 'language', 'note', 'number', 'page_count', 'pages',
    'publisher', 'pubmed_id', 'references', 'source', 'sponsors', 'title',
    'tradenames', 'url', 'volume', 'year', 'openalex_id', 'cr_openalex', 'oa_url',
]


def _safe_str(value, default=""):
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    text = str(value)
    return text if text.strip() else default


def _json_dumps(value):
    try:
        return json.dumps(value or [], ensure_ascii=False)
    except Exception:
        return "[]"


def parse_openalex_id_list(value):
    """Parse OpenAlex reference IDs stored as list, JSON, or semicolon text."""
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_openalex_id(v) for v in value if _safe_str(v)]
    if isinstance(value, tuple):
        return [normalize_openalex_id(v) for v in value if _safe_str(v)]
    text = _safe_str(value)
    if not text or text.upper() == 'UNKNOWN':
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [normalize_openalex_id(v) for v in parsed if _safe_str(v)]
    except Exception:
        pass
    return [normalize_openalex_id(v.strip()) for v in text.split(';') if v.strip()]


def abstract_from_inverted_index(index):
    """Reconstruct a plain abstract from OpenAlex's inverted-index format."""
    if not index or not isinstance(index, dict):
        return ""
    positions = []
    for word, locs in index.items():
        if not isinstance(locs, (list, tuple)):
            continue
        for pos in locs:
            try:
                positions.append((int(pos), str(word)))
            except Exception:
                continue
    return " ".join(word for _, word in sorted(positions))


def normalize_openalex_id(value):
    """Return a canonical URL-style OpenAlex work ID when possible."""
    text = _safe_str(value).strip()
    if not text:
        return ""
    text = text.split('?')[0].rstrip('/')
    if text.lower().startswith('openalex:'):
        text = text.split(':', 1)[-1].strip()
    if text.startswith(OPENALEX_API_HOST):
        text = OPENALEX_HOST + text[len(OPENALEX_API_HOST):]
    if text.startswith('openalex.org/'):
        text = 'https://' + text
    if text.startswith('W') and text[1:].isdigit():
        return OPENALEX_HOST + text
    if '/W' in text and not text.startswith(OPENALEX_HOST):
        return OPENALEX_HOST + text.rsplit('/', 1)[-1]
    return text


def _compact_openalex_id(value):
    text = normalize_openalex_id(value)
    return text.rsplit('/', 1)[-1] if text else ""


def format_openalex_id_label(value):
    """Return the pyBibX-friendly unresolved sentinel for OpenAlex-only references."""
    return 'UNKNOWN'


def _normalize_openalex_reference_item(item):
    text = _safe_str(item).replace('\t', ' ').replace('\n', ' ').strip()
    if not text:
        return ''
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return 'UNKNOWN'
    if text.upper() == 'UNKNOWN':
        return 'UNKNOWN'

    upper = text.upper().replace(' ', '')

    # Raw OpenAlex URLs.
    if OPENALEX_URL_RE.search(text):
        return 'UNKNOWN'

    # Bare or prefixed OpenAlex work IDs.
    if re.fullmatch(r'W\d+', text, flags=re.IGNORECASE):
        return 'UNKNOWN'
    if re.fullmatch(r'OPENALEX:?W\d+', upper, flags=re.IGNORECASE):
        return 'UNKNOWN'
    if re.fullmatch(r'OPENALEXWORK:?W\d+', upper, flags=re.IGNORECASE):
        return 'UNKNOWN'

    # Any token that still embeds an OpenAlex-style work ID should not leak.
    if re.search(r'(?:^|[^A-Z0-9])W\d{6,}(?:[^A-Z0-9]|$)', upper):
        return 'UNKNOWN'

    text = text.replace(';', ',')
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = text.strip(' ,.;')
    return text or 'UNKNOWN'


def sanitize_openalex_reference_text(value):
    text = _safe_str(value)
    if not text:
        return text
    if text.upper() == 'UNKNOWN':
        return 'UNKNOWN'
    if ';' not in text:
        return _normalize_openalex_reference_item(text) or 'UNKNOWN'
    items = [_normalize_openalex_reference_item(part) for part in text.split(';')]
    items = [item for item in items if item]
    return '; '.join(items) if items else 'UNKNOWN'

def _reference_item_count(value):
    text = _safe_str(value)
    if not text or text.upper() == 'UNKNOWN':
        return 0
    return len([item for item in text.split(';') if _safe_str(item).strip()])


def rebuild_references_from_openalex_ids(ref_text, cr_openalex_value):
    """Use OpenAlex IDs as a stable fallback when expanded reference text is malformed.

    Older exports expanded one cited reference into multiple semicolon-delimited
    fragments because co-author separators reused the same delimiter that pyBibX
    uses between references. When the number of visible reference fragments no
    longer matches the number of OpenAlex reference IDs, fall back to one compact
    label per ID so parsing stays stable.
    """
    ids = parse_openalex_id_list(cr_openalex_value)
    if not ids:
        return sanitize_openalex_reference_text(ref_text)
    clean_text = sanitize_openalex_reference_text(ref_text)
    ref_count = _reference_item_count(clean_text)
    id_count = len(ids)
    if ref_count == id_count and ref_count > 0:
        return clean_text
    compact = '; '.join('UNKNOWN' for v in ids if _safe_str(v))
    return compact if compact else ('UNKNOWN' if clean_text.upper() == 'UNKNOWN' else clean_text)



def _build_url(base, params):
    clean = {k: v for k, v in params.items() if v is not None and v != ""}
    return base + "?" + urllib.parse.urlencode(clean)


def _load_json_url(url):
    req = urllib.request.Request(url, headers={"User-Agent": "pybibx-openalex/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def get_authors(work):
    authors = []
    for item in work.get('authorships', []) or []:
        name = ((item or {}).get('author') or {}).get('display_name', '')
        if name:
            authors.append(name)
    return " and ".join(authors) if authors else "UNKNOWN"


def get_affiliations(work):
    affs = []
    for item in work.get('authorships', []) or []:
        author_name = ((item or {}).get('author') or {}).get('display_name', '')
        insts = (item or {}).get('institutions', []) or []
        if not insts:
            continue
        names = []
        for inst in insts:
            inst = inst or {}
            name = inst.get('display_name', '')
            country = inst.get('country_code', '')
            if name:
                names.append(f"{name}, {country}" if country else name)
        if names:
            affs.append((author_name + " " if author_name else "") + "; ".join(names))
    return "; ".join(OrderedDict.fromkeys(affs)) if affs else "UNKNOWN"


def get_institutions(work):
    institutions = []
    for item in work.get('authorships', []) or []:
        insts = (item or {}).get('institutions', []) or []
        chosen = 'UNKNOWN'
        for inst in insts:
            inst = inst or {}
            name = inst.get('display_name', '')
            if name:
                chosen = name
                break
        institutions.append(chosen)
    return '; '.join(institutions) if institutions else 'UNKNOWN'


def get_countries(work):
    countries = []
    for item in work.get('authorships', []) or []:
        insts = (item or {}).get('institutions', []) or []
        chosen = 'UNKNOWN'
        for inst in insts:
            inst = inst or {}
            country = inst.get('country_code', '')
            if country:
                chosen = country
                break
        countries.append(chosen)
    return '; '.join(countries) if countries else 'UNKNOWN'


def get_source_name(work):
    for location_key in ('primary_location', 'best_oa_location'):
        source = ((work.get(location_key) or {}).get('source') or {})
        if source.get('display_name'):
            return source.get('display_name')
    return "UNKNOWN"


def get_keywords(work):
    terms = []
    for key in ('keywords', 'concepts', 'topics'):
        for item in work.get(key, []) or []:
            name = (item or {}).get('display_name') or (item or {}).get('keyword') or (item or {}).get('name')
            if name:
                terms.append(name)
    return "; ".join(OrderedDict.fromkeys(terms)) if terms else "UNKNOWN"


def _get_oa_url(work):
    loc = work.get('best_oa_location') or {}
    return loc.get('pdf_url') or loc.get('landing_page_url') or "UNKNOWN"


def _normalize_doi(doi):
    doi = _safe_str(doi)
    doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '')
    return doi if doi else 'UNKNOWN'


def format_openalex_reference(work):
    """Format an OpenAlex work as a human-readable cited reference."""
    if not isinstance(work, dict):
        return 'UNKNOWN'
    authors = []
    for item in work.get('authorships', []) or []:
        name = ((item or {}).get('author') or {}).get('display_name', '')
        name = _normalize_openalex_reference_item(name)
        if name and name != 'UNKNOWN':
            authors.append(name)
    if len(authors) > 6:
        author_text = ", ".join(authors[:6]) + ", et al."
    else:
        author_text = ", ".join(authors)
    year = _safe_str(work.get('publication_year'))
    title = _normalize_openalex_reference_item(work.get('display_name') or work.get('title'))
    source = _normalize_openalex_reference_item(get_source_name(work))
    doi = _normalize_doi(work.get('doi'))
    if title == 'UNKNOWN':
        return 'UNKNOWN'
    parts = [p for p in [author_text, year, title, source] if p and p != 'UNKNOWN']
    if doi != 'UNKNOWN':
        parts.append('doi:' + doi)
    ref = sanitize_openalex_reference_text(". ".join(parts).strip())
    if 'OPENALEX' in ref.upper() or re.search(r'W\d{6,}', ref, flags=re.IGNORECASE):
        return 'UNKNOWN'
    return ref if ref and ref != 'UNKNOWN' else 'UNKNOWN'


def get_work(openalex_id, mailto=None):
    work_id = _compact_openalex_id(openalex_id)
    if not work_id:
        return None
    params = {}
    if mailto:
        params['mailto'] = mailto
    url = OPENALEX_WORKS_API + '/' + urllib.parse.quote(work_id)
    if params:
        url = _build_url(url, params)
    return _load_json_url(url)


def get_works_batch(openalex_ids, mailto=None):
    """Fetch works by OpenAlex ID using the API filter endpoint.

    Falls back to an empty list on unrecoverable batch errors. Callers can
    use ``get_work`` sequentially for missing items.
    """
    ids = [_compact_openalex_id(x) for x in openalex_ids]
    ids = [x for x in OrderedDict.fromkeys(ids) if x]
    if not ids:
        return []
    out = []
    # Keep chunks conservative to avoid long URLs.
    for start in range(0, len(ids), 50):
        chunk = ids[start:start + 50]
        params = {
            'filter': 'openalex_id:' + '|'.join(chunk),
            'per-page': min(200, len(chunk)),
        }
        if mailto:
            params['mailto'] = mailto
        try:
            payload = _load_json_url(_build_url(OPENALEX_WORKS_API, params))
            out.extend(payload.get('results', []) or [])
        except Exception:
            continue
    return out


def expand_references_for_works(works, mailto=None, max_refs_per_work=None, reference_cache=None):
    reference_cache = reference_cache if reference_cache is not None else {}
    wanted = []
    for work in works or []:
        refs = [normalize_openalex_id(r) for r in (work.get('referenced_works', []) or []) if r]
        if max_refs_per_work is not None:
            refs = refs[:max_refs_per_work]
        for ref in refs:
            if ref and ref not in reference_cache:
                wanted.append(ref)
    wanted = list(OrderedDict.fromkeys(wanted))
    if wanted:
        try:
            for ref_work in get_works_batch(wanted, mailto=mailto):
                ref_id = normalize_openalex_id(ref_work.get('id'))
                if ref_id:
                    reference_cache[ref_id] = ref_work
        except Exception:
            pass
    # Sequential fallback for missing references. Failures store None so they
    # are not retried endlessly during one normalization call.
    for ref in wanted:
        if ref in reference_cache:
            continue
        try:
            reference_cache[ref] = get_work(ref, mailto=mailto)
        except Exception:
            reference_cache[ref] = None
    return reference_cache


def works_to_dataframe(works, expand_references=True, mailto=None, max_refs_per_work=None, reference_cache=None):
    works = works or []
    reference_cache = reference_cache if reference_cache is not None else {}
    if expand_references:
        expand_references_for_works(
            works,
            mailto=mailto,
            max_refs_per_work=max_refs_per_work,
            reference_cache=reference_cache,
        )
    rows = []
    for work in works:
        refs = [normalize_openalex_id(r) for r in (work.get('referenced_works', []) or []) if r]
        if max_refs_per_work is not None:
            refs = refs[:max_refs_per_work]
        if expand_references:
            ref_texts = []
            for ref in refs:
                ref_work = reference_cache.get(ref)
                ref_texts.append(format_openalex_reference(ref_work) if ref_work else 'UNKNOWN')
            cr_text = sanitize_openalex_reference_text("; ".join([r for r in ref_texts if r]) or "UNKNOWN")
        else:
            cr_text = "; ".join(refs) if refs else "UNKNOWN"
        title = work.get('display_name') or work.get('title') or 'UNKNOWN'
        source_name = get_source_name(work)
        year = _safe_str(work.get('publication_year'), 'UNKNOWN')
        doc_type = _safe_str(work.get('type'), 'UNKNOWN')
        doi = _normalize_doi(work.get('doi'))
        abstract = abstract_from_inverted_index(work.get('abstract_inverted_index')) or 'UNKNOWN'
        keywords = get_keywords(work)
        authors = get_authors(work)
        affs = get_affiliations(work)
        institutions = get_institutions(work)
        countries = get_countries(work)
        cited_by = _safe_str(work.get('cited_by_count'), '0')
        language = _safe_str(work.get('language'), 'UNKNOWN')
        openalex_id = normalize_openalex_id(work.get('id')) or 'UNKNOWN'
        oa_url = _get_oa_url(work)
        row = {
            # Human-friendly/OpenAlex export aliases requested by the API.
            'AU': authors,
            'TI': title,
            'SO': source_name,
            'PY': year,
            'DT': doc_type,
            'DI': doi,
            'AB': abstract,
            'DE': keywords,
            'ID': keywords,
            'C1': affs,
            'CR': cr_text,
            'CR_OPENALEX': _json_dumps(refs),
            'TC': cited_by,
            'LA': language,
            'DB': 'OpenAlex',
            'UT': openalex_id,
            'URL': openalex_id,
            'OA_URL': oa_url,
            # Internal pyBibX columns consumed by pbx.py.
            'author': authors,
            'title': title,
            'abbrev_source_title': source_name,
            'journal': source_name,
            'year': year,
            'document_type': doc_type,
            'doi': doi,
            'abstract': abstract,
            'keywords': keywords,
            'author_keywords': keywords,
            'affiliation': affs,
            'address': affs,
            'institution': institutions,
            'country': countries,
            'correspondence_address1': 'UNKNOWN',
            'references': cr_text,
            'cr_openalex': _json_dumps(refs),
            'note': cited_by,
            'language': language,
            'source': 'openalex',
            'url': openalex_id,
            'openalex_id': openalex_id,
            'oa_url': oa_url,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return normalize_openalex_dataframe(df)


def _looks_like_openalex_standard_csv(df):
    cols = set(str(c) for c in df.columns)
    strong = {
        'id',
        'display_name',
        'publication_year',
        'type',
        'doi',
        'abstract',
        'cited_by_count',
        'primary_location.source.display_name',
    }
    return len(cols.intersection(strong)) >= 5


def _looks_like_openalex_normalized(df):
    cols = set(str(c) for c in df.columns)
    normalized = {
        'author', 'title', 'journal', 'year', 'doi',
        'openalex_id', 'references', 'cr_openalex'
    }
    compact = {'AU', 'TI', 'SO', 'PY', 'DI', 'UT'}
    return (len(cols.intersection(normalized)) >= 4) or (len(cols.intersection(compact)) >= 4)


def _series_or_default(df, col, default='UNKNOWN'):
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _clean_pipe_like(value):
    if value is None:
        return 'UNKNOWN'
    if isinstance(value, float) and pd.isna(value):
        return 'UNKNOWN'
    text = str(value).strip()
    if not text:
        return 'UNKNOWN'
    return text.replace('|', '; ')


def _first_existing_series(df, columns, default='UNKNOWN'):
    for col in columns:
        if col in df.columns:
            return df[col]
    return pd.Series([default] * len(df), index=df.index)


def adapt_openalex_standard_csv(df):
    norm = pd.DataFrame(index=df.index)

    norm['author'] = _series_or_default(df, 'authorships.author.display_name').apply(_clean_pipe_like)
    norm['title'] = _series_or_default(df, 'display_name')
    norm['journal'] = _series_or_default(df, 'primary_location.source.display_name')
    norm['abbrev_source_title'] = norm['journal']

    if 'publication_year' in df.columns:
        year = pd.to_numeric(df['publication_year'], errors='coerce').fillna(0).astype(int).astype(str)
        year = year.replace('0', 'UNKNOWN')
        norm['year'] = year
    else:
        norm['year'] = 'UNKNOWN'

    norm['document_type'] = _series_or_default(df, 'type')
    norm['doi'] = _series_or_default(df, 'doi').apply(_normalize_doi)
    norm['abstract'] = _series_or_default(df, 'abstract')
    norm['institution'] = _series_or_default(df, 'authorships.institutions.display_name').apply(_clean_pipe_like)
    norm['country'] = _series_or_default(df, 'authorships.countries').apply(_clean_pipe_like)
    norm['affiliation'] = norm['institution']
    norm['address'] = norm['affiliation']
    norm['language'] = _series_or_default(df, 'language')
    norm['note'] = _series_or_default(df, 'cited_by_count', default='0').astype(str)

    norm['openalex_id'] = _series_or_default(df, 'id')
    norm['url'] = norm['openalex_id']
    norm['oa_url'] = _first_existing_series(
        df,
        ['best_oa_location.pdf_url', 'best_oa_location.landing_page_url', 'open_access.oa_url'],
        default='UNKNOWN'
    )

    refs = _first_existing_series(df, ['referenced_works', 'cr_openalex'], default='UNKNOWN')
    norm['cr_openalex'] = refs.apply(
        lambda x: _json_dumps(parse_openalex_id_list(x)) if _safe_str(x).upper() != 'UNKNOWN' else 'UNKNOWN'
    )
    norm['references'] = norm['cr_openalex'].apply(
        lambda x: sanitize_openalex_reference_text('; '.join(format_openalex_id_label(v) for v in parse_openalex_id_list(x)) or 'UNKNOWN')
    )

    topic = _first_existing_series(df, ['primary_topic.display_name', 'keywords.display_name'], default='UNKNOWN')
    norm['author_keywords'] = topic
    norm['keywords'] = topic
    norm['source'] = 'openalex'

    return normalize_openalex_dataframe(norm, auto_detect=False)


def normalize_openalex_dataframe(data, auto_detect=True):
    """Return a DataFrame with pyBibX internal columns for OpenAlex data."""
    df = data.copy(deep=True) if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    if auto_detect and len(df.columns) > 0:
        if _looks_like_openalex_standard_csv(df):
            return adapt_openalex_standard_csv(df)
        if _looks_like_openalex_normalized(df):
            pass

    rename = {
        'AU': 'author', 'TI': 'title', 'SO': 'abbrev_source_title', 'PY': 'year',
        'DT': 'document_type', 'DI': 'doi', 'AB': 'abstract', 'DE': 'author_keywords',
        'ID': 'keywords', 'C1': 'affiliation', 'CU': 'country', 'C2': 'institution',
        'CR': 'references', 'TC': 'note', 'LA': 'language', 'URL': 'url',
        'UT': 'openalex_id', 'OA_URL': 'oa_url', 'CR_OPENALEX': 'cr_openalex',
    }
    for src, dst in rename.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    if 'journal' not in df.columns and 'abbrev_source_title' in df.columns:
        df['journal'] = df['abbrev_source_title']
    if 'address' not in df.columns and 'affiliation' in df.columns:
        df['address'] = df['affiliation']
    if 'institution' not in df.columns and 'affiliation' in df.columns:
        df['institution'] = df['affiliation']
    if 'source' not in df.columns:
        df['source'] = 'openalex'
    if 'url' not in df.columns and 'openalex_id' in df.columns:
        df['url'] = df['openalex_id']
    if 'references' not in df.columns and 'cr_openalex' in df.columns:
        df['references'] = df['cr_openalex'].apply(lambda x: '; '.join(format_openalex_id_label(v) for v in parse_openalex_id_list(x)) or 'UNKNOWN')
    if 'cr_openalex' not in df.columns and 'references' in df.columns:
        df['cr_openalex'] = df['references'].apply(lambda x: _json_dumps(parse_openalex_id_list(x)))
    if 'references' in df.columns and 'cr_openalex' in df.columns:
        df['references'] = [
            rebuild_references_from_openalex_ids(ref_text, ref_ids)
            for ref_text, ref_ids in zip(df['references'], df['cr_openalex'])
        ]
    for ref_col in ('references', 'CR'):
        if ref_col in df.columns:
            paired_ids = df['cr_openalex'] if 'cr_openalex' in df.columns else pd.Series(['UNKNOWN'] * len(df), index=df.index)
            df[ref_col] = [
                rebuild_references_from_openalex_ids(ref_text, ref_ids)
                for ref_text, ref_ids in zip(df[ref_col], paired_ids)
            ]
    for col in PYBIBX_REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = 'UNKNOWN'
    # Recreate uppercase aliases for export/readability without removing internal columns.
    aliases = {
        'author': 'AU', 'title': 'TI', 'abbrev_source_title': 'SO', 'year': 'PY',
        'document_type': 'DT', 'doi': 'DI', 'abstract': 'AB', 'author_keywords': 'DE',
        'keywords': 'ID', 'affiliation': 'C1', 'country': 'CU', 'institution': 'C2',
        'references': 'CR', 'cr_openalex': 'CR_OPENALEX', 'note': 'TC', 'language': 'LA',
        'openalex_id': 'UT', 'url': 'URL', 'oa_url': 'OA_URL',
    }
    for src, dst in aliases.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    if 'DB' not in df.columns:
        df['DB'] = 'OpenAlex'
    df = df.fillna('UNKNOWN')
    for col in ('doi', 'DI'):
        if col in df.columns:
            df[col] = df[col].apply(_normalize_doi)
    if 'author' in df.columns:
        df['author'] = df['author'].apply(lambda x: x.replace(';', ' and ') if isinstance(x, str) else x)
    return df


def search_works(query, from_year=None, to_year=None, max_results=1000, per_page=200, mailto=None, filter=None, sort=None, sleep=0.1):
    per_page = max(1, min(int(per_page), 200))
    filters = []
    if filter:
        filters.append(filter)
    if from_year is not None:
        filters.append(f"from_publication_date:{int(from_year)}-01-01")
    if to_year is not None:
        filters.append(f"to_publication_date:{int(to_year)}-12-31")
    params = {
        'search': query,
        'per-page': per_page,
        'cursor': '*',
    }
    if filters:
        params['filter'] = ','.join(filters)
    if sort:
        params['sort'] = sort
    if mailto:
        params['mailto'] = mailto
    works = []
    cursor = '*'
    while len(works) < max_results:
        params['cursor'] = cursor
        payload = _load_json_url(_build_url(OPENALEX_WORKS_API, params))
        batch = payload.get('results', []) or []
        works.extend(batch)
        cursor = (payload.get('meta') or {}).get('next_cursor')
        if not cursor or not batch:
            break
        if sleep:
            time.sleep(sleep)
    return works[:max_results]


def search_to_dataframe(query, from_year=None, to_year=None, max_results=1000, per_page=200, mailto=None, filter=None, sort=None, sleep=0.1, expand_references=True, max_refs_per_work=None, reference_cache=None):
    works = search_works(
        query=query,
        from_year=from_year,
        to_year=to_year,
        max_results=max_results,
        per_page=per_page,
        mailto=mailto,
        filter=filter,
        sort=sort,
        sleep=sleep,
    )
    return works_to_dataframe(
        works,
        expand_references=expand_references,
        mailto=mailto,
        max_refs_per_work=max_refs_per_work,
        reference_cache=reference_cache,
    )


def load_openalex_json(path, expand_references=True, mailto=None, max_refs_per_work=None, reference_cache=None):
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if isinstance(payload, dict) and 'results' in payload:
        works = payload.get('results') or []
    elif isinstance(payload, dict) and 'id' in payload:
        works = [payload]
    elif isinstance(payload, list):
        works = payload
    else:
        works = []
    return works_to_dataframe(
        works,
        expand_references=expand_references,
        mailto=mailto,
        max_refs_per_work=max_refs_per_work,
        reference_cache=reference_cache,
    )



def load_openalex_auto(path, expand_references=True, mailto=None, max_refs_per_work=None, reference_cache=None):
    file_extension = path.rsplit('.', 1)[-1].lower() if '.' in path else ''
    if file_extension == 'json':
        return load_openalex_json(
            path,
            expand_references=expand_references,
            mailto=mailto,
            max_refs_per_work=max_refs_per_work,
            reference_cache=reference_cache,
        )

    read_kwargs = {'dtype': str}
    if file_extension in ['tsv', 'txt']:
        read_kwargs['sep'] = '	'
    else:
        read_kwargs['sep'] = None
        read_kwargs['engine'] = 'python'

    df = pd.read_csv(path, **read_kwargs)
    return normalize_openalex_dataframe(df, auto_detect=True)
