############################################################################
# Created for: PyBibX — Temporal Scholarly Graph Explorer
# Author:      Valdecy Pereira, D.Sc.
# Module:      pybibx/base/tsg.py
#
#
# Public API:
#   temporal_sg(pbx_instance, **kwargs)  ->  result dict
############################################################################

from __future__ import annotations

import hashlib
import json
import os
import re
import textwrap
import unicodedata
import webbrowser
from collections  import Counter, defaultdict
from dataclasses  import dataclass, field
from typing       import Any, Dict, List, Optional, Set, Tuple

import numpy  as np
import pandas as pd

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

try:
    from IPython.display import IFrame, display as _ipy_display
    _IPY_AVAILABLE = True
except ImportError:
    _IPY_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# §1  Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TSGConfig:
    view:                   str             = "timeline"
    layers:                 List[str]       = field(default_factory=lambda: ["citations"])
    time_mode:              str             = "range"
    start_year:             Optional[int]   = None
    end_year:               Optional[int]   = None
    center:                 str             = "paper"
    selected:               Optional[str]   = None
    max_papers:             int             = 500
    max_references:         int             = 300
    color_by:               str             = "type"
    size_by:                str             = "citations"
    notebook:               bool            = True
    open_browser:           bool            = True
    save_html:              Optional[str]   = None
    preview:                bool            = True

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ──────────────────────────────────────────────────────────────────────────────
# §2  Data Extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_list(val) -> List:
    """Ensure a value is a (possibly empty) list."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return list(val)


def _safe_str(val, default: str = "UNKNOWN") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    s = str(val).strip()
    return s if s else default


def _norm_title(title: str) -> str:
    """Normalise a title for fuzzy matching."""
    t = unicodedata.normalize("NFKD", title.lower())
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _ref_year_from_string(ref_str: str) -> Optional[int]:
    """Try to parse a publication year from a raw reference string."""
    m = re.search(r"\b(19|20)\d{2}\b", ref_str)
    if m:
        return int(m.group())
    return None


def _ref_first_author(ref_str: str) -> str:
    """Heuristic: first token before a comma or year."""
    parts = re.split(r",|\.", ref_str)
    return parts[0].strip().lower() if parts else ""


def _short_label(text: str, maxlen: int = 40) -> str:
    """Truncate with ellipsis for display."""
    if len(text) <= maxlen:
        return text
    return text[:maxlen - 1] + "…"


def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    """Return the first column name that exists in df, or None."""
    for n in names:
        if n in df.columns:
            return n
    return None


# ──────────────────────────────────────────────────────────────────────────────
# §3  Reference matching (DOI > title > string)
# ──────────────────────────────────────────────────────────────────────────────

def _build_internal_ref_map(
    data: pd.DataFrame,
    u_ref: List[str],
) -> Dict[str, int]:
    """
    Return {ref_string: paper_index} for every cited reference that can be
    matched to a paper inside the dataset.

    Match priority:
      1. Exact DOI match (if data has a 'doi' column and ref contains a DOI).
      2. Normalised-title match (≥ 0.85 similarity via shared trigrams).
      3. First-author + year substring match.
    """
    matched: Dict[str, int] = {}

    doi_col   = _col(data, "doi", "DOI")
    title_col = _col(data, "title", "Title", "TI")

    # Build lookup tables from dataset papers
    doi_to_idx:   Dict[str, int] = {}
    title_to_idx: Dict[str, int] = {}
    fy_to_idx:    Dict[Tuple[str, str], int] = {}   # (first_author_token, year)

    for idx, row in data.iterrows():
        if doi_col:
            doi = _safe_str(row.get(doi_col, ""), "").lower().strip()
            if doi and doi != "unknown":
                doi_to_idx[doi] = idx
        if title_col:
            t = _norm_title(_safe_str(row.get(title_col, ""), ""))
            if t and t != "unknown":
                title_to_idx[t] = idx
        year = _safe_str(row.get("year", ""), "0")
        aut  = _safe_str(row.get("author", ""), "").split(" and ")[0][:20].lower().strip()
        if aut and year and year != "0":
            fy_to_idx[(aut[:6], year)] = idx

    def _trigrams(s: str) -> Set[str]:
        return {s[i:i+3] for i in range(len(s) - 2)} if len(s) > 2 else set()

    for ref_str in u_ref:
        if not ref_str or ref_str.lower() == "unknown":
            continue

        # — DOI match ————————————————————————————————————
        doi_m = re.search(r"10\.\d{4,9}/[^\s;,]+", ref_str, re.I)
        if doi_m:
            doi_cand = doi_m.group().lower().rstrip(".")
            if doi_cand in doi_to_idx:
                matched[ref_str] = doi_to_idx[doi_cand]
                continue

        # — Title trigram match ——————————————————————————
        if title_col:
            ref_norm = _norm_title(ref_str)
            ref_tri  = _trigrams(ref_norm)
            best_sim, best_idx = 0.0, -1
            for t_key, t_idx in title_to_idx.items():
                if abs(len(ref_norm) - len(t_key)) > 60:
                    continue
                t_tri = _trigrams(t_key)
                if not t_tri or not ref_tri:
                    continue
                sim = len(ref_tri & t_tri) / len(ref_tri | t_tri)
                if sim > best_sim:
                    best_sim, best_idx = sim, t_idx
            if best_sim >= 0.70 and best_idx >= 0:
                matched[ref_str] = best_idx
                continue

        # — First-author + year match ————————————————————
        year_cand = _ref_year_from_string(ref_str)
        fa_cand   = _ref_first_author(ref_str)
        if year_cand and fa_cand:
            key = (fa_cand[:6], str(year_cand))
            if key in fy_to_idx:
                matched[ref_str] = fy_to_idx[key]

    return matched


# ──────────────────────────────────────────────────────────────────────────────
# §4  Graph Construction
# ──────────────────────────────────────────────────────────────────────────────

def _stable_ref_id(ref_str: str) -> str:
    """Deterministic short ID for an external reference string."""
    h = hashlib.md5(ref_str.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"ref_{h}"


def _normalize_center(center: str) -> str:
    """Normalise user-provided centre aliases to supported node types."""
    c = _safe_str(center, "paper").lower().strip()
    alias = {
        # paper
        "paper": "paper",
        "document": "paper",
        "doc": "paper",
        "docs": "paper",

        # reference
        "reference": "reference",
        "references": "reference",
        "ref": "reference",
        "refs": "reference",

        # author
        "author": "author",
        "authors": "author",
        "aut": "author",

        # journal
        "journal": "journal",
        "journals": "journal",
        "source": "journal",
        "sources": "journal",
        "jou": "journal",

        # institution
        "institution": "institution",
        "institutions": "institution",
        "inst": "institution",
        "affiliation": "institution",
        "affiliations": "institution",
        "university": "institution",
        "universities": "institution",

        # country
        "country": "country",
        "countries": "country",
        "cout": "country",

        # author keywords
        "author_keyword": "author_keyword",
        "author_keywords": "author_keyword",
        "authors_keyword": "author_keyword",
        "authors_keywords": "author_keyword",
        "kwa": "author_keyword",
        "keyword": "author_keyword",
        "keywords": "author_keyword",

        # keywords plus
        "keyword_plus": "keyword_plus",
        "keywords_plus": "keyword_plus",
        "kw_plus": "keyword_plus",
        "kwp": "keyword_plus",
    }
    return alias.get(c, "paper")


def _table_value_to_id(table: Optional[pd.DataFrame], value_col: str) -> Dict[str, str]:
    if table is None or not isinstance(table, pd.DataFrame):
        return {}
    if "ID" not in table.columns or value_col not in table.columns:
        return {}
    out: Dict[str, str] = {}
    for _, row in table.iterrows():
        val = _safe_str(row.get(value_col, ""), "")
        if val:
            out[val] = str(row["ID"])
    return out


def _table_index_to_id(table: Optional[pd.DataFrame]) -> Dict[int, str]:
    if table is None or not isinstance(table, pd.DataFrame) or "ID" not in table.columns:
        return {}
    return {int(i): str(table.iloc[i]["ID"]) for i in range(len(table))}


def _center_match(node_type: str, center: str) -> bool:
    if center == "reference":
        return node_type in ("reference", "internal_ref")
    return node_type == center


def build_temporal_sg(pbx, cfg: TSGConfig):
    """
    Build the heterogeneous Temporal Scholarly Graph from a PyBibX object.

    The graph reuses the object identifiers already created by ``pbx``
    (documents, authors, journals, institutions, countries, references,
    authors' keywords, and keywords plus) so that the visual workspace stays
    aligned with the rest of the package.
    """
    cfg.center = _normalize_center(cfg.center)

    data   = pbx.data.copy().reset_index(drop=True)
    n_docs = data.shape[0]

    dy     = getattr(pbx, "dy",       pd.Series(dtype="float64"))
    cit    = getattr(pbx, "citation", [0] * n_docs)
    ref    = getattr(pbx, "ref",      [[] for _ in range(n_docs)])
    aut    = getattr(pbx, "aut",      [[] for _ in range(n_docs)])
    kid    = getattr(pbx, "kid",      [[] for _ in range(n_docs)])
    auk    = getattr(pbx, "auk",      [[] for _ in range(n_docs)])
    jou    = getattr(pbx, "jou",      [[] for _ in range(n_docs)])
    ctr    = getattr(pbx, "ctr",      [[] for _ in range(n_docs)])
    uni    = getattr(pbx, "uni",      [[] for _ in range(n_docs)])
    u_ref  = getattr(pbx, "u_ref",    [])
    topics = getattr(pbx, "topics",   None)

    def _pad(lst, length, default=None):
        lst = list(lst) if not isinstance(lst, list) else lst
        while len(lst) < length:
            lst.append([] if default is None else default)
        return lst

    ref = _pad(ref, n_docs)
    aut = _pad(aut, n_docs)
    kid = _pad(kid, n_docs)
    auk = _pad(auk, n_docs)
    jou = _pad(jou, n_docs)
    ctr = _pad(ctr, n_docs)
    uni = _pad(uni, n_docs)
    cit = _pad(list(cit), n_docs, 0)

    years = pd.Series(pd.to_numeric(dy, errors="coerce")).reset_index(drop=True)
    y_min = int(years.min()) if not years.empty and not years.isna().all() else 0
    y_max = int(years.max()) if not years.empty and not years.isna().all() else 9999
    start = cfg.start_year or y_min
    end   = cfg.end_year   or y_max

    active_idx = []
    for i in range(n_docs):
        yr = years.iloc[i] if i < len(years) else np.nan
        if not np.isnan(yr) and start <= int(yr) <= end:
            active_idx.append(i)

    if len(active_idx) > cfg.max_papers:
        ranked = sorted(active_idx, key=lambda i: cit[i], reverse=True)
        active_set = set(ranked[:cfg.max_papers])
    else:
        active_set = set(active_idx)

    doc_id_by_idx = _table_index_to_id(getattr(pbx, "table_id_doc", None))
    aut_id_by_val = getattr(pbx, "dict_aut_id", None) or _table_value_to_id(getattr(pbx, "table_id_aut", None), "Author")
    jou_id_by_val = getattr(pbx, "dict_jou_id", None) or _table_value_to_id(getattr(pbx, "table_id_jou", None), "Source")
    uni_id_by_val = getattr(pbx, "dict_uni_id", None) or _table_value_to_id(getattr(pbx, "table_id_uni", None), "Institution")
    ctr_id_by_val = getattr(pbx, "dict_ctr_id", None) or _table_value_to_id(getattr(pbx, "table_id_ctr", None), "Country")
    kwa_id_by_val = getattr(pbx, "dict_kwa_id", None) or _table_value_to_id(getattr(pbx, "table_id_kwa", None), "KWA")
    kwp_id_by_val = getattr(pbx, "dict_kwp_id", None) or _table_value_to_id(getattr(pbx, "table_id_kwp", None), "KWP")
    ref_id_by_val = dict(zip(getattr(pbx, "u_ref", []), getattr(pbx, "u_ref_id", [])))

    ref_freq: Counter = Counter()
    for i in active_set:
        for r in ref[i]:
            if r and _safe_str(r, "").lower() != "unknown":
                ref_freq[r] += 1

    internal_map = _build_internal_ref_map(data, list(u_ref))
    internal_paper_ids: Set[int] = set(internal_map.values())

    if topics is not None and len(topics) >= n_docs:
        topic_list = [int(t) for t in topics[:n_docs]]
    else:
        topic_list = [-1] * n_docs

    doi_col   = _col(data, "doi", "DOI")
    title_col = _col(data, "title", "Title", "TI")
    ab_col    = _col(data, "abstract", "Abstract", "AB")

    nodes: List[Dict] = []
    edges: List[Dict] = []
    node_ids_seen: Set[str] = set()

    def _add_node(d: Dict):
        if d["id"] not in node_ids_seen:
            node_ids_seen.add(d["id"])
            nodes.append(d)

    def _paper_year(i: int) -> int:
        yr = years.iloc[i] if i < len(years) else np.nan
        return int(yr) if not np.isnan(yr) else 0

    def _entity_stats(doc_idx: List[int]) -> Tuple[int, int, int, int]:
        valid = sorted({i for i in doc_idx if i in active_set})
        if not valid:
            return 0, 0, 0, 0
        yrs = [_paper_year(i) for i in valid if _paper_year(i) > 0]
        first_year = min(yrs) if yrs else 0
        last_year  = max(yrs) if yrs else 0
        return first_year, last_year, len(valid), int(sum(cit[i] for i in valid))

    paper_id_map: Dict[int, str] = {}
    author_docs: Dict[str, List[int]] = defaultdict(list)
    journal_docs: Dict[str, List[int]] = defaultdict(list)
    institution_docs: Dict[str, List[int]] = defaultdict(list)
    country_docs: Dict[str, List[int]] = defaultdict(list)
    kwa_docs: Dict[str, List[int]] = defaultdict(list)
    kwp_docs: Dict[str, List[int]] = defaultdict(list)

    for i in active_set:
        row = data.iloc[i]
        pid = str(doc_id_by_idx.get(i, i))
        paper_id_map[i] = pid
        year = _paper_year(i)
        p_doi = _safe_str(row.get(doi_col, "") if doi_col else "", "")
        p_tit = _safe_str(row.get(title_col, "") if title_col else "", f"Paper {i}")

        auth = [a for a in _safe_list(aut[i]) if a and a != "unknown"]
        jous = [j for j in _safe_list(jou[i]) if j and j != "unknown"]
        unis = [u for u in _safe_list(uni[i]) if u and u != "unknown"]
        ctrs = [c for c in _safe_list(ctr[i]) if c and c != "unknown"]
        kws_a = [k for k in _safe_list(auk[i]) if k and k != "unknown"]
        kws_p = [k for k in _safe_list(kid[i]) if k and k != "unknown"]
        refs = [r for r in _safe_list(ref[i]) if r and _safe_str(r, "").lower() != "unknown"]
        n_int_refs = sum(1 for r in refs if r in internal_map)
        ab_short = _short_label(_safe_str(row.get(ab_col, "") if ab_col else "", ""), 300)

        for a in auth: author_docs[a].append(i)
        for j in jous: journal_docs[j].append(i)
        for u in unis: institution_docs[u].append(i)
        for c in ctrs: country_docs[c].append(i)
        for k in kws_a: kwa_docs[k].append(i)
        for k in kws_p: kwp_docs[k].append(i)

        _add_node({
            "id":               pid,
            "label":            _short_label(p_tit, 60),
            "full_title":       p_tit,
            "type":             "paper",
            "year":             year,
            "first_year":       year,
            "last_year":        year,
            "paper_count":      1,
            "citations":        int(cit[i]),
            "doi":              p_doi,
            "authors":          auth,
            "author_keywords":  kws_a[:10],
            "keyword_plus":     kws_p[:10],
            "keywords":         list(dict.fromkeys(kws_a + kws_p))[:12],
            "journals":         jous[:3],
            "countries":        ctrs[:8],
            "institutions":     unis[:8],
            "references":       refs,
            "topic":            topic_list[i],
            "internal_refs":    n_int_refs,
            "external_refs":    len(refs) - n_int_refs,
            "author_count":     len(auth),
            "keyword_count":    len(set(kws_a + kws_p)),
            "abstract":         ab_short,
            "size_val":         max(5, min(40, 5 + np.log1p(max(0, cit[i])) * 5)),
            "match_method":     "dataset",
            "group":            topic_list[i],
            "is_internal_reference_target": i in internal_paper_ids,
            "is_center":        _center_match("paper", cfg.center),
        })

    # Internal threshold: a reference must appear in ≥2 dataset papers to
    # earn its own node. (Used to be exposed as min_reference_citations; the
    # parameter is gone but the safeguard stays so single-occurrence refs
    # don't flood the graph with one-shot nodes.)
    _MIN_REF_CIT = 2
    top_refs = sorted(
        [(r, cnt) for r, cnt in ref_freq.items() if cnt >= _MIN_REF_CIT],
        key=lambda x: -x[1]
    )[:cfg.max_references]
    top_ref_set = {r for r, _ in top_refs}
    ref_id_map: Dict[str, str] = {}
    ref_docs: Dict[str, List[int]] = defaultdict(list)
    for i in active_set:
        for r in _safe_list(ref[i]):
            if r in top_ref_set:
                ref_docs[r].append(i)

    for ref_str, freq in top_refs:
        is_internal = ref_str in internal_map
        if is_internal:
            paper_idx = internal_map[ref_str]
            if paper_idx in active_set:
                ref_id_map[ref_str] = paper_id_map[paper_idx]
                continue

        rid = str(ref_id_by_val.get(ref_str, _stable_ref_id(ref_str)))
        ref_id_map[ref_str] = rid
        first_year, last_year, paper_count, citation_sum = _entity_stats(ref_docs.get(ref_str, []))
        ref_year = _ref_year_from_string(ref_str) or first_year or (start - 5)
        _add_node({
            "id":            rid,
            "label":         _short_label(ref_str, 60),
            "full_title":    ref_str,
            "type":          "internal_ref" if is_internal else "reference",
            "year":          ref_year,
            "first_year":    first_year,
            "last_year":     last_year,
            "paper_count":   paper_count or int(freq),
            "citations":     int(freq),
            "citation_sum":  citation_sum,
            "doi":           "",
            "authors":       [],
            "keywords":      [],
            "journals":      [],
            "countries":     [],
            "institutions":  [],
            "references":    [],
            "topic":         -1,
            "internal_refs": 0,
            "external_refs": 0,
            "author_count":  0,
            "keyword_count": 0,
            "abstract":      "",
            "size_val":      max(4, min(20, 4 + np.log1p(max(1, freq)) * 3)),
            "match_method":  "doi_title_or_author" if is_internal else "dataset",
            "group":         -2 if not is_internal else -3,
            "is_internal_reference_target": False,
            "is_center":     _center_match("reference", cfg.center),
        })

    def _node_payload(node_id: str, label: str, node_type: str, doc_idx: List[int], group: int, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        first_year, last_year, paper_count, citation_sum = _entity_stats(doc_idx)
        payload = {
            "id":            node_id,
            "label":         _short_label(label, 40),
            "full_title":    label,
            "type":          node_type,
            "year":          first_year,
            "first_year":    first_year,
            "last_year":     last_year,
            "paper_count":   paper_count,
            "citations":     citation_sum,
            "doi":           "",
            "authors":       [],
            "keywords":      [],
            "journals":      [],
            "countries":     [],
            "institutions":  [],
            "references":    [],
            "topic":         -1,
            "internal_refs": 0,
            "external_refs": 0,
            "author_count":  0,
            "keyword_count": 0,
            "abstract":      "",
            "size_val":      max(4, min(20, 4 + np.log1p(max(1, paper_count)) * 3.2)),
            "match_method":  "dataset",
            "group":         group,
            "is_internal_reference_target": False,
            "is_center":     _center_match(node_type, cfg.center),
        }
        if extra:
            payload.update(extra)
        return payload

    for a_name, doc_idx in author_docs.items():
        aid = str(aut_id_by_val.get(a_name, "aut_" + hashlib.md5(a_name.encode()).hexdigest()[:8]))
        _add_node(_node_payload(aid, a_name, "author", doc_idx, -4, {"author_count": 1}))
        for i in sorted(set(doc_idx)):
            edges.append({"source": aid, "target": paper_id_map[i], "edge_type": "wrote", "weight": 1, "year": _paper_year(i)})

    for j_name, doc_idx in journal_docs.items():
        jid = str(jou_id_by_val.get(j_name, "jou_" + hashlib.md5(j_name.encode()).hexdigest()[:8]))
        _add_node(_node_payload(jid, j_name, "journal", doc_idx, -6))
        for i in sorted(set(doc_idx)):
            edges.append({"source": paper_id_map[i], "target": jid, "edge_type": "published_in", "weight": 1, "year": _paper_year(i)})

    for u_name, doc_idx in institution_docs.items():
        uid = str(uni_id_by_val.get(u_name, "ins_" + hashlib.md5(u_name.encode()).hexdigest()[:8]))
        _add_node(_node_payload(uid, u_name, "institution", doc_idx, -7))
        for i in sorted(set(doc_idx)):
            edges.append({"source": paper_id_map[i], "target": uid, "edge_type": "has_institution", "weight": 1, "year": _paper_year(i)})

    for c_name, doc_idx in country_docs.items():
        cid = str(ctr_id_by_val.get(c_name, "ctr_" + hashlib.md5(c_name.encode()).hexdigest()[:8]))
        _add_node(_node_payload(cid, c_name, "country", doc_idx, -8))
        for i in sorted(set(doc_idx)):
            edges.append({"source": paper_id_map[i], "target": cid, "edge_type": "has_country", "weight": 1, "year": _paper_year(i)})

    for kw, doc_idx in kwa_docs.items():
        kid_id = str(kwa_id_by_val.get(kw, "kwa_" + hashlib.md5(kw.encode()).hexdigest()[:8]))
        _add_node(_node_payload(kid_id, kw, "author_keyword", doc_idx, -5))
        for i in sorted(set(doc_idx)):
            edges.append({"source": paper_id_map[i], "target": kid_id, "edge_type": "has_author_keyword", "weight": 1, "year": _paper_year(i)})

    for kw, doc_idx in kwp_docs.items():
        kwp_id = str(kwp_id_by_val.get(kw, "kwp_" + hashlib.md5(kw.encode()).hexdigest()[:8]))
        _add_node(_node_payload(kwp_id, kw, "keyword_plus", doc_idx, -9))
        for i in sorted(set(doc_idx)):
            edges.append({"source": paper_id_map[i], "target": kwp_id, "edge_type": "has_keyword_plus", "weight": 1, "year": _paper_year(i)})

    for i in active_set:
        pid = paper_id_map[i]
        year = _paper_year(i)
        for r in _safe_list(ref[i]):
            if r in top_ref_set and r in ref_id_map:
                rid = ref_id_map[r]
                edge_type = "cites_internal" if r in internal_map else "cites_external"
                edges.append({"source": pid, "target": rid, "edge_type": edge_type, "weight": 1, "year": year})

    nodes_df = pd.DataFrame(nodes) if nodes else pd.DataFrame(columns=["id","label","type","year","citations","size_val","group"])
    edges_df = pd.DataFrame(edges) if edges else pd.DataFrame(columns=["source","target","edge_type","weight","year"])
    papers_df = nodes_df[nodes_df["type"] == "paper"].copy().reset_index(drop=True)

    G = None
    if _NX_AVAILABLE and not nodes_df.empty:
        G = nx.DiGraph()
        for _, row in nodes_df.iterrows():
            G.add_node(row["id"], **row.to_dict())
        for _, row in edges_df.iterrows():
            G.add_edge(row["source"], row["target"], edge_type=row["edge_type"], weight=row.get("weight", 1))

    return nodes_df, edges_df, papers_df, G


# ──────────────────────────────────────────────────────────────────────────────
# §5  Indicator Computation
# ──────────────────────────────────────────────────────────────────────────────

def _compute_indicators(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    G,
    cfg: TSGConfig,
) -> pd.DataFrame:
    """Compute per-node scientometric indicators."""
    if nodes_df.empty:
        return pd.DataFrame()

    rows: List[Dict] = []

    # networkx centrality (only for paper nodes, to keep it fast)
    pr_map:  Dict[str, float] = {}
    bt_map:  Dict[str, float] = {}

    if G is not None and _NX_AVAILABLE and len(G.nodes) > 0:
        try:
            paper_nodes = {n for n, d in G.nodes(data=True) if d.get("type") == "paper"}
            sub = G.subgraph(paper_nodes).copy()
            if len(sub.nodes) > 0:
                pr_map = nx.pagerank(sub, alpha=0.85, max_iter=200, weight="weight")
                bt_map = nx.betweenness_centrality(sub, normalized=True, weight=None)
        except Exception:
            pass

    # Citation range for reference nodes
    cit_edges = edges_df[edges_df["edge_type"].isin(["cites_external", "cites_internal"])] if not edges_df.empty else pd.DataFrame()
    ref_cite_years: Dict[str, List[int]] = defaultdict(list)
    if not cit_edges.empty:
        for _, e in cit_edges.iterrows():
            tgt  = e["target"]
            yr   = e.get("year", 0)
            if yr and yr > 0:
                ref_cite_years[tgt].append(int(yr))

    for _, row in nodes_df.iterrows():
        nid   = row["id"]
        ntype = row.get("type", "unknown")
        yr    = row.get("year", 0)
        cit   = row.get("citations", 0)

        ind = {
            "id":    nid,
            "label": row.get("label", ""),
            "type":  ntype,
            "year":  yr,
        }

        if ntype == "paper":
            ind["citations"]             = cit
            ind["pagerank"]              = round(pr_map.get(nid, 0.0), 6)
            ind["betweenness"]           = round(bt_map.get(nid, 0.0), 6)
            ind["author_count"]          = row.get("author_count", 0)
            ind["keyword_count"]         = row.get("keyword_count", 0)
            ind["reference_count"]       = len(row.get("references", []))
            ind["internal_reference_count"] = row.get("internal_refs", 0)
            ind["external_reference_count"] = row.get("external_refs", 0)
            ind["topic"]                 = row.get("topic", -1)
            # Bridge score: high betweenness + moderate citations
            ind["bridge_score"]          = round(
                ind["betweenness"] * 0.7 + min(cit / max(1, cit + 50), 1) * 0.3, 4
            )

        elif ntype in ("reference", "internal_ref"):
            yrs = ref_cite_years.get(nid, [])
            ind["cited_by_dataset_count"] = cit
            ind["first_cited_year"]       = min(yrs) if yrs else None
            ind["last_cited_year"]        = max(yrs) if yrs else None
            ind["citation_span"]          = (max(yrs) - min(yrs)) if len(yrs) > 1 else 0
            ind["citation_velocity"]      = round(
                len(yrs) / max(1, (max(yrs) - min(yrs) + 1)) if len(yrs) > 1 else len(yrs), 3
            )
            ind["is_internal_reference"]  = ntype == "internal_ref"

        elif ntype == "author":
            ind["paper_count"]  = cit
            ind["active_years"] = None   # enriched later if needed

        elif ntype in ("keyword", "author_keyword", "keyword_plus"):
            ind["paper_count"]  = row.get("paper_count", cit)

        elif ntype in ("journal", "institution", "country"):
            ind["paper_count"]  = row.get("paper_count", cit)

        rows.append(ind)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# §6  HTML / D3.js Visualization
# ──────────────────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Temporal Scholarly Graph · PyBibX</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
:root{
  --bg:        #060b16;
  --bg2:       #0b1120;
  --bg3:       #111827;
  --panel:     rgba(11,17,32,0.92);
  --border:    rgba(99,179,237,0.12);
  --border2:   rgba(99,179,237,0.22);
  --text:      #dde3f0;
  --muted:     #5e7090;
  --amber:     #f59e0b;
  --amber-dim: rgba(245,158,11,0.18);
  --cyan:      #22d3ee;
  --cyan-dim:  rgba(34,211,238,0.15);
  --violet:    #a78bfa;
  --rose:      #f87171;
  --emerald:   #34d399;
  --sky:       #38bdf8;
  --fuchsia:   #e879f9;
  --orange:    #fb923c;
  --lime:      #a3e635;
  --gold:      #fbbf24;
  --glow-a:    rgba(245,158,11,0.35);
  --glow-c:    rgba(34,211,238,0.25);
  --glow-v:    rgba(167,139,250,0.25);
  --accent:    var(--amber);
  --font-head: 'DM Serif Display', serif;
  --font-ui:   'Syne', sans-serif;
  --font-mono: 'IBM Plex Mono', monospace;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { width: 100%; height: 100%; overflow: hidden; background: var(--bg); color: var(--text); font-family: var(--font-ui); font-size: 13px; }

/* ── Layout ──────────────────────────────────────────────── */
#root { display: flex; flex-direction: column; height: 100vh; }
#topbar { display: flex; align-items: center; gap: 10px; padding: 8px 14px; background: var(--bg2); border-bottom: 1px solid var(--border); z-index: 100; flex-shrink: 0; }
#canvas-row { flex: 1; display: flex; overflow: hidden; }
#left-panel { width: 188px; flex-shrink: 0; background: var(--panel); border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow-y: auto; }
#svg-wrap { flex: 1; position: relative; overflow: hidden; background: var(--bg); }
#right-panel { width: 270px; flex-shrink: 0; background: var(--panel); border-left: 1px solid var(--border); display: flex; flex-direction: column; overflow-y: auto; }

/* ── Topbar ──────────────────────────────────────────────── */
.brand { font-family: var(--font-head); font-size: 16px; color: var(--amber); white-space: nowrap; letter-spacing: 0.02em; }
.brand > span { color: var(--cyan); }
.brand-sub { font-family: var(--font-ui); font-size: 11px; font-weight: 400; color: var(--muted); margin-left: 4px; letter-spacing: 0.02em; }
.brand-sub .brand-divider { opacity: .45; margin: 0 4px; }
#center-sub { color: var(--cyan); font-weight: 500; }
#search-wrap { flex: 1; max-width: 340px; position: relative; }
#search-box { width: 100%; background: var(--bg3); border: 1px solid var(--border2); border-radius: 6px; color: var(--text); font-family: var(--font-mono); font-size: 12px; padding: 5px 10px 5px 30px; outline: none; transition: border-color .2s; }
#search-box:focus { border-color: var(--cyan); }
.search-icon { position: absolute; left: 9px; top: 50%; transform: translateY(-50%); color: var(--muted); font-size: 12px; pointer-events: none; }
#view-tabs { display: flex; gap: 2px; }
.vtab { background: none; border: 1px solid var(--border); border-radius: 5px; color: var(--muted); font-family: var(--font-ui); font-size: 11px; font-weight: 600; padding: 4px 9px; cursor: pointer; transition: all .2s; text-transform: uppercase; letter-spacing: .06em; }
.vtab:hover { color: var(--text); border-color: var(--border2); }
.vtab.active { background: var(--amber-dim); border-color: var(--amber); color: var(--amber); }
.top-btn { background: none; border: 1px solid var(--border); border-radius: 5px; color: var(--muted); font-size: 11px; padding: 4px 8px; cursor: pointer; font-family: var(--font-ui); transition: all .2s; }
.top-btn:hover { border-color: var(--border2); color: var(--text); }

/* ── Left Panel ──────────────────────────────────────────── */
.panel-section { padding: 7px 10px 6px; border-bottom: 1px solid var(--border); }
.panel-title { font-size: 9px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin-bottom: 5px; }
.toggle-row { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; cursor: pointer; user-select: none; }
.toggle-row:last-child { margin-bottom: 0; }
.toggle-swatch { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.toggle-label { font-size: 10px; color: var(--text); flex: 1; }
.toggle-cb { width: 12px; height: 12px; accent-color: var(--amber); cursor: pointer; }
.range-row { display: flex; gap: 5px; align-items: center; margin-bottom: 4px; }
.range-row label { font-size: 10px; color: var(--muted); width: 32px; }
.range-row input[type="number"] { flex: 1; background: var(--bg3); border: 1px solid var(--border); border-radius: 4px; color: var(--text); font-family: var(--font-mono); font-size: 10px; padding: 2px 5px; outline: none; }
.range-row input:focus { border-color: var(--cyan); }
.slider-wrap { padding: 0 2px; }
.slider-wrap label { font-size: 10px; color: var(--muted); display: block; margin-bottom: 3px; }
#year-slider { width: 100%; accent-color: var(--amber); }
.apply-btn { width: 100%; margin-top: 5px; background: var(--amber-dim); border: 1px solid var(--amber); border-radius: 5px; color: var(--amber); font-family: var(--font-ui); font-size: 10px; font-weight: 700; padding: 4px; cursor: pointer; letter-spacing: .05em; transition: all .2s; text-transform: uppercase; }
.apply-btn:hover { background: rgba(245,158,11,0.28); }
.panel-select { width: 100%; background: var(--bg3); border: 1px solid var(--border); border-radius: 4px; color: var(--text); font-family: var(--font-mono); font-size: 10px; padding: 3px 5px; outline: none; cursor: pointer; }
.panel-select:focus { border-color: var(--cyan); }

/* ── Right Panel ─────────────────────────────────────────── */
#info-placeholder { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; color: var(--muted); padding: 20px; text-align: center; }
#info-placeholder .ph-icon { font-size: 32px; opacity: .3; }
#info-placeholder p { font-size: 11px; line-height: 1.6; }
#info-content { display: none; flex-direction: column; height: 100%; }
.info-header { padding: 12px 14px 8px; border-bottom: 1px solid var(--border); }
.info-type-badge { display: inline-block; font-size: 9px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; padding: 2px 7px; border-radius: 10px; margin-bottom: 6px; }
.info-title { font-family: var(--font-head); font-size: 13px; color: var(--text); line-height: 1.4; }
.info-body { flex: 1; overflow-y: auto; padding: 10px 14px; }
.info-row { display: flex; gap: 6px; margin-bottom: 6px; align-items: flex-start; }
.info-key { font-family: var(--font-mono); font-size: 10px; color: var(--muted); width: 72px; flex-shrink: 0; padding-top: 1px; }
.info-val { font-size: 11px; color: var(--text); line-height: 1.5; word-break: break-word; }
.tag-list { display: flex; flex-wrap: wrap; gap: 3px; }
.tag { background: var(--bg3); border: 1px solid var(--border); border-radius: 3px; color: var(--muted); font-size: 10px; padding: 1px 5px; cursor: pointer; transition: all .15s; }
.tag:hover { border-color: var(--cyan); color: var(--cyan); }
.ind-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; }
.ind-cell { background: var(--bg3); border: 1px solid var(--border); border-radius: 4px; padding: 5px 7px; }
.ind-key { font-family: var(--font-mono); font-size: 9px; color: var(--muted); }
.ind-val { font-family: var(--font-mono); font-size: 13px; color: var(--amber); font-weight: 500; margin-top: 1px; }
.ego-btn { width: 100%; margin-top: 8px; background: none; border: 1px solid var(--cyan); border-radius: 5px; color: var(--cyan); font-family: var(--font-ui); font-size: 11px; font-weight: 600; padding: 5px; cursor: pointer; letter-spacing: .05em; text-transform: uppercase; transition: all .2s; }
.ego-btn:hover { background: var(--cyan-dim); }

/* ── Graph Canvas ────────────────────────────────────────── */
#main-svg { width: 100%; height: 100%; }
.node { cursor: pointer; transition: opacity .2s; }
.node:hover { opacity: .85; }
.node circle { stroke-width: 1.5; transition: r .3s; }
.node text { pointer-events: none; fill: var(--text); font-family: var(--font-mono); font-size: 9px; }
.link { fill: none; transition: opacity .2s; }
.link.cites_external  { stroke: rgba(239,68,68,0.26); }
.link.cites_internal    { stroke: rgba(124,58,237,0.32); }
.link.wrote             { stroke: rgba(193,154,107,0.22); }
.link.published_in      { stroke: rgba(203,213,225,0.22); }
.link.has_institution   { stroke: rgba(250,204,21,0.22); }
.link.has_country       { stroke: rgba(20,184,166,0.22); }
.link.has_author_keyword{ stroke: rgba(56,189,248,0.15); }
.link.has_keyword_plus  { stroke: rgba(232,121,249,0.18); }
.node.dimmed { opacity: 0.08; }
.link.dimmed { opacity: 0.04; }
.node.highlighted circle { filter: drop-shadow(0 0 6px currentColor); }

/* ── Tooltip ─────────────────────────────────────────────── */
#tooltip { position: fixed; background: var(--bg2); border: 1px solid var(--border2); border-radius: 7px; padding: 8px 12px; pointer-events: none; opacity: 0; transition: opacity .15s; z-index: 999; max-width: 260px; }
#tooltip .tt-type { font-size: 9px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
#tooltip .tt-title { font-size: 12px; color: var(--text); margin-top: 2px; line-height: 1.4; }
#tooltip .tt-stat { font-family: var(--font-mono); font-size: 10px; color: var(--amber); margin-top: 4px; }

/* ── Search Results ──────────────────────────────────────── */
#search-results { position: absolute; top: 100%; left: 0; right: 0; background: var(--bg2); border: 1px solid var(--border2); border-radius: 6px; z-index: 999; max-height: 220px; overflow-y: auto; margin-top: 2px; }
.sr-item { padding: 6px 10px; cursor: pointer; border-bottom: 1px solid var(--border); }
.sr-item:last-child { border-bottom: none; }
.sr-item:hover { background: var(--bg3); }
.sr-type { font-size: 9px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
.sr-label { font-size: 11px; color: var(--text); margin-top: 1px; }

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

/* ── Glow pulse animation ────────────────────────────────── */
@keyframes glow-pulse { 0%,100%{opacity:.6} 50%{opacity:1} }
.loading-glow { animation: glow-pulse 1.6s ease-in-out infinite; }

</style>
</head>
<body>
<div id="root">
  <!-- ── TOPBAR ──────────────────────────────────────────── -->
  <div id="topbar">
    <div class="brand">Py<span>Bib</span>X<span class="brand-sub">TSG<span class="brand-divider">·</span><span id="center-sub">Paper-centred</span></span></div>
    <div id="search-wrap">
      <span class="search-icon">⌕</span>
      <input type="text" id="search-box" placeholder="Search paper, author, journal, institution, country, keyword, reference…" autocomplete="off"/>
      <div id="search-results" style="display:none;"></div>
    </div>
    <div id="view-tabs">
      <button class="vtab active" data-view="timeline">Timeline</button>
      <button class="vtab" data-view="force">Force</button>
      <button class="vtab" data-view="ego">Ego</button>
    </div>
    <button class="top-btn" id="btn-reset">Reset</button>
    <button class="top-btn" id="btn-export">Export HTML</button>
  </div>

  <div id="canvas-row">
    <!-- ── LEFT PANEL ─────────────────────────────────────── -->
    <div id="left-panel">
      <div class="panel-section">
        <div class="panel-title">Node Toggles</div>
        <label class="toggle-row"><span class="toggle-swatch" style="background:var(--amber)"></span><span class="toggle-label">Papers</span><input type="checkbox" class="toggle-cb node-cb" data-ntype="paper" checked/></label>
        <label class="toggle-row"><span class="toggle-swatch" style="background:var(--rose)"></span><span class="toggle-label">Ext. References</span><input type="checkbox" class="toggle-cb node-cb" data-ntype="reference"/></label>
        <label class="toggle-row"><span class="toggle-swatch" style="background:#7c3aed"></span><span class="toggle-label">Int. References</span><input type="checkbox" class="toggle-cb node-cb" data-ntype="internal_ref"/></label>
        <label class="toggle-row"><span class="toggle-swatch" style="background:#c19a6b"></span><span class="toggle-label">Authors</span><input type="checkbox" class="toggle-cb node-cb" data-ntype="author"/></label>
        <label class="toggle-row"><span class="toggle-swatch" style="background:#cbd5e1"></span><span class="toggle-label">Journals</span><input type="checkbox" class="toggle-cb node-cb" data-ntype="journal"/></label>
        <label class="toggle-row"><span class="toggle-swatch" style="background:#facc15"></span><span class="toggle-label">Institutions</span><input type="checkbox" class="toggle-cb node-cb" data-ntype="institution"/></label>
        <label class="toggle-row"><span class="toggle-swatch" style="background:#14b8a6"></span><span class="toggle-label">Countries</span><input type="checkbox" class="toggle-cb node-cb" data-ntype="country"/></label>
        <label class="toggle-row"><span class="toggle-swatch" style="background:var(--sky)"></span><span class="toggle-label">Authors' Keywords</span><input type="checkbox" class="toggle-cb node-cb" data-ntype="author_keyword"/></label>
        <label class="toggle-row"><span class="toggle-swatch" style="background:var(--fuchsia)"></span><span class="toggle-label">Keywords Plus</span><input type="checkbox" class="toggle-cb node-cb" data-ntype="keyword_plus"/></label>
      </div>
      <div class="panel-section">
        <div class="panel-title">Edge Toggles</div>
        <label class="toggle-row"><span class="toggle-swatch" style="background:var(--rose)"></span><span class="toggle-label">Ext. References</span><input type="checkbox" class="toggle-cb layer-cb" data-layer="cites_external" checked/></label>
        <label class="toggle-row"><span class="toggle-swatch" style="background:#7c3aed"></span><span class="toggle-label">Int. References</span><input type="checkbox" class="toggle-cb layer-cb" data-layer="cites_internal" checked/></label>
        <label class="toggle-row"><span class="toggle-swatch" style="background:#c19a6b"></span><span class="toggle-label">Authors</span><input type="checkbox" class="toggle-cb layer-cb" data-layer="wrote"/></label>
        <label class="toggle-row"><span class="toggle-swatch" style="background:#cbd5e1"></span><span class="toggle-label">Journals</span><input type="checkbox" class="toggle-cb layer-cb" data-layer="published_in"/></label>
        <label class="toggle-row"><span class="toggle-swatch" style="background:#facc15"></span><span class="toggle-label">Institutions</span><input type="checkbox" class="toggle-cb layer-cb" data-layer="has_institution"/></label>
        <label class="toggle-row"><span class="toggle-swatch" style="background:#14b8a6"></span><span class="toggle-label">Countries</span><input type="checkbox" class="toggle-cb layer-cb" data-layer="has_country"/></label>
        <label class="toggle-row"><span class="toggle-swatch" style="background:var(--sky)"></span><span class="toggle-label">Authors' Keywords</span><input type="checkbox" class="toggle-cb layer-cb" data-layer="has_author_keyword"/></label>
        <label class="toggle-row"><span class="toggle-swatch" style="background:var(--fuchsia)"></span><span class="toggle-label">Keywords Plus</span><input type="checkbox" class="toggle-cb layer-cb" data-layer="has_keyword_plus"/></label>
      </div>
      <div class="panel-section">
        <div class="panel-title">Color By</div>
        <select id="color-by-sel" class="panel-select">
          <option value="type">Node Type</option>
          <option value="citations">Citations</option>
          <option value="year">Year</option>
        </select>
        <div class="panel-title" style="margin-top:7px;">Size By</div>
        <select id="size-by-sel" class="panel-select">
          <option value="citations">Citations</option>
          <option value="uniform">Uniform</option>
          <option value="references">References</option>
        </select>
      </div>
      <div class="panel-section" style="border-bottom:none;">
        <div class="panel-title">Year Range</div>
        <div class="range-row"><label>From</label><input type="number" id="yr-from" placeholder="auto"/></div>
        <div class="range-row"><label>To</label><input type="number" id="yr-to"   placeholder="auto"/></div>
        <button class="apply-btn" id="btn-year-apply">Apply</button>
      </div>
    </div>

    <!-- ── SVG CANVAS ──────────────────────────────────────── -->
    <div id="svg-wrap">
      <svg id="main-svg"></svg>
    </div>

    <!-- ── RIGHT PANEL ─────────────────────────────────────── -->
    <div id="right-panel">
      <div id="info-placeholder">
        <div class="ph-icon">◎</div>
        <p>Click any node to explore its scholarly context, indicators, and connections.</p>
      </div>
      <div id="info-content"></div>
    </div>
  </div>
</div>

<div id="tooltip"><div class="tt-type" id="tt-type"></div><div class="tt-title" id="tt-title"></div><div class="tt-stat" id="tt-stat"></div></div>

<script>
// ═══════════════════════════════════════════════════════════════════════════
//  DATA (injected by Python)
// ═══════════════════════════════════════════════════════════════════════════
const RAW_NODES = __NODES_JSON__;
const RAW_EDGES = __EDGES_JSON__;
const RAW_INDICATORS = __INDICATORS_JSON__;
const META = __META_JSON__;

// ═══════════════════════════════════════════════════════════════════════════
//  Constants & colour helpers
// ═══════════════════════════════════════════════════════════════════════════
const TYPE_COLOR = {
  paper:          '#f59e0b',
  reference:      '#ef4444',
  internal_ref:   '#7c3aed',
  author:         '#c19a6b',
  journal:        '#cbd5e1',
  institution:    '#facc15',
  country:        '#14b8a6',
  author_keyword: '#38bdf8',
  keyword_plus:   '#e879f9',
  keyword:        '#38bdf8',
};

function nodeColor(d, colorBy) {
  if (colorBy === 'type') {
    if (d.type === 'paper' && d.is_internal_reference_target) return TYPE_COLOR.internal_ref;
    return TYPE_COLOR[d.type] || '#5e7090';
  }
  // Non-'type' scales apply only to center-matching nodes.
  // Everything else keeps its TYPE_COLOR so context stays readable.
  if (!isCenterNode(d)) return TYPE_COLOR[d.type] || '#5e7090';
  if (colorBy === 'citations') return citScale(d.citations || 0);
  if (colorBy === 'papers')    return paperCountScale(d.paper_count || d.citations || 0);
  if (colorBy === 'year')      return yearColorScale(d.year || d.first_year || yearMin);
  return TYPE_COLOR[d.type] || '#5e7090';
}

function nodeStroke(d, colorBy) {
  if (d.type === 'internal_ref') return '#d8b4fe';
  if (d.is_internal_reference_target && colorBy === 'type') return '#d8b4fe';
  return 'rgba(0,0,0,0.5)';
}

// ═══════════════════════════════════════════════════════════════════════════
//  State
// ═══════════════════════════════════════════════════════════════════════════
const CENTER_DEFAULTS = {
  paper:          {layers:[], types:['paper']},
  reference:      {layers:[], types:['reference','internal_ref']},
  author:         {layers:[], types:['author']},
  journal:        {layers:[], types:['journal']},
  institution:    {layers:[], types:['institution']},
  country:        {layers:[], types:['country']},
  author_keyword: {layers:[], types:['author_keyword']},
  keyword_plus:   {layers:[], types:['keyword_plus']},
};

// Per-center option lists for Color By / Size By and the brand subtitle.
//   - 'type'       always means TYPE_COLOR by node.type        (universal)
//   - 'citations'  means d.citations            (papers: citation count, refs: cited-by)
//   - 'papers'     means d.paper_count          (non-paper entities: how many papers)
//   - 'year'       means d.year / d.first_year  (paper pub. year / entity first activity)
//   - 'references' means d.external_refs        (papers only)
//   - 'uniform'    means a constant size
const CENTER_OPTIONS = {
  paper:          {subtitle: 'Paper-centred',
                   color: [['type','Node Type'],['citations','Citations'],['year','Year']],
                   size:  [['citations','Citations'],['references','References'],['uniform','Uniform']]},
  reference:      {subtitle: 'Reference-centred',
                   color: [['type','Node Type'],['citations','Times Cited'],['year','First Year']],
                   size:  [['citations','Times Cited'],['uniform','Uniform']]},
  author:         {subtitle: 'Author-centred',
                   color: [['type','Node Type'],['papers','Papers Authored'],['year','First Year']],
                   size:  [['papers','Papers Authored'],['uniform','Uniform']]},
  journal:        {subtitle: 'Journal-centred',
                   color: [['type','Node Type'],['papers','Papers Published'],['year','First Year']],
                   size:  [['papers','Papers Published'],['uniform','Uniform']]},
  institution:    {subtitle: 'Institution-centred',
                   color: [['type','Node Type'],['papers','Papers'],['year','First Year']],
                   size:  [['papers','Papers'],['uniform','Uniform']]},
  country:        {subtitle: 'Country-centred',
                   color: [['type','Node Type'],['papers','Papers'],['year','First Year']],
                   size:  [['papers','Papers'],['uniform','Uniform']]},
  author_keyword: {subtitle: "Authors' Keyword-centred",
                   color: [['type','Node Type'],['papers','Paper Count'],['year','First Year']],
                   size:  [['papers','Paper Count'],['uniform','Uniform']]},
  keyword_plus:   {subtitle: 'Keyword Plus-centred',
                   color: [['type','Node Type'],['papers','Paper Count'],['year','First Year']],
                   size:  [['papers','Paper Count'],['uniform','Uniform']]},
};

function centerOpts() {
  return CENTER_OPTIONS[state.center] || CENTER_OPTIONS.paper;
}

// `state.center === 'reference'` includes both 'reference' and 'internal_ref' nodes.
function isCenterNode(d) {
  if (state.center === 'reference') return d.type === 'reference' || d.type === 'internal_ref';
  return d.type === state.center;
}

function rebuildSelectOptions(sel, options, currentValue) {
  if (!sel) return;
  sel.innerHTML = '';
  let restored = false;
  options.forEach(([val, label]) => {
    const opt = document.createElement('option');
    opt.value = val;
    opt.textContent = label;
    if (val === currentValue) { opt.selected = true; restored = true; }
    sel.appendChild(opt);
  });
  // If the previous value isn't valid for this center, fall back to the first option
  if (!restored && options.length) {
    sel.value = options[0][0];
  }
}

function applyCenterOptions() {
  const opts = centerOpts();
  const colorSel = document.getElementById('color-by-sel');
  const sizeSel  = document.getElementById('size-by-sel');
  rebuildSelectOptions(colorSel, opts.color, state.colorBy);
  rebuildSelectOptions(sizeSel,  opts.size,  state.sizeBy);
  // Sync state in case the previous value was dropped
  if (colorSel) state.colorBy = colorSel.value;
  if (sizeSel)  state.sizeBy  = sizeSel.value;
  const sub = document.getElementById('center-sub');
  if (sub) sub.textContent = opts.subtitle;
}

function buildInitialState(){
  const center = (META.config && META.config.center) || 'paper';
  const base = CENTER_DEFAULTS[center] || CENTER_DEFAULTS.paper;
  return {
    view:         (META.config && META.config.view) || 'timeline',
    colorBy:      (META.config && META.config.color_by) || 'type',
    sizeBy:       (META.config && META.config.size_by) || 'citations',
    selectedId:   (META.config && META.config.selected) || null,
    lensFilter:   null,
    activeLayers: new Set(base.layers),
    activeTypes:  new Set(base.types),
    yearFrom:     (META.config && META.config.start_year) || null,
    yearTo:       (META.config && META.config.end_year) || null,
    center:       center,
  };
}

let state = buildInitialState();

function syncControlsFromState(){
  document.querySelectorAll('.layer-cb').forEach(cb => { cb.checked = state.activeLayers.has(cb.dataset.layer); });
  document.querySelectorAll('.node-cb').forEach(cb => { cb.checked = state.activeTypes.has(cb.dataset.ntype); });
  const viewBtns = document.querySelectorAll('.vtab');
  viewBtns.forEach(b => b.classList.toggle('active', b.dataset.view === state.view));
  // Rebuild center-aware Color / Size options first, then sync remaining inputs.
  applyCenterOptions();
  const yrFrom = document.getElementById('yr-from');
  const yrTo = document.getElementById('yr-to');
  if (yrFrom) yrFrom.value = state.yearFrom || '';
  if (yrTo) yrTo.value = state.yearTo || '';
}

// ═══════════════════════════════════════════════════════════════════════════
//  Derived data helpers
// ═══════════════════════════════════════════════════════════════════════════
const indicatorMap = {};
RAW_INDICATORS.forEach(d => { indicatorMap[d.id] = d; });

const allYears = RAW_NODES
  .filter(d => d.type === 'paper' && d.year > 0)
  .map(d => d.year);
const yearMin = allYears.length ? Math.min(...allYears) : 2000;
const yearMax = allYears.length ? Math.max(...allYears) : 2025;

const maxCit = d3.max(RAW_NODES, d => d.citations || 0) || 1;
const maxPaperCount = d3.max(
  RAW_NODES.filter(d => d.type !== 'paper'),
  d => d.paper_count || d.citations || 0
) || 1;
const citScale = d3.scaleSequential()
  .domain([0, maxCit])
  .interpolator(d3.interpolateYlOrRd);
const paperCountScale = d3.scaleSequential()
  .domain([0, maxPaperCount])
  .interpolator(d3.interpolateViridis);
const yearColorScale = d3.scaleSequential()
  .domain([yearMin, yearMax])
  .interpolator(d3.interpolatePlasma);

// Fast id → node lookup for the edge-projection logic.
const NODE_BY_ID = new Map();
RAW_NODES.forEach(n => NODE_BY_ID.set(n.id, n));

// Precomputed indexes for the info-panel's "related entities" section.
//   entityToPapers          : entityId → Set<paperId>
//   paperToEntitiesByType   : paperId → { type: Set<entityId> }
const entityToPapers        = new Map();
const paperToEntitiesByType = new Map();
RAW_EDGES.forEach(e => {
  const s = NODE_BY_ID.get(e.source);
  const t = NODE_BY_ID.get(e.target);
  if (!s || !t) return;
  let pid, eid, etype;
  if (s.type === 'paper' && t.type !== 'paper') { pid = e.source; eid = e.target; etype = t.type; }
  else if (t.type === 'paper' && s.type !== 'paper') { pid = e.target; eid = e.source; etype = s.type; }
  else return;
  let pset = entityToPapers.get(eid);
  if (!pset) { pset = new Set(); entityToPapers.set(eid, pset); }
  pset.add(pid);
  let bag = paperToEntitiesByType.get(pid);
  if (!bag) { bag = {}; paperToEntitiesByType.set(pid, bag); }
  if (!bag[etype]) bag[etype] = new Set();
  bag[etype].add(eid);
});

// Top N entities of `targetType` co-occurring with `entityId` through papers.
function topRelatedEntities(entityId, targetType, limit) {
  limit = limit || 8;
  const papers = entityToPapers.get(entityId);
  if (!papers || !papers.size) return [];
  const counts = new Map();
  papers.forEach(pid => {
    const bag = paperToEntitiesByType.get(pid);
    if (!bag || !bag[targetType]) return;
    bag[targetType].forEach(eid => {
      if (eid === entityId) return;
      counts.set(eid, (counts.get(eid) || 0) + 1);
    });
  });
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([eid, count]) => ({
      id: eid,
      label: (NODE_BY_ID.get(eid) || {}).label || eid,
      count,
    }));
}

// Top N papers connected to a non-paper entity, sorted by paper citations.
function topConnectedPapers(entityId, limit) {
  limit = limit || 6;
  const papers = entityToPapers.get(entityId);
  if (!papers || !papers.size) return [];
  return [...papers]
    .map(pid => NODE_BY_ID.get(pid))
    .filter(Boolean)
    .sort((a, b) => (b.citations || 0) - (a.citations || 0))
    .slice(0, limit)
    .map(p => ({id: p.id, label: p.label, count: p.citations || 0}));
}

// ═══════════════════════════════════════════════════════════════════════════
//  SVG setup
// ═══════════════════════════════════════════════════════════════════════════
const svgEl = d3.select('#main-svg');
let W = 0, H = 0;

const zoomGroup = svgEl.append('g').attr('id', 'zoom-group');
const yearAxisGroup = zoomGroup.append('g').attr('id', 'year-axis');
const linkGroup     = zoomGroup.append('g').attr('id', 'link-layer');
const nodeGroup     = zoomGroup.append('g').attr('id', 'node-layer');

const zoom = d3.zoom()
  .scaleExtent([0.1, 8])
  .on('zoom', ev => zoomGroup.attr('transform', ev.transform));
svgEl.call(zoom);

// ═══════════════════════════════════════════════════════════════════════════
//  Size helpers
// ═══════════════════════════════════════════════════════════════════════════
function nodeRadius(d) {
  const sb = state.sizeBy;
  if (sb === 'uniform') return d.type === 'paper' ? 8 : 5;
  // Center-matching nodes get the chosen metric; everything else keeps size_val.
  if (isCenterNode(d)) {
    if (sb === 'citations')  return Math.max(3, Math.min(32, 3 + Math.log1p(d.citations || 0)   * 4));
    if (sb === 'papers')     return Math.max(3, Math.min(32, 3 + Math.log1p(d.paper_count || d.citations || 0) * 4));
    if (sb === 'references') return Math.max(3, Math.min(28, 3 + Math.log1p(d.external_refs || 0) * 4));
  }
  return Math.max(3, Math.min(32, d.size_val || 5));
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

// ═══════════════════════════════════════════════════════════════════════════
//  Filter logic
// ═══════════════════════════════════════════════════════════════════════════
function getVisibleNodes() {
  let nodes = RAW_NODES;

  // Type filter. Special case for papers: a type=paper node is also surfaced
  // when the 'internal_ref' toggle is on AND the paper is itself an internal
  // reference (is_internal_reference_target=true). That makes Int. Ref a real
  // toggle even in datasets where every cited paper sits inside active_set
  // (so there are no separate type='internal_ref' nodes to display).
  nodes = nodes.filter(d => {
    if (d.type === 'paper') {
      if (state.activeTypes.has('paper')) return true;
      if (d.is_internal_reference_target && state.activeTypes.has('internal_ref')) return true;
      return false;
    }
    return state.activeTypes.has(d.type);
  });

  // year filter
  const yf = state.yearFrom, yt = state.yearTo;
  nodes = nodes.filter(d => {
    // Keep undated nodes visible. In timeline view they are placed
    // in a dedicated undated lane instead of being removed.
    if (!(d.year > 0)) return true;
    if (yf && d.year < yf) return false;
    if (yt && d.year > yt) return false;
    return true;
  });

  // lens filter
  if (state.lensFilter) {
    const { type: lt, value: lv } = state.lensFilter;
    const visiblePaperIds = new Set();

    RAW_NODES.filter(d => d.type === 'paper').forEach(d => {
      if (lt === 'author'    && (d.authors || []).includes(lv)) visiblePaperIds.add(d.id);
      if (lt === 'topic'     && String(d.topic) === String(lv)) visiblePaperIds.add(d.id);
      if (lt === 'keyword'   && (d.keywords || []).includes(lv)) visiblePaperIds.add(d.id);
      if (lt === 'reference' && (d.references || []).includes(lv)) visiblePaperIds.add(d.id);
    });

    const connectedIds = new Set(visiblePaperIds);
    RAW_EDGES.forEach(e => {
      if (visiblePaperIds.has(e.source)) connectedIds.add(e.target);
      if (visiblePaperIds.has(e.target)) connectedIds.add(e.source);
    });

    nodes = nodes.filter(d => connectedIds.has(d.id));
  }

  // ego filter
  if (state.view === 'ego' && state.selectedId) {
    const ego = state.selectedId;
    const egoNode = NODE_BY_ID.get(ego);
    const neighbours = new Set([ego]);
    const egoPapers = new Set();

    // Always keep direct neighbours of the selected node. For a non-paper
    // focal node, also remember the papers that mediate its relations to
    // journals, institutions, countries, keywords, references, etc.
    RAW_EDGES.forEach(e => {
      if (e.source === ego || e.target === ego) {
        const otherId = e.source === ego ? e.target : e.source;
        const otherNode = NODE_BY_ID.get(otherId);
        neighbours.add(e.source);
        neighbours.add(e.target);
        if (otherNode && otherNode.type === 'paper') egoPapers.add(otherId);
      }
    });

    if (egoNode && egoNode.type === 'paper') egoPapers.add(ego);

    // Ego view used to stop at one-hop neighbours. That isolated authors,
    // journals, countries, institutions, and keywords from each other because
    // their actual relation is paper-mediated: Author -> Paper -> Country,
    // Author -> Paper -> Journal, Keyword -> Paper -> Institution, etc.
    // Add those second-hop endpoints when their edge layer is enabled; the
    // later type filter still decides whether the node category is displayed.
    if (egoPapers.size) {
      RAW_EDGES.forEach(e => {
        if (!state.activeLayers.has(e.edge_type)) return;
        let otherId = null;
        if (egoPapers.has(e.source)) otherId = e.target;
        else if (egoPapers.has(e.target)) otherId = e.source;
        if (!otherId || otherId === ego) return;
        neighbours.add(otherId);
      });
    }

    nodes = nodes.filter(d => neighbours.has(d.id));
  }

  return nodes;
}

// When papers are hidden, direct paper-mediated edges (e.g. author→paper→journal)
// drop out of view. To preserve the relationship, project them: for every paper
// P, link any visible focal/center entity to every visible non-paper entity X
// reached via an enabled layer. In Ego view the selected node is the focal node,
// even when it is not the configured center. The projected edge inherits the
// styling of the outward layer (published_in, has_country, …), so it reads like
// a direct connection without forcing the intermediary paper nodes onscreen.
function buildProjectedEdges(visibleNodeIds, addEdge) {
  const centerIsRefFamily = state.center === 'reference';
  const egoProjectionId = (state.view === 'ego' && state.selectedId) ? state.selectedId : null;
  function isProjectionSource(c) {
    if (egoProjectionId) return c.other === egoProjectionId;
    return centerIsRefFamily ? (c.otherType === 'reference' || c.otherType === 'internal_ref') : c.otherType === state.center;
  }

  // Group every paper-attached edge under its paper id.
  const paperConn = new Map();   // paper_id -> [{other, otherType, edgeType}, …]
  for (const e of RAW_EDGES) {
    const s = NODE_BY_ID.get(e.source);
    const t = NODE_BY_ID.get(e.target);
    if (!s || !t) continue;
    let pid, other, otherType;
    if (s.type === 'paper' && t.type !== 'paper') { pid = e.source; other = e.target; otherType = t.type; }
    else if (t.type === 'paper' && s.type !== 'paper') { pid = e.target; other = e.source; otherType = s.type; }
    else continue;
    let arr = paperConn.get(pid);
    if (!arr) { arr = []; paperConn.set(pid, arr); }
    arr.push({other, otherType, edgeType: e.edge_type});
  }

  paperConn.forEach(conns => {
    const centerSide = [];
    const targetSide = [];
    for (const c of conns) {
      if (!visibleNodeIds.has(c.other)) continue;
      if (isProjectionSource(c))                      centerSide.push(c);
      else if (state.activeLayers.has(c.edgeType))    targetSide.push(c);
    }
    if (!centerSide.length || !targetSide.length) return;
    for (const a of centerSide) {
      for (const b of targetSide) {
        addEdge({source: a.other, target: b.other, edge_type: b.edgeType, weight: 1, year: 0, projected: true});
      }
    }
  });
}

function getVisibleEdges(visibleNodeIds) {
  const out = [];
  const seen = new Set();
  function addEdge(e) {
    const k = e.source + '|' + e.target + '|' + e.edge_type;
    if (seen.has(k)) return;
    seen.add(k);
    out.push(e);
  }

  // 1) Direct edges with both endpoints visible.
  for (const e of RAW_EDGES) {
    if (!state.activeLayers.has(e.edge_type)) continue;
    if (visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target)) addEdge(e);
  }

  // 2) Project through hidden papers. In ego view this is what allows
  //    author/journal/country/institution/keyword/reference connections to be
  //    shown around the selected focal node when paper nodes are hidden.
  if (!state.activeTypes.has('paper') && (state.center !== 'paper' || state.view === 'ego')) {
    buildProjectedEdges(visibleNodeIds, addEdge);
  }

  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Force simulation
// ═══════════════════════════════════════════════════════════════════════════
let sim = null;

function buildXScale(innerW) {
  const domainMin = (state.yearFrom || yearMin) - 1;
  const domainMax = (state.yearTo   || yearMax) + 1;
  // Leave room on the left for an undated lane in timeline view.
  return d3.scaleLinear().domain([domainMin, domainMax]).range([120, innerW - 40]);
}

function renderGraph() {
  if (!W || !H) measureCanvas();

  const innerW = W - 10;
  const innerH = H - 40;
  const xScale = buildXScale(innerW);
  const undatedX = 58;

  const visNodes = getVisibleNodes();
  const nodeIdSet = new Set(visNodes.map(d => d.id));
  const visEdges = getVisibleEdges(nodeIdSet);

  // Year gridlines and axis labels removed — the timeline lays out by year
  // implicitly via x-forces; the rulers added visual noise without adding info.
  yearAxisGroup.selectAll('*').remove();

  // ── Clone nodes for simulation ────────────────────────────────────────
  const simNodes = visNodes.map(d => ({ ...d }));
  const simEdges = visEdges.map(e => ({ ...e }));
  const nodeById = new Map(simNodes.map(d => [d.id, d]));

  const linkedEdges = simEdges.map(e => ({
    ...e,
    source: nodeById.get(e.source) || e.source,
    target: nodeById.get(e.target) || e.target,
  })).filter(e => e.source && e.target && typeof e.source === 'object' && typeof e.target === 'object');

  // initial positions
  simNodes.forEach(d => {
    if (state.view === 'timeline') {
      d.x = d.year > 0 ? xScale(d.year) : undatedX;
      d.y = innerH / 2 + (Math.random() - 0.5) * Math.min(140, innerH * 0.45);
    } else if (d.year > 0) {
      d.x = xScale(d.year) + (Math.random() - 0.5) * 20;
      d.y = innerH / 2 + (Math.random() - 0.5) * innerH * 0.35;
    } else {
      d.x = innerW / 2 + (Math.random() - 0.5) * innerW * 0.5;
      d.y = innerH / 2 + (Math.random() - 0.5) * innerH * 0.5;
    }
  });

  if (sim) sim.stop();

  // ── Simulation ────────────────────────────────────────────────────────
  if (state.view === 'timeline') {
    // Beeswarm / constrained force layout:
    // forceX -> year column or undated lane
    // forceY -> vertical center
    // collide -> vertical stacking without overlap
    sim = d3.forceSimulation(simNodes)
      .force('x', d3.forceX(d => d.year > 0 ? xScale(d.year) : undatedX).strength(1))
      .force('y', d3.forceY(innerH / 2).strength(0.12))
      .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 2).strength(1))
      .alpha(1)
      .alphaDecay(0.05)
      .velocityDecay(0.32);
  } else if (state.view === 'ego' && state.selectedId) {
    const cx = innerW / 2;
    const cy = innerH / 2;

    sim = d3.forceSimulation(simNodes)
      .force('link', d3.forceLink(linkedEdges).id(d => d.id).distance(70).strength(0.45))
      .force('charge', d3.forceManyBody().strength(-120).distanceMax(260))
      .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 3).strength(0.9))
      .force('x', d3.forceX(d => d.id === state.selectedId ? cx : cx + (Math.random() - .5) * 300).strength(d => d.id === state.selectedId ? 1 : 0.12))
      .force('y', d3.forceY(d => d.id === state.selectedId ? cy : cy + (Math.random() - .5) * 240).strength(d => d.id === state.selectedId ? 1 : 0.12))
      .alphaDecay(0.03);
  } else {
    sim = d3.forceSimulation(simNodes)
      .force('link', d3.forceLink(linkedEdges).id(d => d.id).distance(55).strength(0.3))
      .force('charge', d3.forceManyBody().strength(-80).distanceMax(220))
      .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 3).strength(0.8))
      .force('x', d3.forceX(innerW / 2).strength(0.03))
      .force('y', d3.forceY(innerH / 2).strength(0.04))
      .alphaDecay(0.025);
  }

  // ── Links ─────────────────────────────────────────────────────────────
  const linkSel = linkGroup.selectAll('.link')
    .data(linkedEdges, e => `${typeof e.source === 'object' ? e.source.id : e.source}--${typeof e.target === 'object' ? e.target.id : e.target}--${e.edge_type}`)
    .join(
      enter => enter.append('line')
        .attr('class', e => `link ${e.edge_type.replace(/\s/g, '_')}`)
        .attr('stroke-width', 1)
        .attr('opacity', 0)
        .call(l => l.transition().duration(300).attr('opacity', 1)),
      update => update,
      exit => exit.transition().duration(180).attr('opacity', 0).remove()
    );

  // ── Nodes ─────────────────────────────────────────────────────────────
  const nodeSel = nodeGroup.selectAll('.node')
    .data(simNodes, d => d.id)
    .join(
      enter => {
        const g = enter.append('g')
          .attr('class', 'node')
          .attr('opacity', 0)
          .call(d3.drag().on('start', dragStart).on('drag', dragged).on('end', dragEnd));

        g.append('circle');
        g.append('text').attr('dy', '0.32em').attr('text-anchor', 'middle');
        g.call(sel => sel.transition().duration(300).attr('opacity', 1));
        return g;
      },
      update => update,
      exit => exit.transition().duration(180).attr('opacity', 0).remove()
    );

  nodeSel.select('circle')
    .attr('r', d => nodeRadius(d))
    .attr('fill', d => nodeColor(d, state.colorBy))
    .attr('stroke', d => nodeStroke(d, state.colorBy))
    .attr('stroke-width', d => {
      if (d.type === 'internal_ref') return 2.5;
      if (d.is_internal_reference_target && state.colorBy === 'type') return 2.5;
      return 1.5;
    });

  nodeSel.select('text')
    .text(d => {
      if (state.view === 'timeline') return '';
      const r = nodeRadius(d);
      return r >= 10 ? _truncText(d.label, 12) : '';
    });

  nodeSel
    .on('mouseover', onNodeOver)
    .on('mousemove', onNodeMove)
    .on('mouseout', onNodeOut)
    .on('click', onNodeClick);


  sim.on('tick', () => {
    simNodes.forEach(d => {
      d.x = clamp(d.x ?? innerW / 2, 18, innerW - 18);
      d.y = clamp(d.y ?? innerH / 2, 18, innerH - 18);
    });

    linkSel
      .attr('x1', e => e.source.x || 0)
      .attr('y1', e => e.source.y || 0)
      .attr('x2', e => e.target.x || 0)
      .attr('y2', e => e.target.y || 0);

    nodeSel.attr('transform', d => `translate(${d.x || 0},${d.y || 0})`);
  });

  updateDimHighlight();
}

function _truncText(s, n) {
  if (!s) return '';
  return s.length > n ? s.slice(0,n)+'…' : s;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Drag
// ═══════════════════════════════════════════════════════════════════════════
function dragStart(event, d) {
  if (!event.active) sim && sim.alphaTarget(0.3).restart();
  d.fx = d.x; d.fy = d.y;
}
function dragged(event, d) {
  d.fx = event.x; d.fy = event.y;
}
function dragEnd(event, d) {
  if (!event.active) sim && sim.alphaTarget(0);
  d.fx = null; d.fy = null;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Highlight / dim logic
// ═══════════════════════════════════════════════════════════════════════════
function updateDimHighlight() {
  const sel = state.selectedId;
  if (!sel) {
    nodeGroup.selectAll('.node').classed('dimmed', false).classed('highlighted', false);
    linkGroup.selectAll('.link').classed('dimmed', false);
    return;
  }
  const neighbors = new Set([sel]);
  linkGroup.selectAll('.link').each(function(e) {
    const s = typeof e.source === 'object' ? e.source.id : e.source;
    const t = typeof e.target === 'object' ? e.target.id : e.target;
    if (s === sel || t === sel) { neighbors.add(s); neighbors.add(t); }
  });
  nodeGroup.selectAll('.node')
    .classed('dimmed',      d => !neighbors.has(d.id))
    .classed('highlighted', d => d.id === sel);
  linkGroup.selectAll('.link').classed('dimmed', function(e) {
    const s = typeof e.source === 'object' ? e.source.id : e.source;
    const t = typeof e.target === 'object' ? e.target.id : e.target;
    return !(s === sel || t === sel);
  });
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tooltip
// ═══════════════════════════════════════════════════════════════════════════
const tooltip = document.getElementById('tooltip');
function onNodeOver(event, d) {
  const typeMap = { paper:'Dataset Paper', reference:'External Reference', internal_ref:'Internal Reference', author:'Author', journal:'Journal', institution:'Institution', country:'Country', author_keyword:"Authors' Keyword", keyword_plus:'Keyword Plus', keyword:'Keyword' };
  document.getElementById('tt-type').textContent  = d.is_internal_reference_target ? 'Internal Reference (Dataset Paper)' : (typeMap[d.type] || d.type);
  document.getElementById('tt-title').textContent = d.label || d.id;
  const stat = d.type === 'paper'
    ? `${d.citations||0} citations · ${d.year||'?'}`
    : d.type === 'reference' || d.type === 'internal_ref'
    ? `cited by ${d.paper_count||d.citations||0} papers`
    : `${d.paper_count||0} papers`;
  document.getElementById('tt-stat').textContent  = stat;
  tooltip.style.opacity = '1';
  onNodeMove(event);
}
function onNodeMove(event) {
  tooltip.style.left = (event.clientX + 14) + 'px';
  tooltip.style.top  = (event.clientY - 14) + 'px';
}
function onNodeOut() {
  tooltip.style.opacity = '0';
}

// ═══════════════════════════════════════════════════════════════════════════
//  Node click → info panel
// ═══════════════════════════════════════════════════════════════════════════
function onNodeClick(event, d) {
  event.stopPropagation();
  if (state.selectedId === d.id) {
    state.selectedId = null;
    updateDimHighlight();
    showInfoPlaceholder();
    return;
  }
  state.selectedId = d.id;
  updateDimHighlight();
  showInfoPanel(d);
}

svgEl.on('click', () => {
  state.selectedId = null;
  updateDimHighlight();
  showInfoPlaceholder();
});

function showInfoPlaceholder() {
  document.getElementById('info-placeholder').style.display = 'flex';
  document.getElementById('info-content').style.display     = 'none';
}
showInfoPlaceholder();

function showInfoPanel(d) {
  document.getElementById('info-placeholder').style.display = 'none';
  const panel = document.getElementById('info-content');
  panel.style.display = 'flex';

  const typeColor = d.is_internal_reference_target ? TYPE_COLOR.internal_ref : (TYPE_COLOR[d.type] || '#5e7090');
  const typeMap   = {
    paper:'Dataset Paper', reference:'External Reference', internal_ref:'Internal Reference',
    author:'Author', keyword:'Keyword', journal:'Journal',
    institution:'Institution', country:'Country',
    author_keyword:"Authors' Keyword", keyword_plus:'Keyword Plus',
  };
  const ind = indicatorMap[d.id] || {};

  let html = `
    <div class="info-header">
      <div class="info-type-badge" style="background:${typeColor}22;color:${typeColor};">${d.is_internal_reference_target ? 'Internal Reference (Dataset Paper)' : (typeMap[d.type]||d.type)}</div>
      <div class="info-title">${escHtml(d.full_title || d.label)}</div>
    </div>
    <div class="info-body">`;

  // ID is shown for every node — useful for cross-reference and debugging.
  html += infoRow('ID', `<span style="font-family:var(--font-mono);font-size:10px;">${escHtml(d.id)}</span>`);

  if (d.type === 'paper') {
    html += infoRow('Year',    d.year || '—');
    html += infoRow('DOI',     d.doi  ? `<a href="https://doi.org/${d.doi}" target="_blank" style="color:var(--cyan);text-decoration:none;">${d.doi}</a>` : '—');
    if (d.authors?.length) html += infoRow('Authors', tagList(d.authors, 'author'));
    if (d.journals?.length) html += infoRow('Journal', d.journals.join('; '));
    if (d.keywords?.length) html += infoRow('Keywords', tagList(d.keywords, 'keyword'));
    if (d.is_internal_reference_target) html += infoRow('Status', `<span style="color:${TYPE_COLOR.internal_ref}">Cited in dataset</span>`);
    if (d.abstract) html += infoRow('Abstract', `<span style="font-size:10px;color:var(--muted);line-height:1.5;">${escHtml(d.abstract)}</span>`);

    html += `<div style="margin-top:10px;"><div class="panel-title" style="margin-bottom:6px;">Indicators</div><div class="ind-grid">`;
    html += indCell('Citations', d.citations ?? '—');
    html += indCell('PageRank',  ind.pagerank !== undefined ? ind.pagerank.toFixed(5) : '—');
    html += indCell('Betweenness', ind.betweenness !== undefined ? ind.betweenness.toFixed(4) : '—');
    html += indCell('Bridge Score', ind.bridge_score !== undefined ? ind.bridge_score.toFixed(4) : '—');
    html += indCell('Int. Refs', d.internal_refs ?? '—');
    html += indCell('Ext. Refs', d.external_refs ?? '—');
    html += indCell('Authors', d.author_count ?? '—');
    html += indCell('Keywords', d.keyword_count ?? '—');
    html += '</div></div>';

    html += `<button class="ego-btn" onclick="setEgoView('${d.id}')">Explore Ego View ↗</button>`;

  } else if (d.type === 'reference' || d.type === 'internal_ref') {
    html += infoRow('Cited by', `${d.citations||0} papers in dataset`);
    if (ind.first_cited_year) html += infoRow('First cited', ind.first_cited_year);
    if (ind.last_cited_year)  html += infoRow('Last cited',  ind.last_cited_year);
    if (ind.citation_velocity !== undefined) html += infoRow('Velocity', ind.citation_velocity + ' cit/yr');
    if (d.type === 'internal_ref') html += infoRow('Status', `<span style="color:${TYPE_COLOR.internal_ref}">Also in dataset</span>`);
    html += renderRelatedSections(d);
    html += `<button class="ego-btn" onclick="setLens('reference','${escHtml(d.full_title||d.id)}')">Explore Reference Lens ↗</button>`;

  } else if (d.type === 'author') {
    html += infoRow('Papers',     d.paper_count||d.citations||0);
    html += infoRow('First year', d.first_year || '—');
    html += infoRow('Last year',  d.last_year || '—');
    html += renderRelatedSections(d);
    html += `<button class="ego-btn" onclick="setLens('author','${escHtml(d.full_title||d.label)}')">Explore Author Lens ↗</button>`;

  } else if (d.type === 'author_keyword' || d.type === 'keyword_plus' || d.type === 'keyword') {
    html += infoRow('Papers',     d.paper_count||d.citations||0);
    html += infoRow('First year', d.first_year || '—');
    html += infoRow('Last year',  d.last_year || '—');
    html += renderRelatedSections(d);
    html += `<button class="ego-btn" onclick="setLens('keyword','${escHtml(d.full_title||d.label)}')">Explore Keyword Lens ↗</button>`;

  } else if (d.type === 'journal' || d.type === 'institution' || d.type === 'country') {
    html += infoRow('Papers',     d.paper_count||d.citations||0);
    html += infoRow('First year', d.first_year || '—');
    html += infoRow('Last year',  d.last_year || '—');
    html += renderRelatedSections(d);
  }

  html += '</div>';
  panel.innerHTML = html;
}

// Per-type "related entities" sections. Each list is a chip row; clicking a
// chip selects that node (auto-enabling its type toggle if needed).
const RELATED_SECTIONS = {
  author:         [['journal','Top journals'],['institution','Top institutions'],['country','Top countries'],['keywords','Top keywords']],
  journal:        [['author','Top authors'],['institution','Top institutions'],['country','Top countries'],['keywords','Top keywords']],
  institution:    [['author','Top authors'],['country','Top countries'],['journal','Top journals'],['keywords','Top keywords']],
  country:        [['institution','Top institutions'],['author','Top authors'],['journal','Top journals'],['keywords','Top keywords']],
  author_keyword: [['author','Top authors'],['journal','Top journals'],['institution','Top institutions'],['country','Top countries']],
  keyword_plus:   [['author','Top authors'],['journal','Top journals'],['institution','Top institutions'],['country','Top countries']],
  reference:      [['papers','Top citing papers']],
  internal_ref:   [['papers','Top citing papers']],
};

function renderRelatedSections(d) {
  const list = RELATED_SECTIONS[d.type];
  if (!list) return '';
  let h = '';
  list.forEach(([key, label]) => {
    let items;
    if (key === 'keywords') {
      items = topRelatedEntities(d.id, 'author_keyword', 6)
        .concat(topRelatedEntities(d.id, 'keyword_plus', 6))
        .slice(0, 8);
    } else if (key === 'papers') {
      items = topConnectedPapers(d.id, 6);
    } else {
      items = topRelatedEntities(d.id, key, 8);
    }
    if (items.length) h += infoRow(label, entityChipList(items));
  });
  return h;
}

// Clicking a chip jumps to that node's info panel. If its type toggle is off,
// turn it on first so the node actually shows up on the canvas.
function focusNode(id) {
  const node = NODE_BY_ID.get(id);
  if (!node) return;
  let needsRender = false;
  if (!state.activeTypes.has(node.type)) {
    state.activeTypes.add(node.type);
    const cb = document.querySelector(`.node-cb[data-ntype="${node.type}"]`);
    if (cb) cb.checked = true;
    needsRender = true;
  }
  state.selectedId = id;
  if (needsRender) renderGraph();
  updateDimHighlight();
  showInfoPanel(node);
}

function infoRow(key, val) {
  return `<div class="info-row"><div class="info-key">${key}</div><div class="info-val">${val}</div></div>`;
}
function indCell(key, val) {
  return `<div class="ind-cell"><div class="ind-key">${key}</div><div class="ind-val">${val}</div></div>`;
}
function tagList(items, lensType) {
  return `<div class="tag-list">${items.slice(0,8).map(item => `<span class="tag" onclick="setLens('${lensType}','${escHtml(item)}')">${escHtml(item)}</span>`).join('')}</div>`;
}
// Chips for "related entities" sections. Each chip jumps to that node's info
// panel (focusNode handles the auto-toggle if its node-type is hidden). The
// `count` shows how many papers link the current node to that one.
function entityChipList(items) {
  return `<div class="tag-list">${items.map(it =>
    `<span class="tag" onclick="focusNode('${escHtml(it.id)}')" title="${it.count} paper${it.count===1?'':'s'}">${escHtml(it.label)} <span style="opacity:.55;font-family:var(--font-mono);">${it.count}</span></span>`
  ).join('')}</div>`;
}
function escHtml(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// ═══════════════════════════════════════════════════════════════════════════
//  View switching
// ═══════════════════════════════════════════════════════════════════════════
function setView(v) {
  state.view = v;
  state.lensFilter = null;
  document.querySelectorAll('.vtab').forEach(b => b.classList.toggle('active', b.dataset.view === v));
  renderGraph();
}
function setEgoView(id) {
  state.view = 'ego';
  state.selectedId = id;
  document.querySelectorAll('.vtab').forEach(b => b.classList.toggle('active', b.dataset.view === 'ego'));
  renderGraph();
  updateDimHighlight();
}
function setLens(type, value) {
  const viewMap = { author: 'timeline', keyword: 'timeline', reference: 'timeline' };
  state.view = viewMap[type] || 'timeline';
  state.lensFilter = { type, value };
  document.querySelectorAll('.vtab').forEach(b => b.classList.toggle('active', b.dataset.view === state.view));
  renderGraph();
}

document.querySelectorAll('.vtab').forEach(b => {
  b.addEventListener('click', () => setView(b.dataset.view));
});

// ═══════════════════════════════════════════════════════════════════════════
//  Edge & node toggles
// ═══════════════════════════════════════════════════════════════════════════
document.querySelectorAll('.layer-cb').forEach(cb => {
  cb.addEventListener('change', () => {
    const layer = cb.dataset.layer;

    if (cb.checked) {
      state.activeLayers.add(layer);
    } else {
      state.activeLayers.delete(layer);
    }

    renderGraph();
  });
});

document.querySelectorAll('.node-cb').forEach(cb => {
  cb.addEventListener('change', () => {
    const nt = cb.dataset.ntype;
    cb.checked ? state.activeTypes.add(nt) : state.activeTypes.delete(nt);
    renderGraph();
  });
});

// Colour/size selects
const colorBySel = document.getElementById('color-by-sel');
const sizeBySel = document.getElementById('size-by-sel');
if (colorBySel) colorBySel.addEventListener('change', e => { state.colorBy = e.target.value; renderGraph(); });
if (sizeBySel)  sizeBySel.addEventListener('change',  e => { state.sizeBy  = e.target.value; renderGraph(); });

// Year filter
document.getElementById('btn-year-apply').addEventListener('click', () => {
  const f = parseInt(document.getElementById('yr-from').value, 10);
  const t = parseInt(document.getElementById('yr-to').value,   10);
  state.yearFrom = isNaN(f) ? null : f;
  state.yearTo   = isNaN(t) ? null : t;
  renderGraph();
});

// Reset
document.getElementById('btn-reset').addEventListener('click', () => {
  state = buildInitialState();
  syncControlsFromState();
  showInfoPlaceholder();
  renderGraph();
});

// Export HTML
// Important: export a clean, re-runnable document, not the current live D3 DOM.
// Serializing document.documentElement.outerHTML after interaction captures the
// already-rendered <svg> nodes/links. When that file is reopened, the script runs
// again and draws a second graph, leaving the old nodes fixed in the background.
function buildCleanExportHtml() {
  const clone = document.documentElement.cloneNode(true);

  // Clear runtime-rendered SVG content. The embedded JSON data and script remain
  // intact, so the graph is rendered once, cleanly, when the exported file opens.
  const svg = clone.querySelector('#main-svg');
  if (svg) {
    svg.innerHTML = '';
    svg.removeAttribute('style');
  }

  // Reset transient UI state that should not become baked into the export.
  const searchBox = clone.querySelector('#search-box');
  if (searchBox) searchBox.setAttribute('value', '');

  const searchResults = clone.querySelector('#search-results');
  if (searchResults) {
    searchResults.innerHTML = '';
    searchResults.setAttribute('style', 'display:none;');
  }

  const infoPlaceholder = clone.querySelector('#info-placeholder');
  if (infoPlaceholder) infoPlaceholder.setAttribute('style', 'display:flex;');

  const infoContent = clone.querySelector('#info-content');
  if (infoContent) {
    infoContent.innerHTML = '';
    infoContent.setAttribute('style', 'display:none;');
  }

  const tooltip = clone.querySelector('#tooltip');
  if (tooltip) tooltip.setAttribute('style', 'opacity:0;');

  const body = clone.querySelector('body');
  if (body) {
    // Remove common browser-extension attributes that sometimes appear in exports.
    [...body.attributes].forEach(attr => {
      if (attr.name.startsWith('data-gr-') || attr.name.startsWith('data-new-gr-')) {
        body.removeAttribute(attr.name);
      }
    });
  }

  return '<!DOCTYPE html>\n' + clone.outerHTML;
}

document.getElementById('btn-export').addEventListener('click', () => {
  const blob = new Blob([buildCleanExportHtml()], {type:'text/html;charset=utf-8'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'tsg_export.html';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 500);
});

// ═══════════════════════════════════════════════════════════════════════════
//  Search
// ═══════════════════════════════════════════════════════════════════════════
const searchBox = document.getElementById('search-box');
const searchResults = document.getElementById('search-results');

searchBox.addEventListener('input', () => {
  const q = searchBox.value.trim().toLowerCase();
  if (!q) { searchResults.style.display='none'; return; }
  const hits = RAW_NODES.filter(d => (d.label||'').toLowerCase().includes(q) || (d.full_title||'').toLowerCase().includes(q)).slice(0,12);
  if (!hits.length) { searchResults.style.display='none'; return; }
  searchResults.style.display='block';
  const typeMap = { paper:'Paper', reference:'Ref', internal_ref:'Int.Ref', author:'Author', journal:'Journal', institution:'Institution', country:'Country', author_keyword:'AKW', keyword_plus:'KWP', keyword:'KW' };
  searchResults.innerHTML = hits.map(d => `
    <div class="sr-item" data-id="${d.id}">
      <div class="sr-type">${typeMap[d.type]||d.type}</div>
      <div class="sr-label">${escHtml(_truncText(d.label||d.id, 60))}</div>
    </div>`).join('');
  searchResults.querySelectorAll('.sr-item').forEach(el => {
    el.addEventListener('click', () => {
      const node = RAW_NODES.find(n => n.id === el.dataset.id);
      if (!node) return;
      searchBox.value = '';
      searchResults.style.display = 'none';
      state.selectedId = node.id;
      if (node.type === 'paper') setView('timeline');
      renderGraph();
      updateDimHighlight();
      showInfoPanel(node);
    });
  });
});
document.addEventListener('click', e => { if (!e.target.closest('#search-wrap')) searchResults.style.display='none'; });

// ═══════════════════════════════════════════════════════════════════════════
//  ResizeObserver
// ═══════════════════════════════════════════════════════════════════════════
function measureCanvas() {
  const wrap = document.getElementById('svg-wrap');
  W = wrap.clientWidth;
  H = wrap.clientHeight;
}

const ro = new ResizeObserver(() => { measureCanvas(); renderGraph(); });
ro.observe(document.getElementById('svg-wrap'));

// ═══════════════════════════════════════════════════════════════════════════
//  Initial render
// ═══════════════════════════════════════════════════════════════════════════
syncControlsFromState();
measureCanvas();
renderGraph();
</script>
</body>
</html>
"""


def render_temporal_sg_html(
    nodes_df:       pd.DataFrame,
    edges_df:       pd.DataFrame,
    indicators_df:  pd.DataFrame,
    papers_df:      pd.DataFrame,
    cfg:            TSGConfig,
) -> str:
    """
    Render the complete standalone HTML string with embedded JSON data.

    Parameters
    ----------
    nodes_df, edges_df, indicators_df, papers_df : DataFrames from build_temporal_sg
    cfg : TSGConfig

    Returns
    -------
    str : full HTML page
    """
    def _df_to_records(df: pd.DataFrame) -> List[Dict]:
        if df is None or df.empty:
            return []
        records = []
        for _, row in df.iterrows():
            rec = {}
            for col, val in row.items():
                if isinstance(val, float) and np.isnan(val):
                    rec[col] = None
                elif isinstance(val, (np.integer,)):
                    rec[col] = int(val)
                elif isinstance(val, (np.floating,)):
                    rec[col] = float(val)
                elif isinstance(val, (np.bool_,)):
                    rec[col] = bool(val)
                else:
                    rec[col] = val
            records.append(rec)
        return records

    # Serialise with compact json
    nodes_json      = json.dumps(_df_to_records(nodes_df),      ensure_ascii=False, separators=(',',':'))
    edges_json      = json.dumps(_df_to_records(edges_df),      ensure_ascii=False, separators=(',',':'))
    indicators_json = json.dumps(_df_to_records(indicators_df), ensure_ascii=False, separators=(',',':'))

    # Metadata
    n_papers = int(len(nodes_df[nodes_df["type"] == "paper"])) if not nodes_df.empty else 0
    n_refs   = int(len(nodes_df[nodes_df["type"].isin(["reference","internal_ref"])])) if not nodes_df.empty else 0
    meta = {
        "n_papers":   n_papers,
        "n_refs":     n_refs,
        "n_edges":    len(edges_df) if not edges_df.empty else 0,
        "year_min":   int(papers_df["year"].min()) if not papers_df.empty and "year" in papers_df.columns else 0,
        "year_max":   int(papers_df["year"].max()) if not papers_df.empty and "year" in papers_df.columns else 9999,
        "config":     cfg.to_dict(),
        "generated":  pd.Timestamp.now().isoformat(),
        "pybibx_tsg": "1.0",
    }
    meta_json = json.dumps(meta, ensure_ascii=False, separators=(',',':'), default=str)

    html = _HTML_TEMPLATE
    html = html.replace("__NODES_JSON__",      nodes_json)
    html = html.replace("__EDGES_JSON__",      edges_json)
    html = html.replace("__INDICATORS_JSON__", indicators_json)
    html = html.replace("__META_JSON__",       meta_json)

    return html


# ──────────────────────────────────────────────────────────────────────────────
# §7  Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def temporal_sg(
    pbx,
    view:                   str             = "timeline",
    layers:                 Optional[List[str]] = None,
    time_mode:              str             = "range",
    start_year:             Optional[int]   = None,
    end_year:               Optional[int]   = None,
    center:                 str             = "paper",
    selected:               Optional[str]   = None,
    max_papers:             int             = 500,
    max_references:         int             = 300,
    color_by:               str             = "type",
    size_by:                str             = "citations",
    notebook:               bool            = True,
    open_browser:           bool            = True,
    save_html:              Optional[str]   = None,
    preview:                bool            = True,
) -> Dict[str, Any]:
    """

    Parameters
    ----------
    pbx : pyBibX object
        The loaded bibliometric dataset.
    view : str
        Initial view: 'timeline' | 'force' | 'ego' | 'author' | 'reference'.
    layers : list of str
        Active edge layers on load. Default: ['citations'].
    time_mode : str
        'range' (use start_year/end_year) | 'all'.
    start_year, end_year : int or None
        Year range filter. None = use dataset extremes.
    center : str
        Entity type to centre the view on ('paper').
    selected : str or None
        ID or label of the initially selected node.
    max_papers : int
        Maximum number of dataset papers to include in the graph.
    max_references : int
        Maximum number of external references to include.
    color_by : str
        Node colouring scheme. Options depend on `center`; common values
        include 'type', 'citations', 'papers', 'year'.
    size_by : str
        Node sizing. Options depend on `center`; common values include
        'citations', 'papers', 'references', 'uniform'.
    notebook : bool
        If True and IPython is available, display inline.
    open_browser : bool
        If True, open the result in the system browser.
    save_html : str or None
        If given, path to write the standalone HTML file.
    preview : bool
        If True, generate and return the HTML.

    Returns
    -------
    dict with keys:
        'html'        : HTML string (or '' if preview=False)
        'nodes'       : pd.DataFrame of all graph nodes
        'edges'       : pd.DataFrame of all graph edges
        'papers'      : pd.DataFrame of dataset papers only
        'indicators'  : pd.DataFrame of per-node indicators
        'config'      : dict of active configuration
    """
    if layers is None:
        layers = ["citations"]

    cfg = TSGConfig(
        view                   = view,
        layers                 = layers,
        time_mode              = time_mode,
        start_year             = start_year,
        end_year               = end_year,
        center                 = center,
        selected               = selected,
        max_papers             = max_papers,
        max_references         = max_references,
        color_by               = color_by,
        size_by                = size_by,
        notebook               = notebook,
        open_browser           = open_browser,
        save_html              = save_html,
        preview                = preview,
    )

    # ── Build graph ───────────────────────────────────────────────────────
    print("[TSG] Building Temporal Scholarly Graph…")
    nodes_df, edges_df, papers_df, G = build_temporal_sg(pbx, cfg)
    print(f"[TSG] Nodes: {len(nodes_df)}  Edges: {len(edges_df)}  Papers: {len(papers_df)}")

    # ── Compute indicators ────────────────────────────────────────────────
    print("[TSG] Computing indicators…")
    indicators_df = _compute_indicators(nodes_df, edges_df, G, cfg)

    # ── Generate HTML ─────────────────────────────────────────────────────
    html_str = ""
    if preview:
        print("[TSG] Rendering HTML…")
        html_str = render_temporal_sg_html(nodes_df, edges_df, indicators_df, papers_df, cfg)

        # Save to file
        if save_html:
            out_path = os.path.abspath(save_html)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html_str)
            print(f"[TSG] HTML saved → {out_path}")

        # Open in browser
        if open_browser and save_html:
            webbrowser.open(f"file://{os.path.abspath(save_html)}")
        elif open_browser and not save_html:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8")
            tmp.write(html_str)
            tmp.close()
            webbrowser.open(f"file://{tmp.name}")
            print(f"[TSG] Opened in browser (temp file: {tmp.name})")

        # IPython display
        if notebook and _IPY_AVAILABLE and save_html:
            try:
                _ipy_display(IFrame(src=save_html, width="100%", height="780px"))
            except Exception:
                pass
        elif notebook and _IPY_AVAILABLE and not save_html:
            from IPython.display import HTML as _HTML
            try:
                _ipy_display(_HTML(html_str))
            except Exception:
                pass

    print("[TSG] Done.")

    return {
        "html":       html_str,
        "nodes":      nodes_df,
        "edges":      edges_df,
        "papers":     papers_df,
        "indicators": indicators_df,
        "config":     cfg.to_dict(),
    }
