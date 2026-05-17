############################################################################
# Created for: pyBibX - Advanced Scientometric Analytics
# Author:      Valdecy Pereira, D.Sc.
# Module:      pybibx/base/advanced.py
#
# Public API (each function takes a pbx_probe instance as first arg,
# matching the delegation pattern of pybibx.base.tsg):
#
#   portfolio_analysis(pbx, ...)        - BCG-style productivity x impact
#   specialization_analysis(pbx, ...)   - Activity Index / RCA / share matrix
#   collaboration_impact(pbx, ...)      - Solo vs collab impact + centralities
#   collaboration_brokerage(pbx, ...)   - Collaboration brokerage / bridge roles
#   normalize_entities(pbx, ...)        - Detect candidate entity normalization mappings
#   apply_entity_normalization(pbx, ...) - Apply reviewed entity normalization mappings
#   burst_detection(pbx, ...)           - Kleinberg 2-state discrete bursts
#   knowledge_diffusion(pbx, ...)       - Temporal or citation-driven diffusion
#   reference_diversity(pbx, ...)        - Reference-base breadth, age, source entropy
#   disruption_index(pbx, ...)           - Disruptive/developmental citation behavior
#
# Each function:
#   - returns a pandas.DataFrame
#   - stores the table on the pbx instance (e.g. pbx.portfolio_analysis_table)
#   - renders a Plotly figure when view in {'browser', 'notebook'}
############################################################################

from __future__ import annotations

import math
import csv
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from io import StringIO
from typing      import Any, Dict, List, Optional, Tuple

import numpy  as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Lazy plotly bootstrap
# ──────────────────────────────────────────────────────────────────────────────

def _lazy_plotly():
    try:
        import plotly.graph_objects as go
        import plotly.io            as pio
        return go, pio
    except ImportError as e:
        raise ImportError(
            "plotly is required for visualization in pybibx.advanced. "
            "Install with: pip install plotly"
        ) from e


def _render_figure(fig, view: Optional[str]):
    if view == 'browser':
        _, pio = _lazy_plotly()
        pio.renderers.default = 'browser'
        fig.show()
    elif view == 'notebook':
        fig.show()
    # else: caller keeps the figure object via return value path; we return nothing


# ──────────────────────────────────────────────────────────────────────────────
# Design tokens — refined visual style applied across all figures
# ──────────────────────────────────────────────────────────────────────────────

FONT_FAMILY = ('"Inter", "SF Pro Text", -apple-system, BlinkMacSystemFont, '
               '"Segoe UI", Roboto, sans-serif')

PALETTE: Dict[str, Any] = {
    # Quadrant accents (Portfolio, Collaboration)
    'stars':       '#0d9488',   # teal-600
    'emerging':    '#6366f1',   # indigo-500
    'mature':      '#d97706',   # amber-600
    'marginal':    '#64748b',   # slate-500
    # Burst gradient (amber → rose); stored as RGB triples for interpolation
    'burst_low':   (254, 243, 199),
    'burst_high':  (159,  18,  57),
    # Concept cycle (Chord, Sankey link colors)
    'concept_cycle': [
        '#0d9488', '#6366f1', '#d97706', '#dc2626',
        '#7c3aed', '#0891b2', '#65a30d', '#db2777',
        '#ea580c', '#0284c7', '#84cc16', '#9333ea',
    ],
    # Text & chrome
    'text_strong': '#0f172a',   # slate-900
    'text_normal': '#334155',   # slate-700
    'text_soft':   '#64748b',   # slate-500
    'text_faint':  '#cbd5e1',   # slate-300
    'bg_panel':    '#ffffff',
    'bg_paper':    '#fafafa',
    'grid':        '#e2e8f0',   # slate-200
    # Quadrant background washes (very subtle, layered behind markers)
    'wash_stars':    'rgba(13,148,136,0.06)',
    'wash_emerging': 'rgba(99,102,241,0.06)',
    'wash_mature':   'rgba(217,119,6,0.06)',
    'wash_marginal': 'rgba(100,116,139,0.04)',
}

# Custom colorscales (sequential & diverging) used by heatmaps
_DIVERGING_SCALE = [
    [0.0, '#1e40af'],   # blue-800
    [0.5, '#f8fafc'],   # slate-50 (neutral midpoint)
    [1.0, '#b91c1c'],   # red-700
]
_TIME_SCALE = [
    [0.0, '#1e3a8a'],   # blue-900 (oldest)
    [0.5, '#7c3aed'],   # violet-600
    [1.0, '#f97316'],   # orange-500 (most recent)
]
_SHARE_SCALE = [
    [0.0, '#f1f5f9'],   # slate-100
    [0.5, '#7c3aed'],   # violet-600
    [1.0, '#1e1b4b'],   # indigo-950
]


def _base_layout(title: str, subtitle: Optional[str] = None,
                 height: int = 620, width: int = 980) -> Dict[str, Any]:
    """Shared Plotly layout chrome — title, fonts, paper, margins, hover."""
    title_html = f'<b>{title}</b>'
    if subtitle:
        title_html += (f'<br><span style="font-size:12px;color:{PALETTE["text_soft"]};'
                       f'font-weight:400">{subtitle}</span>')
    return dict(
        title=dict(
            text=title_html,
            font=dict(family=FONT_FAMILY, size=17, color=PALETTE['text_strong']),
            x=0.02, xanchor='left', y=0.96, yanchor='top',
        ),
        font=dict(family=FONT_FAMILY, size=11, color=PALETTE['text_normal']),
        plot_bgcolor=PALETTE['bg_panel'],
        paper_bgcolor=PALETTE['bg_paper'],
        height=height, width=width,
        margin=dict(l=70, r=40, t=100, b=70),
        hoverlabel=dict(
            bgcolor='white',
            bordercolor=PALETTE['text_faint'],
            font=dict(family=FONT_FAMILY, size=11, color=PALETTE['text_strong']),
        ),
    )


def _clean_axis(title: Optional[str] = None,
                show_grid: bool = True) -> Dict[str, Any]:
    """Refined axis style — subtle grid, no zeroline, clean ticks."""
    out: Dict[str, Any] = dict(
        gridcolor=PALETTE['grid'], gridwidth=1, showgrid=show_grid,
        zeroline=False,
        showline=True, linecolor=PALETTE['text_faint'], linewidth=1,
        ticks='outside', tickcolor=PALETTE['text_faint'], ticklen=4,
        tickfont=dict(family=FONT_FAMILY, size=10, color=PALETTE['text_soft']),
    )
    if title:
        out['title'] = dict(text=f'<b>{title}</b>',
                            font=dict(family=FONT_FAMILY, size=12,
                                      color=PALETTE['text_strong']))
    return out


def _interp_rgb(rgb_lo: Tuple[int, int, int], rgb_hi: Tuple[int, int, int],
                t: float) -> str:
    """Linearly interpolate two RGB triples to a CSS rgb() string."""
    t = max(0.0, min(1.0, float(t)))
    r = int(rgb_lo[0] + (rgb_hi[0] - rgb_lo[0]) * t)
    g = int(rgb_lo[1] + (rgb_hi[1] - rgb_lo[1]) * t)
    b = int(rgb_lo[2] + (rgb_hi[2] - rgb_lo[2]) * t)
    return f'rgb({r},{g},{b})'


# ──────────────────────────────────────────────────────────────────────────────
# Entity aliases — resolve user-facing names to pbx_probe attributes
# ──────────────────────────────────────────────────────────────────────────────

# (per_paper_attr, unique_list_attr, count_attr_or_None, citation_attr_or_None)
_ENTITY_TABLE: Dict[str, Tuple[str, str, Optional[str], Optional[str]]] = {
    'aut':  ('aut', 'u_aut',  'aut_docs',   'aut_cit'),
    'cout': ('ctr', 'u_ctr',  'ctr_count',  'ctr_cit'),
    'inst': ('uni', 'u_uni',  'uni_count',  'uni_cit'),
    'jou':  ('jou', 'u_jou',  'jou_count',  'jou_cit'),
    'kwa':  ('auk', 'u_auk',  'auk_count',  None),
    'kwp':  ('kid', 'u_kid',  'kid_count',  None),
    'ref':  ('ref', 'u_ref',  None,         None),
    'lan':  ('lan', 'u_lan',  'lan_count',  None),
}

_ENTITY_ALIASES: Dict[str, str] = {
    'author': 'aut', 'authors': 'aut', 'aut': 'aut',
    'country': 'cout', 'countries': 'cout', 'cout': 'cout', 'ctr': 'cout',
    'institution': 'inst', 'institutions': 'inst', 'affiliation': 'inst',
    'affiliations': 'inst', 'inst': 'inst', 'uni': 'inst', 'university': 'inst',
    'source': 'jou', 'sources': 'jou', 'journal': 'jou', 'journals': 'jou',
    'jou': 'jou',
    'keyword': 'kwa', 'keywords': 'kwa', 'author_keyword': 'kwa',
    'author_keywords': 'kwa', 'kwa': 'kwa', 'auk': 'kwa',
    'keyword_plus': 'kwp', 'keywords_plus': 'kwp', 'kwp': 'kwp', 'kid': 'kwp',
    'reference': 'ref', 'references': 'ref', 'ref': 'ref',
    'language': 'lan', 'languages': 'lan', 'lan': 'lan',
}

_ENTITY_PRETTY: Dict[str, str] = {
    'aut':  'Authors',
    'cout': 'Countries',
    'inst': 'Institutions',
    'jou':  'Sources',
    'kwa':  'Author Keywords',
    'kwp':  'Keywords Plus',
    'ref':  'References',
    'lan':  'Languages',
}


def _resolve_entity_alias(value: Any, default: str = 'aut') -> str:
    if value is None:
        return default
    key = str(value).strip().lower()
    if key in _ENTITY_TABLE:
        return key
    if key in _ENTITY_ALIASES:
        return _ENTITY_ALIASES[key]
    valid = sorted(set(list(_ENTITY_TABLE.keys()) + list(_ENTITY_ALIASES.keys())))
    raise ValueError(f"Unknown entity '{value}'. Valid options: {valid}")


def _get_entity_data(pbx, alias: str) -> Tuple[List[List[str]], List[str], List[float], List[float]]:
    """
    Return (per_paper, u_list, counts, citations) for a given entity.
    Computes missing pieces (counts, per-entity citations) on demand.
    """
    per_paper_attr, u_attr, count_attr, cit_attr = _ENTITY_TABLE[alias]
    per_paper = getattr(pbx, per_paper_attr, []) or []
    u_list    = list(getattr(pbx, u_attr, []) or [])

    # counts
    if count_attr is not None and hasattr(pbx, count_attr):
        try:
            counts = [int(c) for c in getattr(pbx, count_attr)]
        except Exception:
            counts = []
    else:
        counts = []
    if not counts or len(counts) != len(u_list):
        idx_map = {it: k for k, it in enumerate(u_list)}
        counts  = [0] * len(u_list)
        for items in per_paper:
            for it in (items or []):
                k = idx_map.get(it)
                if k is not None:
                    counts[k] += 1

    # citations
    if cit_attr is not None and hasattr(pbx, cit_attr):
        try:
            cits = [float(c) for c in getattr(pbx, cit_attr)]
        except Exception:
            cits = []
    else:
        cits = []
    if not cits or len(cits) != len(u_list):
        idx_map = {it: k for k, it in enumerate(u_list)}
        cits    = [0.0] * len(u_list)
        citations = getattr(pbx, 'citation', []) or []
        for paper_idx, items in enumerate(per_paper):
            if paper_idx >= len(citations):
                continue
            try:
                c = float(citations[paper_idx]) if citations[paper_idx] is not None else 0.0
            except Exception:
                c = 0.0
            for it in (items or []):
                k = idx_map.get(it)
                if k is not None:
                    cits[k] += c

    return per_paper, u_list, counts, cits


def _resolve_years(pbx) -> np.ndarray:
    """Return per-paper year array (np.ndarray of ints; -1 if missing)."""
    if hasattr(pbx, 'dy') and pbx.dy is not None:
        y = pd.to_numeric(pd.Series(pbx.dy), errors='coerce')
    elif hasattr(pbx, 'data') and pbx.data is not None and 'year' in pbx.data.columns:
        y = pd.to_numeric(pbx.data['year'], errors='coerce')
    else:
        return np.array([], dtype=int)
    return y.fillna(-1).astype(int).to_numpy()


def _year_bounds(pbx, start=None, end=None) -> Tuple[int, int]:
    y_lo = int(getattr(pbx, 'date_str', 0) or 0)
    y_hi = int(getattr(pbx, 'date_end', 0) or 0)
    try:
        s = int(start) if start is not None else y_lo
    except Exception:
        s = y_lo
    try:
        e = int(end) if end is not None else y_hi
    except Exception:
        e = y_hi
    if s < y_lo or s == -1:
        s = y_lo
    if e > y_hi or e == -1:
        e = y_hi
    if e < s:
        s, e = e, s
    return s, e


def _format_label(text: Any, maxlen: int = 50) -> str:
    s = str(text)
    if len(s) > maxlen:
        return s[:maxlen - 1] + '…'
    return s


def _safe_div(a: float, b: float, fallback: float = 0.0) -> float:
    try:
        if b == 0 or b is None:
            return fallback
        return a / b
    except Exception:
        return fallback


def _is_unknown(value: Any) -> bool:
    if value is None:
        return True
    s = str(value).strip().lower()
    return s == '' or s == 'unknown'


def _clean_entity_items(items, drop_unknown: bool = True) -> List[str]:
    """Clean, deduplicate, and preserve order for per-paper entity lists."""
    seen = set()
    out: List[str] = []
    for item in (items or []):
        s = str(item).strip()
        if not s:
            continue
        if drop_unknown and _is_unknown(s):
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _minmax(values: pd.Series, fallback: float = 0.0) -> pd.Series:
    """Return min-max normalized numeric values as a pandas Series."""
    vals = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if vals.notna().sum() == 0:
        return pd.Series([fallback] * len(vals), index=values.index, dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    if hi == lo:
        return pd.Series([0.5] * len(vals), index=values.index, dtype=float)
    return ((vals - lo) / (hi - lo)).fillna(fallback).astype(float)


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    """Stable undirected pair key."""
    return tuple(sorted((str(a), str(b))))


# ──────────────────────────────────────────────────────────────────────────────
# Entity normalization helpers
# ──────────────────────────────────────────────────────────────────────────────

_NORMALIZATION_MAIN_ALIASES = ['aut', 'inst', 'cout', 'jou', 'kwa', 'kwp', 'ref']

_NORMALIZATION_SOURCE_COLUMNS: Dict[str, List[str]] = {
    'aut':  ['author'],
    'kwa':  ['author_keywords'],
    'kwp':  ['keywords'],
    'jou':  ['abbrev_source_title', 'source_title'],
    'ref':  ['references'],
    'lan':  ['language'],
    'inst': ['institution', 'affiliation', 'affiliation_', 'correspondence_address1'],
    'cout': ['country', 'CU', 'affiliation', 'affiliation_', 'correspondence_address1'],
}

_TOKENIZED_COLUMNS = {
    'author': ' and ',
    'author_keywords': ';',
    'keywords': ';',
    'references': ';',
    'abbrev_source_title': ';',
    'source_title': ';',
    'language': '.',
    'country': ';',
    'CU': ';',
    'institution': ';',
}


def _strip_accents(text: Any) -> str:
    text = '' if text is None else str(text)
    text = unicodedata.normalize('NFKD', text)
    return ''.join(ch for ch in text if not unicodedata.combining(ch))


def _norm_compare(text: Any) -> str:
    """Comparison key used only for matching; display strings are preserved."""
    s = _strip_accents(text).lower().strip()
    s = s.replace('&', ' and ')
    s = re.sub(r"['’`´]", '', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _token_similarity(a: str, b: str) -> float:
    ca, cb = _norm_compare(a), _norm_compare(b)
    if not ca or not cb:
        return 0.0
    sa, sb = set(ca.split()), set(cb.split())
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    denom = max(len(sa), len(sb))
    jacc = inter / denom if denom else 0.0
    seq = SequenceMatcher(None, ' '.join(sorted(sa)), ' '.join(sorted(sb))).ratio()
    return float(max(jacc, seq))


def _entity_similarity(a: str, b: str, method: str = 'fuzzy') -> float:
    ca, cb = _norm_compare(a), _norm_compare(b)
    if not ca or not cb:
        return 0.0
    if method == 'exact_clean':
        return 1.0 if ca == cb else 0.0
    if method == 'token':
        return _token_similarity(a, b)
    return float(max(SequenceMatcher(None, ca, cb).ratio(), _token_similarity(a, b) * 0.96))


def _normalization_bucket_key(text: str) -> str:
    c = _norm_compare(text)
    if not c:
        return ''
    toks = c.split()
    if toks:
        return toks[0][0]
    return c[0]


def _safe_mapping_rows_from_text(text: str, sep: str = ';') -> List[List[str]]:
    rows: List[List[str]] = []
    if text is None:
        return rows
    text = str(text).lstrip('\ufeff')
    reader = csv.reader(StringIO(text), delimiter=sep, quotechar='"')
    for row in reader:
        vals = [str(v).strip() for v in row]
        if not vals or all(v == '' for v in vals):
            continue
        rows.append(vals)
    return rows


def _mapping_dict_from_pairs(pairs: List[Tuple[str, str]], case_sensitive: bool = False) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    clean_mapping: Dict[str, str] = {}
    conflicts: Dict[str, set] = defaultdict(set)
    for candidate, normalized in pairs:
        cand = '' if candidate is None else str(candidate).strip()
        norm = '' if normalized is None else str(normalized).strip()
        if not cand or not norm or cand == norm:
            continue
        key = cand if case_sensitive else cand.lower()
        if key in mapping and mapping[key] != norm:
            conflicts[cand].update({mapping[key], norm})
        mapping[key] = norm
        ckey = _norm_compare(cand)
        if ckey:
            if ckey in clean_mapping and clean_mapping[ckey] != norm:
                conflicts[cand].update({clean_mapping[ckey], norm})
            clean_mapping[ckey] = norm
    if conflicts:
        msg = '; '.join(f"{k!r} -> {sorted(v)}" for k, v in conflicts.items())
        raise ValueError(f'Ambiguous normalization mapping detected: {msg}')
    # Store cleaned keys with a sentinel prefix so exact and clean matching can coexist.
    for k, v in clean_mapping.items():
        mapping['__clean__' + k] = v
    return mapping


def _replace_token_value(value: Any, mapping: Dict[str, str], case_sensitive: bool = False) -> str:
    s = '' if value is None or pd.isna(value) else str(value).strip()
    if not s:
        return s
    key = s if case_sensitive else s.lower()
    if key in mapping:
        return mapping[key]
    ckey = '__clean__' + _norm_compare(s)
    if ckey in mapping:
        return mapping[ckey]
    return s


def _split_author_field(text: str) -> List[str]:
    return [p.strip() for p in re.split(r'\s+and\s+', str(text), flags=re.IGNORECASE) if p.strip()]


def _replace_in_tokenized_cell(value: Any, sep_token: str, mapping: Dict[str, str], case_sensitive: bool = False) -> str:
    if value is None or pd.isna(value):
        return ''
    text = str(value)
    if not text.strip():
        return text
    if sep_token == ' and ':
        parts = _split_author_field(text)
        return ' and '.join(_replace_token_value(p, mapping, case_sensitive=case_sensitive) for p in parts)
    sep = sep_token
    parts = [p.strip() for p in text.split(sep)]
    if len(parts) <= 1:
        return _replace_token_value(text, mapping, case_sensitive=case_sensitive)
    return (sep + ' ').join(_replace_token_value(p, mapping, case_sensitive=case_sensitive) for p in parts)


def _replace_in_free_text(value: Any, pairs: List[Tuple[str, str]], case_sensitive: bool = False) -> str:
    if value is None or pd.isna(value):
        return ''
    text = str(value)
    if not text:
        return text
    flags = 0 if case_sensitive else re.IGNORECASE
    # Longest candidates first avoids partial replacement before longer variants.
    for candidate, normalized in sorted(pairs, key=lambda kv: len(str(kv[0])), reverse=True):
        cand = str(candidate).strip()
        norm = str(normalized).strip()
        if not cand or not norm or cand == norm:
            continue
        pattern = re.compile(r'(?<!\w)' + re.escape(cand) + r'(?!\w)', flags=flags)
        text = pattern.sub(norm, text)
    return text


def _candidate_pairs_from_mapping_rows(rows: List[List[str]], entity: str = 'aut') -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse no-header mapping rows.

    Two-column format, for a single entity:
        normalized_string ; candidate_string

    Three-column mixed format, for entity='all':
        entity ; normalized_string ; candidate_string
    """
    out: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    global_alias = _resolve_entity_alias(entity, default='aut') if str(entity).strip().lower() != 'all' else 'all'
    for row in rows:
        if len(row) >= 3 and str(row[0]).strip().lower() in set(_ENTITY_ALIASES.keys()) | set(_ENTITY_TABLE.keys()):
            alias = _resolve_entity_alias(row[0], default='aut')
            normalized, candidate = row[1], row[2]
        elif len(row) >= 2:
            aliases = _NORMALIZATION_MAIN_ALIASES if global_alias == 'all' else [global_alias]
            normalized, candidate = row[0], row[1]
            for alias in aliases:
                if str(normalized).strip() and str(candidate).strip() and str(normalized).strip() != str(candidate).strip():
                    out[alias].append((str(candidate).strip(), str(normalized).strip()))
            continue
        else:
            continue
        if str(normalized).strip() and str(candidate).strip() and str(normalized).strip() != str(candidate).strip():
            out[alias].append((str(candidate).strip(), str(normalized).strip()))
    return dict(out)


def _apply_pairs_to_dataframe(df: pd.DataFrame, alias: str, pairs: List[Tuple[str, str]], case_sensitive: bool = False) -> pd.DataFrame:
    if df is None or len(pairs) == 0:
        return df
    out = df.copy(deep=True)
    mapping = _mapping_dict_from_pairs(pairs, case_sensitive=case_sensitive)
    for col in _NORMALIZATION_SOURCE_COLUMNS.get(alias, []):
        if col not in out.columns:
            continue
        sep_token = _TOKENIZED_COLUMNS.get(col)
        if sep_token is not None:
            out[col] = out[col].apply(lambda x: _replace_in_tokenized_cell(x, sep_token, mapping, case_sensitive=case_sensitive))
        else:
            out[col] = out[col].apply(lambda x: _replace_in_free_text(x, pairs, case_sensitive=case_sensitive))
    return out


def _normalization_export(df: pd.DataFrame, export_path: Optional[str], sep: str = ';') -> None:
    if not export_path:
        return
    export_df = df[['Normalized String', 'Candidate String']].copy() if not df.empty else pd.DataFrame(columns=['Normalized String', 'Candidate String'])
    export_df.to_csv(export_path, sep=sep, header=False, index=False, quoting=csv.QUOTE_ALL)


def normalize_entities(
    pbx,
    entity: str = 'all',
    method: str = 'fuzzy',
    threshold: float = 0.88,
    min_count: int = 1,
    max_items: Optional[int] = 500,
    sep: str = ';',
    export_path: Optional[str] = None,
    view: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Detect candidate entity-normalization mappings without modifying the dataset.

    The returned table uses the safe manual workflow format:
    ``Normalized String`` ; ``Candidate String``.  To reject a suggested mapping,
    users delete the row before applying it with ``apply_entity_normalization``.

    Parameters
    ----------
    pbx : pybibx pbx instance
        Active PyBibX object.
    entity : {'all', 'aut', 'inst', 'cout', 'jou', 'kwa', 'kwp', 'ref'} or aliases
        Entity family to inspect.
    method : {'fuzzy', 'token', 'exact_clean'}, default='fuzzy'
        Similarity strategy used for detection.
    threshold : float, default=0.88
        Similarity threshold from 0 to 1.
    min_count : int, default=1
        Minimum observed frequency for a string to be inspected.
    max_items : int or None, default=500
        Safety cap per entity family. Use None to compare all unique strings.
    sep : str, default=';'
        Delimiter used when export_path is provided.
    export_path : str or None
        Optional no-header mapping CSV path. Fields are quoted.
    view : {'browser', 'notebook', None}, default=None
        If not None, shows a compact Plotly bar chart of candidate counts.
    verbose : bool, default=True
        Print a compact summary.

    Returns
    -------
    pandas.DataFrame
        Candidate mapping table with canonical/candidate strings and diagnostics.
    """
    method = str(method or 'fuzzy').strip().lower()
    if method not in {'fuzzy', 'token', 'exact_clean'}:
        raise ValueError("method must be 'fuzzy', 'token', or 'exact_clean'")
    try:
        threshold = float(threshold)
    except Exception:
        threshold = 0.88
    threshold = max(0.0, min(1.0, threshold))
    min_count = max(1, int(min_count or 1))

    aliases = _NORMALIZATION_MAIN_ALIASES if str(entity).strip().lower() == 'all' else [_resolve_entity_alias(entity, default='aut')]
    rows: List[Dict[str, Any]] = []

    for alias in aliases:
        _, u_list, counts, _ = _get_entity_data(pbx, alias)
        count_map = {str(item): int(counts[i]) if i < len(counts) else 1 for i, item in enumerate(u_list)}
        values = [str(v).strip() for v in u_list if str(v).strip() and not _is_unknown(v) and count_map.get(str(v), 1) >= min_count]
        # Keep the most frequent values when a safety cap is requested.
        if max_items is not None and len(values) > int(max_items):
            values = sorted(values, key=lambda x: (count_map.get(x, 0), len(x)), reverse=True)[:int(max_items)]
        if len(values) < 2:
            continue

        # Blocking drastically reduces accidental O(n²) comparisons for large reference lists.
        buckets: Dict[str, List[str]] = defaultdict(list)
        for val in values:
            buckets[_normalization_bucket_key(val)].append(val)

        parent = {v: v for v in values}
        best_sim: Dict[Tuple[str, str], float] = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for bucket_vals in buckets.values():
            n = len(bucket_vals)
            for i in range(n):
                a = bucket_vals[i]
                ca = _norm_compare(a)
                if not ca:
                    continue
                for j in range(i + 1, n):
                    b = bucket_vals[j]
                    cb = _norm_compare(b)
                    if not cb:
                        continue
                    # Cheap length guard for fuzzy/token; exact_clean still needs equality check.
                    if method != 'exact_clean':
                        lmin, lmax = min(len(ca), len(cb)), max(len(ca), len(cb))
                        if lmax and lmin / lmax < 0.45:
                            continue
                    sim = _entity_similarity(a, b, method=method)
                    if sim >= threshold and a != b:
                        union(a, b)
                        best_sim[_pair_key(a, b)] = sim

        comps: Dict[str, List[str]] = defaultdict(list)
        for val in values:
            comps[find(val)].append(val)

        for comp in comps.values():
            if len(comp) < 2:
                continue
            canonical = sorted(comp, key=lambda x: (count_map.get(x, 0), len(_norm_compare(x)), len(x)), reverse=True)[0]
            for cand in sorted(comp, key=lambda x: x.lower()):
                if cand == canonical:
                    continue
                sim = best_sim.get(_pair_key(canonical, cand), _entity_similarity(canonical, cand, method=method))
                rows.append({
                    'Entity': _ENTITY_PRETTY.get(alias, alias),
                    'Entity Alias': alias,
                    'Normalized String': canonical,
                    'Candidate String': cand,
                    'Similarity': round(float(sim), 4),
                    'Normalized Count': int(count_map.get(canonical, 0)),
                    'Candidate Count': int(count_map.get(cand, 0)),
                })

    df = pd.DataFrame(rows, columns=['Entity', 'Entity Alias', 'Normalized String', 'Candidate String', 'Similarity', 'Normalized Count', 'Candidate Count'])
    if not df.empty:
        df = df.sort_values(['Entity Alias', 'Similarity', 'Normalized Count', 'Candidate Count'], ascending=[True, False, False, False]).reset_index(drop=True)

    pbx.normalize_entities_table = df
    pbx.ask_gpt_normalize_entities = df
    _normalization_export(df, export_path, sep=sep)

    if verbose:
        print(f'[normalize_entities] candidates={len(df)} | entity={entity} | method={method} | threshold={threshold}')

    if view is not None and not df.empty:
        go, _ = _lazy_plotly()
        counts_df = df.groupby('Entity', as_index=False).size().rename(columns={'size': 'Candidate Rows'})
        fig = go.Figure(go.Bar(
            x=counts_df['Entity'], y=counts_df['Candidate Rows'],
            text=counts_df['Candidate Rows'], textposition='outside',
            marker=dict(color=PALETTE['stars'], line=dict(color='rgba(15,23,42,0.18)', width=1)),
        ))
        layout = _base_layout('Entity Normalization Candidates', 'Rows suggested for manual review before applying normalization', height=520, width=880)
        layout.update(xaxis=_clean_axis('Entity'), yaxis=_clean_axis('Candidate rows'), margin=dict(l=70, r=40, t=100, b=80))
        fig.update_layout(**layout)
        pbx.normalize_entities_figure = fig
        _render_figure(fig, view)

    return df


def apply_entity_normalization(
    pbx,
    mapping_file: Optional[str] = None,
    mapping_text: Optional[str] = None,
    entity: str = 'aut',
    sep: str = ';',
    inplace: bool = True,
    case_sensitive: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply a reviewed no-header entity-normalization mapping and recompute PyBibX.

    Accepted CSV formats
    --------------------
    Single-entity, two-column format::

        "normalized string";"candidate string"

    Mixed-entity, three-column format::

        "entity";"normalized string";"candidate string"

    The second value is replaced by the first. To reject a candidate, delete the
    row from the mapping file before applying it.
    """
    if mapping_text is None and mapping_file is None:
        raise ValueError('Provide mapping_text or mapping_file')
    if mapping_text is None:
        with open(mapping_file, 'r', encoding='utf-8-sig', newline='') as fh:
            mapping_text = fh.read()
    rows = _safe_mapping_rows_from_text(mapping_text, sep=sep)
    mappings = _candidate_pairs_from_mapping_rows(rows, entity=entity)
    if not mappings:
        raise ValueError('No valid normalization rows were found. Expected: normalized_string;candidate_string')

    data_before = getattr(pbx, 'data', None)
    if data_before is None:
        raise ValueError('No dataset is loaded')
    data_after = data_before.copy(deep=True)
    applied_rows = []
    for alias, pairs in mappings.items():
        if not pairs:
            continue
        data_after = _apply_pairs_to_dataframe(data_after, alias, pairs, case_sensitive=case_sensitive)
        for candidate, normalized in pairs:
            applied_rows.append({
                'Entity Alias': alias,
                'Entity': _ENTITY_PRETTY.get(alias, alias),
                'Normalized String': normalized,
                'Candidate String': candidate,
            })

    if inplace:
        pbx.load_database_df(data_after)

    report = pd.DataFrame(applied_rows, columns=['Entity', 'Entity Alias', 'Normalized String', 'Candidate String'])
    pbx.entity_normalization_report = report
    pbx.ask_gpt_entity_normalization = report
    if verbose:
        print(f'[apply_entity_normalization] applied_rows={len(report)} | inplace={inplace}')
    return report


# ══════════════════════════════════════════════════════════════════════════════
# §1  Portfolio Analysis (BCG-style productivity × impact quadrants)
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_analysis(
    pbx,
    entity: str = 'jou',
    productivity: str = 'documents',
    impact: str = 'citations',
    thresholds: Any = 'median',
    topn: Optional[int] = None,
    drop_unknown: bool = True,
    view: Optional[str] = 'browser',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Classify entities (journals / authors / countries / institutions / keywords /
    references) into a Stars / Emerging / Mature-Core / Marginal quadrant matrix
    based on productivity and impact.

    Parameters
    ----------
    entity : str
        Entity alias ('jou', 'aut', 'cout', 'inst', 'kwa', 'kwp', 'ref')
        or any friendly synonym ('journals', 'countries', ...).
    productivity : str
        'documents' (count of documents per entity). Currently the only metric.
    impact : str
        'citations' (total), 'mean_citations', or 'h_index' (per-entity local).
    thresholds : 'median' | 'mean' | tuple(prod_cut, impact_cut)
        Quadrant cutoffs. Median (default) is robust to outliers.
    topn : int or None
        If set, keep only the top-N entities by combined (z-score) rank.
    drop_unknown : bool
        Remove items normalized to 'unknown' / empty before classification.
    view : 'browser' | 'notebook' | None
        Plot rendering target. None returns the table only.
    verbose : bool
        Print a short summary line.

    Returns
    -------
    pandas.DataFrame
        Columns: entity, productivity, impact_total, mean_impact,
                 [h_index if requested], quadrant.
    """
    alias = _resolve_entity_alias(entity)
    per_paper, u_list, counts, cits = _get_entity_data(pbx, alias)

    rows = []
    for i, name in enumerate(u_list):
        prod = float(counts[i]) if i < len(counts) else 0.0
        tot  = float(cits[i])   if i < len(cits)   else 0.0
        rows.append({
            'entity':       name,
            'productivity': prod,
            'impact_total': tot,
            'mean_impact':  _safe_div(tot, prod, 0.0),
        })

    df = pd.DataFrame(rows)
    if drop_unknown and not df.empty:
        df = df[~df['entity'].apply(_is_unknown)].reset_index(drop=True)

    if df.empty:
        if verbose:
            print(f'[portfolio_analysis] no data for entity={alias!r}')
        out = df.assign(quadrant=pd.Series(dtype=str))
        try:
            setattr(pbx, 'portfolio_analysis_table', out.copy())
        except Exception:
            pass
        return out

    # impact metric column selection
    impact_lc = str(impact).strip().lower()
    if impact_lc in ('citations', 'total_citations', 'sum', 'impact_total'):
        impact_col = 'impact_total'
    elif impact_lc in ('mean_citations', 'mean', 'avg', 'avg_citations', 'mean_impact'):
        impact_col = 'mean_impact'
    elif impact_lc in ('h_index', 'h-index', 'hindex', 'h'):
        h_vals = _per_entity_h_index(pbx, alias)
        df['h_index'] = df['entity'].map(h_vals).fillna(0).astype(int)
        impact_col = 'h_index'
    else:
        raise ValueError(f"Unknown impact metric: {impact!r}")

    df = df.sort_values(['productivity', impact_col], ascending=[False, False]).reset_index(drop=True)

    # topn truncation by z-score rank
    if topn is not None:
        try:
            n = int(topn)
            if n > 0 and n < len(df):
                p_std = df['productivity'].std(ddof=0)
                i_std = df[impact_col].std(ddof=0)
                pz = (df['productivity'] - df['productivity'].mean()) / (p_std + 1e-12)
                iz = (df[impact_col]    - df[impact_col].mean())    / (i_std    + 1e-12)
                df = (df.assign(__rank=pz + iz)
                        .sort_values('__rank', ascending=False)
                        .head(n)
                        .drop(columns='__rank')
                        .reset_index(drop=True))
        except Exception:
            pass

    # thresholds
    if isinstance(thresholds, (tuple, list)) and len(thresholds) == 2:
        try:
            p_cut, i_cut = float(thresholds[0]), float(thresholds[1])
        except Exception:
            p_cut = float(df['productivity'].median())
            i_cut = float(df[impact_col].median())
    elif str(thresholds).strip().lower() == 'mean':
        p_cut = float(df['productivity'].mean())
        i_cut = float(df[impact_col].mean())
    else:
        p_cut = float(df['productivity'].median())
        i_cut = float(df[impact_col].median())

    def _classify(row):
        hp = row['productivity'] >= p_cut
        hi = row[impact_col]    >= i_cut
        if hp and hi:
            return 'Stars'
        if (not hp) and hi:
            return 'Emerging'
        if hp and (not hi):
            return 'Mature/Core'
        return 'Marginal'

    df['quadrant'] = df.apply(_classify, axis=1)
    pretty = _ENTITY_PRETTY.get(alias, alias.title())

    if verbose:
        q_counts = df['quadrant'].value_counts().to_dict()
        print(f'[portfolio_analysis] entity={pretty} | n={len(df)} | '
              f'cuts: prod>={p_cut:.2f}, {impact_col}>={i_cut:.2f} | {q_counts}')

    try:
        setattr(pbx, 'portfolio_analysis_table', df.copy())
    except Exception:
        pass

    if view is not None and not df.empty:
        fig = _portfolio_figure(df, impact_col, p_cut, i_cut, pretty)
        _render_figure(fig, view)

    return df


def _per_entity_h_index(pbx, alias: str) -> Dict[str, int]:
    """Local h-index for each entity item, based on citations of papers it appears in."""
    per_paper_attr, u_attr, _, _ = _ENTITY_TABLE[alias]
    per_paper = getattr(pbx, per_paper_attr, []) or []
    u_list    = list(getattr(pbx, u_attr, []) or [])
    citations = getattr(pbx, 'citation', []) or []
    idx_map   = {it: k for k, it in enumerate(u_list)}
    bucket: Dict[int, List[float]] = defaultdict(list)
    for paper_idx, items in enumerate(per_paper):
        if paper_idx >= len(citations):
            continue
        try:
            c = float(citations[paper_idx] or 0.0)
        except Exception:
            c = 0.0
        for it in (items or []):
            k = idx_map.get(it)
            if k is not None:
                bucket[k].append(c)
    out: Dict[str, int] = {}
    for k, citlist in bucket.items():
        citlist_sorted = sorted(citlist, reverse=True)
        h = 0
        for rank, c in enumerate(citlist_sorted, start=1):
            if c >= rank:
                h = rank
            else:
                break
        out[u_list[k]] = h
    return out


def _portfolio_figure(df: pd.DataFrame, impact_col: str, p_cut: float, i_cut: float, pretty: str):
    go, _ = _lazy_plotly()

    color_map = {
        'Stars':       PALETTE['stars'],
        'Emerging':    PALETTE['emerging'],
        'Mature/Core': PALETTE['mature'],
        'Marginal':    PALETTE['marginal'],
    }
    wash_map = {
        'Stars':       PALETTE['wash_stars'],
        'Emerging':    PALETTE['wash_emerging'],
        'Mature/Core': PALETTE['wash_mature'],
        'Marginal':    PALETTE['wash_marginal'],
    }

    # Plot bounds (a touch of padding so markers aren't on the axis line)
    p_min, p_max = float(df['productivity'].min()), float(df['productivity'].max())
    i_min, i_max = float(df[impact_col].min()),    float(df[impact_col].max())
    p_pad = max((p_max - p_min) * 0.08, 1.0)
    i_pad = max((i_max - i_min) * 0.08, 1.0)
    x_lo, x_hi = p_min - p_pad, p_max + p_pad
    y_lo, y_hi = i_min - i_pad, i_max + i_pad

    fig = go.Figure()

    # Quadrant background washes (very subtle, layered below markers)
    for x0, y0, x1, y1, fill in [
        (p_cut, i_cut, x_hi, y_hi, wash_map['Stars']),
        (x_lo,  i_cut, p_cut, y_hi, wash_map['Emerging']),
        (p_cut, y_lo,  x_hi, i_cut, wash_map['Mature/Core']),
        (x_lo,  y_lo,  p_cut, i_cut, wash_map['Marginal']),
    ]:
        fig.add_shape(type='rect', x0=x0, y0=y0, x1=x1, y1=y1,
                      fillcolor=fill, line=dict(width=0), layer='below')

    # Large faint quadrant watermarks
    pad = 0.04
    for label, x, y, xa, ya, c in [
        ('STARS',       x_hi - pad * (x_hi - x_lo), y_hi - pad * (y_hi - y_lo), 'right', 'top',    color_map['Stars']),
        ('EMERGING',    x_lo + pad * (x_hi - x_lo), y_hi - pad * (y_hi - y_lo), 'left',  'top',    color_map['Emerging']),
        ('MATURE / CORE', x_hi - pad * (x_hi - x_lo), y_lo + pad * (y_hi - y_lo), 'right', 'bottom', color_map['Mature/Core']),
        ('MARGINAL',    x_lo + pad * (x_hi - x_lo), y_lo + pad * (y_hi - y_lo), 'left',  'bottom', color_map['Marginal']),
    ]:
        fig.add_annotation(x=x, y=y, text=label, showarrow=False,
                           xanchor=xa, yanchor=ya,
                           font=dict(family=FONT_FAMILY, size=22, color=c),
                           opacity=0.16)

    # Subtle dotted cut lines
    fig.add_shape(type='line', x0=p_cut, x1=p_cut, y0=y_lo, y1=y_hi,
                  line=dict(color=PALETTE['text_soft'], width=1, dash='dot'), layer='above')
    fig.add_shape(type='line', x0=x_lo, x1=x_hi, y0=i_cut, y1=i_cut,
                  line=dict(color=PALETTE['text_soft'], width=1, dash='dot'), layer='above')

    # Markers per quadrant
    show_text = len(df) <= 25
    for q in ['Stars', 'Emerging', 'Mature/Core', 'Marginal']:
        sub = df[df['quadrant'] == q]
        if sub.empty:
            continue
        sizes = np.log1p(sub['mean_impact'].clip(lower=0.0).to_numpy()) * 5.5 + 11.0
        c = color_map[q]
        hover = [
            f"<b>{_format_label(r['entity'], 80)}</b><br>"
            f"<span style='color:{PALETTE['text_faint']}'>──────────</span><br>"
            f"Productivity: <b>{r['productivity']:.0f}</b> docs<br>"
            f"{impact_col.replace('_', ' ').title()}: <b>{r[impact_col]:.2f}</b><br>"
            f"Citations / doc: <b>{r['mean_impact']:.2f}</b><br>"
            f"<span style='color:{c}'>● {q}</span>"
            for _, r in sub.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=sub['productivity'], y=sub[impact_col],
            mode='markers+text' if show_text else 'markers',
            marker=dict(size=sizes, color=c, opacity=0.88,
                        line=dict(width=1.5, color='white')),
            name=q,
            text=[_format_label(e, 22) for e in sub['entity']] if show_text else None,
            textposition='top center',
            textfont=dict(family=FONT_FAMILY, size=10, color=PALETTE['text_normal']),
            hovertext=hover, hoverinfo='text',
        ))

    metric_name = impact_col.replace('_', ' ').title()
    layout = _base_layout(
        title=f'Portfolio Analysis — {pretty}',
        subtitle=(f'Productivity × {metric_name} · {len(df)} entities · '
                  f'cuts: prod = {p_cut:.0f}, {metric_name.lower()} = {i_cut:.1f}'),
        height=680, width=1020,
    )
    xa = _clean_axis('Productivity (documents)')
    xa['range'] = [x_lo, x_hi]
    ya = _clean_axis(metric_name)
    ya['range'] = [y_lo, y_hi]
    layout.update(
        xaxis=xa, yaxis=ya,
        legend=dict(
            title=dict(text='<b>Quadrant</b>', font=dict(size=11)),
            orientation='h', y=1.06, x=1.0, xanchor='right',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor=PALETTE['text_faint'], borderwidth=1,
            font=dict(family=FONT_FAMILY, size=10),
        ),
    )
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# §2  Specialization Analysis (Activity Index / RCA / Symmetric Index / Share)
# ══════════════════════════════════════════════════════════════════════════════

def specialization_analysis(
    pbx,
    entity: str = 'cout',
    field: str = 'kwa',
    metric: str = 'activity_index',
    topn_entity: Optional[int] = 15,
    topn_field: Optional[int] = 15,
    min_count: int = 2,
    drop_unknown: bool = True,
    view: Optional[str] = 'browser',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build an entity × field co-occurrence matrix M and compute three indices:

      AI[i,j] = (M[i,j] / R_i) / (C_j / T)         -- Activity Index (== Balassa RCA)
      SI[i,j] = (AI - 1) / (AI + 1)                -- Symmetric Index in [-1, 1]
      share[i,j] = M[i,j] / R_i                    -- Entity's portfolio share in field j

    Where R_i = Σ_j M[i,j], C_j = Σ_i M[i,j], T = ΣΣ M.

    Parameters
    ----------
    entity : str    -- rows of the matrix ('cout', 'inst', 'aut', 'jou', ...).
    field  : str    -- columns ('kwa', 'kwp', 'ref', ...).
    metric : str    -- which index to render: 'activity_index'|'rca' (default),
                       'symmetric_index'|'si', or 'share'.
    topn_entity, topn_field : int or None
        Cap each axis to its top-N by document count.
    min_count : int
        Drop items below this raw count before building the matrix.

    Returns
    -------
    pandas.DataFrame in long form with columns
        entity, field, co_count, activity_index, symmetric_index, share.
    """
    e_alias = _resolve_entity_alias(entity)
    f_alias = _resolve_entity_alias(field)
    if e_alias == f_alias:
        raise ValueError(f"entity and field must differ; both resolved to {_ENTITY_PRETTY.get(e_alias)}")

    e_per, e_u, e_cnt, _ = _get_entity_data(pbx, e_alias)
    f_per, f_u, f_cnt, _ = _get_entity_data(pbx, f_alias)

    def _filter_top(u_list, counts, topn, mc):
        items = []
        for i, nm in enumerate(u_list):
            if drop_unknown and _is_unknown(nm):
                continue
            c = counts[i] if i < len(counts) else 0
            if c >= mc:
                items.append((nm, c))
        items.sort(key=lambda t: -t[1])
        if topn is not None:
            try:
                n = int(topn)
                if n > 0:
                    items = items[:n]
            except Exception:
                pass
        return [nm for nm, _ in items]

    e_labels = _filter_top(e_u, e_cnt, topn_entity, int(min_count))
    f_labels = _filter_top(f_u, f_cnt, topn_field,  int(min_count))

    if not e_labels or not f_labels:
        if verbose:
            print('[specialization_analysis] empty after filtering')
        return pd.DataFrame(columns=['entity', 'field', 'co_count',
                                     'activity_index', 'symmetric_index', 'share'])

    e_rank = {nm: r for r, nm in enumerate(e_labels)}
    f_rank = {nm: r for r, nm in enumerate(f_labels)}

    R, C = len(e_labels), len(f_labels)
    M = np.zeros((R, C), dtype=float)
    npapers = min(len(e_per), len(f_per))
    for p in range(npapers):
        e_items = [it for it in (e_per[p] or []) if it in e_rank]
        if not e_items:
            continue
        fitems = [it for it in (f_per[p] or []) if it in f_rank]
        if not fitems:
            continue
        # dedupe within a paper
        e_items = list(dict.fromkeys(e_items))
        fitems  = list(dict.fromkeys(fitems))
        for ei in e_items:
            r = e_rank[ei]
            for fi in fitems:
                c = f_rank[fi]
                M[r, c] += 1.0

    if M.sum() == 0:
        if verbose:
            print('[specialization_analysis] no co-occurrences found')
        return pd.DataFrame(columns=['entity', 'field', 'co_count',
                                     'activity_index', 'symmetric_index', 'share'])

    Ri = M.sum(axis=1, keepdims=True)
    Cj = M.sum(axis=0, keepdims=True)
    T  = float(M.sum())

    with np.errstate(divide='ignore', invalid='ignore'):
        AI = np.where((Ri > 0) & (Cj > 0), (M / np.where(Ri == 0, 1, Ri)) / (Cj / T), 0.0)
        SI = np.where(AI + 1 > 0, (AI - 1) / (AI + 1), 0.0)
        SH = np.where(Ri > 0, M / np.where(Ri == 0, 1, Ri), 0.0)
    AI = np.nan_to_num(AI, nan=0.0, posinf=0.0, neginf=0.0)
    SI = np.nan_to_num(SI, nan=0.0, posinf=0.0, neginf=0.0)
    SH = np.nan_to_num(SH, nan=0.0, posinf=0.0, neginf=0.0)

    rows = []
    for r in range(R):
        for c in range(C):
            if M[r, c] == 0:
                continue
            rows.append({
                'entity':          e_labels[r],
                'field':           f_labels[c],
                'co_count':        int(M[r, c]),
                'activity_index':  round(float(AI[r, c]), 4),
                'symmetric_index': round(float(SI[r, c]), 4),
                'share':           round(float(SH[r, c]), 4),
            })
    long_df = (pd.DataFrame(rows)
               .sort_values(['entity', 'activity_index'], ascending=[True, False])
               .reset_index(drop=True))

    e_pretty = _ENTITY_PRETTY.get(e_alias, e_alias)
    f_pretty = _ENTITY_PRETTY.get(f_alias, f_alias)

    if verbose:
        print(f'[specialization_analysis] {e_pretty} ({R}) × {f_pretty} ({C}) | '
              f'nonzero cells={len(long_df)} | total co-occurrences={int(T)}')

    try:
        setattr(pbx, 'specialization_analysis_table', long_df.copy())
    except Exception:
        pass

    if view is not None:
        metric_lc = str(metric).strip().lower()
        if metric_lc in ('rca', 'activity_index', 'ai'):
            Z, zname, cmid = AI, 'Activity Index (RCA)', 1.0
        elif metric_lc in ('symmetric_index', 'si'):
            Z, zname, cmid = SI, 'Symmetric Index', 0.0
        elif metric_lc in ('share', 'specialization', 'portfolio_share'):
            Z, zname, cmid = SH, 'Portfolio Share', None
        else:
            Z, zname, cmid = AI, 'Activity Index (RCA)', 1.0
        fig = _specialization_heatmap(Z, e_labels, f_labels, e_pretty, f_pretty, zname, cmid)
        _render_figure(fig, view)

    return long_df


def _specialization_heatmap(Z, e_labels, f_labels, e_pretty, f_pretty, zname, cmid):
    go, _ = _lazy_plotly()
    if cmid is not None:
        colorscale = _DIVERGING_SCALE
    else:
        colorscale = _SHARE_SCALE
    fig = go.Figure(data=go.Heatmap(
        z=Z,
        x=[_format_label(s, 30) for s in f_labels],
        y=[_format_label(s, 30) for s in e_labels],
        colorscale=colorscale,
        zmid=cmid,
        hoverongaps=False,
        xgap=2, ygap=2,
        colorbar=dict(
            title=dict(text=f'<b>{zname}</b>',
                       font=dict(family=FONT_FAMILY, size=11, color=PALETTE['text_strong'])),
            thickness=14, len=0.7, x=1.02, xanchor='left',
            tickfont=dict(family=FONT_FAMILY, size=10, color=PALETTE['text_soft']),
            outlinewidth=0,
        ),
        hovertemplate=(f"<b>%{{y}}</b><br>"
                       f"<span style='color:{PALETTE['text_faint']}'>──────────</span><br>"
                       f"{f_pretty}: %{{x}}<br>{zname}: <b>%{{z:.3f}}</b><extra></extra>"),
    ))
    layout = _base_layout(
        title=f'Specialization — {e_pretty} × {f_pretty}',
        subtitle=f'{zname} · matrix {len(e_labels)} × {len(f_labels)}',
        height=max(480, 30 * len(e_labels) + 220),
        width =max(760, 30 * len(f_labels) + 360),
    )
    xa = _clean_axis(f_pretty, show_grid=False)
    xa['tickangle'] = -40
    ya = _clean_axis(e_pretty, show_grid=False)
    layout.update(xaxis=xa, yaxis=ya, margin=dict(l=160, r=120, t=100, b=120))
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# §3  Collaboration Impact (solo vs collab + centralities)
# ══════════════════════════════════════════════════════════════════════════════

def collaboration_impact(
    pbx,
    entity: str = 'cout',
    topn: Optional[int] = 25,
    min_documents: int = 1,
    drop_unknown: bool = True,
    view: Optional[str] = 'browser',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Collaboration-quality indicators per entity (countries / institutions / authors).

    For each entity item, the function computes:
      - documents     : total appearances
      - solo          : papers with a single distinct entity item
      - collab        : papers with >=2 distinct entity items of the same type
      - collab_rate   : collab / documents
      - mean_cit_solo, mean_cit_collab
      - impact_ratio_collab_over_solo
      - degree_centrality, betweenness_centrality (on the entity co-occurrence graph)
    """
    alias = _resolve_entity_alias(entity)
    if alias not in ('cout', 'inst', 'aut'):
        raise ValueError(f"collaboration_impact supports entity in {{cout, inst, aut}}; got {alias!r}")

    per_paper, u_list, counts, _ = _get_entity_data(pbx, alias)
    citations = getattr(pbx, 'citation', []) or []
    n_papers  = len(per_paper)

    # per-paper unique items + collab type
    paper_unique: List[List[str]] = []
    paper_type:   List[str]       = []
    for items in per_paper:
        seen, dedup = set(), []
        for it in (items or []):
            if drop_unknown and _is_unknown(it):
                continue
            key = str(it).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(it)
        paper_unique.append(dedup)
        if len(dedup) >= 2:
            paper_type.append('collab')
        elif len(dedup) == 1:
            paper_type.append('solo')
        else:
            paper_type.append('unknown')

    # build adjacency (co-occurrence weights)
    adj: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for items in paper_unique:
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a == b:
                    continue
                adj[a][b] += 1
                adj[b][a] += 1

    # centralities (networkx if available; degree-from-adjacency fallback otherwise)
    centrality_deg: Dict[str, float] = {}
    centrality_btw: Dict[str, float] = {}
    nodes_valid = [n for n in u_list if not _is_unknown(n)]
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(nodes_valid)
        seen_edge: set = set()
        for a, nbrs in adj.items():
            for b, w in nbrs.items():
                key = (a, b) if a < b else (b, a)
                if key in seen_edge:
                    continue
                seen_edge.add(key)
                G.add_edge(a, b, weight=int(w))
        if len(G) > 0:
            try:
                centrality_deg = {k: float(v) for k, v in nx.degree_centrality(G).items()}
            except Exception:
                pass
            try:
                if G.number_of_edges() > 0:
                    centrality_btw = {k: float(v)
                                      for k, v in nx.betweenness_centrality(G, normalized=True).items()}
            except Exception:
                pass
    except ImportError:
        # fallback: normalized degree from adjacency
        n_nodes = max(len(nodes_valid), 1)
        denom = max(n_nodes - 1, 1)
        for a, nbrs in adj.items():
            centrality_deg[a] = len(nbrs) / denom

    # per-entity tallies
    rows = []
    for name in u_list:
        if drop_unknown and _is_unknown(name):
            continue
        n_total = 0
        n_solo  = 0
        n_collab = 0
        cit_solo = 0.0
        cit_collab = 0.0
        for p in range(n_papers):
            if name not in paper_unique[p]:
                continue
            n_total += 1
            try:
                c = float(citations[p]) if p < len(citations) and citations[p] is not None else 0.0
            except Exception:
                c = 0.0
            t = paper_type[p]
            if t == 'solo':
                n_solo   += 1
                cit_solo += c
            elif t == 'collab':
                n_collab   += 1
                cit_collab += c

        if n_total < int(min_documents):
            continue

        mean_solo   = _safe_div(cit_solo,   n_solo,   0.0)
        mean_collab = _safe_div(cit_collab, n_collab, 0.0)
        impact_ratio = _safe_div(mean_collab, mean_solo, np.nan) if n_solo > 0 else np.nan

        rows.append({
            'entity':                          name,
            'documents':                       n_total,
            'solo':                            n_solo,
            'collab':                          n_collab,
            'collab_rate':                     round(_safe_div(n_collab, n_total, 0.0), 4),
            'mean_cit_solo':                   round(mean_solo, 3),
            'mean_cit_collab':                 round(mean_collab, 3),
            'impact_ratio_collab_over_solo':   (round(float(impact_ratio), 3)
                                                if pd.notna(impact_ratio) else np.nan),
            'degree_centrality':               round(centrality_deg.get(name, 0.0), 4),
            'betweenness_centrality':          round(centrality_btw.get(name, 0.0), 4),
        })

    df = pd.DataFrame(rows).sort_values('documents', ascending=False).reset_index(drop=True)
    if topn is not None:
        try:
            n = int(topn)
            if n > 0:
                df = df.head(n).reset_index(drop=True)
        except Exception:
            pass

    pretty = _ENTITY_PRETTY.get(alias, alias)
    if verbose and not df.empty:
        mean_rate = float(df['collab_rate'].mean())
        n_with_btw = int((df['betweenness_centrality'] > 0).sum())
        print(f'[collaboration_impact] entity={pretty} | n={len(df)} | '
              f'mean collab_rate={mean_rate:.2f} | nodes with betweenness>0: {n_with_btw}')

    try:
        setattr(pbx, 'collaboration_impact_table', df.copy())
    except Exception:
        pass

    if view is not None and not df.empty:
        fig = _collaboration_figure(df, pretty)
        _render_figure(fig, view)

    return df


def _collaboration_figure(df: pd.DataFrame, pretty: str):
    go, _ = _lazy_plotly()
    sizes = np.sqrt(df['documents'].astype(float).clip(lower=1).to_numpy()) * 3.8 + 10.0
    impact_safe = df['impact_ratio_collab_over_solo'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    x = df['collab_rate'].astype(float).to_numpy()
    y = impact_safe.astype(float).to_numpy()

    fig = go.Figure(data=go.Scatter(
        x=x, y=y,
        mode='markers+text' if len(df) <= 25 else 'markers',
        marker=dict(
            size=sizes,
            color=df['betweenness_centrality'],
            colorscale=[[0.0, '#f1f5f9'], [0.5, '#6366f1'], [1.0, '#1e1b4b']],
            cmin=0.0,
            showscale=True,
            colorbar=dict(
                title=dict(text='<b>Betweenness</b>',
                           font=dict(family=FONT_FAMILY, size=11,
                                     color=PALETTE['text_strong'])),
                thickness=14, len=0.7, x=1.02, xanchor='left',
                tickfont=dict(family=FONT_FAMILY, size=10, color=PALETTE['text_soft']),
                outlinewidth=0,
            ),
            line=dict(width=1.5, color='white'),
            opacity=0.9,
        ),
        text=[_format_label(e, 22) for e in df['entity']] if len(df) <= 25 else None,
        textposition='top center',
        textfont=dict(family=FONT_FAMILY, size=10, color=PALETTE['text_normal']),
        hovertext=[
            f"<b>{r['entity']}</b><br>"
            f"<span style='color:{PALETTE['text_faint']}'>──────────</span><br>"
            f"Documents: <b>{int(r['documents'])}</b> "
            f"({int(r['solo'])} solo · {int(r['collab'])} collab)<br>"
            f"Collab rate: <b>{r['collab_rate']:.2f}</b><br>"
            f"Mean citations · solo: <b>{r['mean_cit_solo']:.2f}</b><br>"
            f"Mean citations · collab: <b>{r['mean_cit_collab']:.2f}</b><br>"
            f"Impact ratio: <b>{r['impact_ratio_collab_over_solo']}</b><br>"
            f"Degree: {r['degree_centrality']:.3f} · "
            f"Betweenness: {r['betweenness_centrality']:.3f}"
            for _, r in df.iterrows()
        ],
        hoverinfo='text',
    ))

    # Reference line at impact parity (collab citations == solo citations)
    fig.add_shape(type='line', x0=-0.02, x1=1.02, y0=1.0, y1=1.0,
                  line=dict(color=PALETTE['text_soft'], width=1, dash='dot'),
                  layer='below')
    fig.add_annotation(x=1.0, y=1.0, text='<i>impact parity</i>',
                       showarrow=False, xanchor='right', yanchor='bottom',
                       font=dict(family=FONT_FAMILY, size=9, color=PALETTE['text_soft']))

    n_boost = int((y > 1.0).sum())
    layout = _base_layout(
        title=f'Collaboration Quality — {pretty}',
        subtitle=(f'{len(df)} entities · {n_boost} with collab → impact boost · '
                  f'marker size = documents · color = betweenness'),
        height=660, width=1020,
    )
    xa = _clean_axis('Collaboration Rate (share of multi-entity papers)')
    xa['range'] = [-0.02, 1.02]
    xa['tickformat'] = '.0%'
    ya = _clean_axis('Impact Ratio (mean citations · collab / solo)')
    layout.update(xaxis=xa, yaxis=ya, margin=dict(l=80, r=120, t=100, b=70))
    fig.update_layout(**layout)
    return fig



# ══════════════════════════════════════════════════════════════════════════════
# §4  Collaboration Brokerage (structural bridge roles in collaboration graphs)
# ══════════════════════════════════════════════════════════════════════════════

def _collaboration_brokerage_figure(df: pd.DataFrame, pretty: str):
    go, _ = _lazy_plotly()
    plot_df = df.copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df['Weighted Degree'],
        y=plot_df['Betweenness Centrality'],
        mode='markers+text',
        text=[_format_label(v, 24) for v in plot_df['Entity']],
        textposition='top center',
        textfont=dict(family=FONT_FAMILY, size=9, color=PALETTE['text_soft']),
        marker=dict(
            size=np.clip(10 + 3.0 * np.sqrt(pd.to_numeric(plot_df['Documents'], errors='coerce').fillna(0)), 10, 42),
            color=plot_df['Brokerage Score'],
            colorscale=_TIME_SCALE,
            showscale=True,
            colorbar=dict(
                title=dict(text='<b>Brokerage</b>', font=dict(family=FONT_FAMILY, size=11, color=PALETTE['text_strong'])),
                thickness=14,
                len=0.74,
                outlinewidth=0,
                tickfont=dict(family=FONT_FAMILY, size=10, color=PALETTE['text_soft']),
            ),
            line=dict(color='white', width=1.0),
            opacity=0.90,
        ),
        customdata=np.stack([
            plot_df['Documents'],
            plot_df['Citations'],
            plot_df['Degree'],
            plot_df['Constraint'],
            plot_df['Community'],
            plot_df['Mean Citations'],
        ], axis=-1),
        hovertemplate=(
            '<b>%{text}</b><br>'
            '<span style="color:#cbd5e1">────────────────</span><br>'
            'Weighted degree: <b>%{x:.2f}</b><br>'
            'Betweenness: <b>%{y:.4f}</b><br>'
            'Documents: %{customdata[0]} · Citations: %{customdata[1]}<br>'
            'Mean citations: %{customdata[5]:.2f}<br>'
            'Degree: %{customdata[2]} · Constraint: %{customdata[3]:.3f}<br>'
            'Community: %{customdata[4]}<extra></extra>'
        ),
        name='Entities',
    ))

    layout = _base_layout(
        title=f'Collaboration Brokerage — {pretty}',
        subtitle='Brokerage highlights entities connecting otherwise separated collaboration neighborhoods',
        height=660,
        width=1060,
    )
    layout.update(
        xaxis=_clean_axis('Weighted collaboration degree'),
        yaxis=_clean_axis('Betweenness centrality'),
        margin=dict(l=80, r=120, t=100, b=80),
    )
    fig.update_layout(**layout)
    return fig


def collaboration_brokerage(
    pbx,
    entity: str = 'aut',
    topn: Optional[int] = 40,
    min_documents: int = 1,
    drop_unknown: bool = True,
    view: Optional[str] = 'browser',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Quantify collaboration brokerage for authors, institutions, or countries.

    The function builds a co-occurrence collaboration graph from the selected
    entity type and reports centrality, structural-hole constraint, detected
    community, and a normalized brokerage score.

    Parameters
    ----------
    pbx : pybibx pbx instance
        Active PyBibX object.
    entity : {'aut', 'inst', 'cout'} or aliases, default='aut'
        Collaboration level: authors, institutions, or countries.
    topn : int or None, default=40
        Number of top-ranked entities to return. Use None for all entities.
    min_documents : int, default=1
        Minimum number of documents required for an entity to be included.
    drop_unknown : bool, default=True
        Remove unknown or empty entity labels.
    view : {'browser', 'notebook', None}, default='browser'
        Render a Plotly figure when not None.
    verbose : bool, default=True
        Print a compact execution summary.

    Returns
    -------
    pandas.DataFrame
        Brokerage table with productivity, citation, centrality, constraint,
        community, and composite brokerage indicators.
    """
    alias = _resolve_entity_alias(entity, default='aut')
    if alias not in {'aut', 'inst', 'cout'}:
        raise ValueError("collaboration_brokerage entity must be one of: 'aut', 'inst', 'cout' or their aliases")

    per_paper, _, _, _ = _get_entity_data(pbx, alias)
    citations = list(getattr(pbx, 'citation', []) or [])

    docs: Dict[str, int] = defaultdict(int)
    cits: Dict[str, float] = defaultdict(float)
    edge_weights: Dict[Tuple[str, str], float] = defaultdict(float)

    for i, items in enumerate(per_paper):
        clean_items = _clean_entity_items(items, drop_unknown=drop_unknown)
        if len(clean_items) == 0:
            continue
        try:
            paper_cit = float(citations[i]) if i < len(citations) and citations[i] is not None else 0.0
        except Exception:
            paper_cit = 0.0
        for item in clean_items:
            docs[item] += 1
            cits[item] += paper_cit
        if len(clean_items) >= 2:
            for pos, a in enumerate(clean_items):
                for b in clean_items[pos + 1:]:
                    if a != b:
                        edge_weights[_pair_key(a, b)] += 1.0

    keep = {k for k, v in docs.items() if v >= int(min_documents)}
    empty_cols = [
        'Entity', 'Documents', 'Citations', 'Mean Citations', 'Degree',
        'Weighted Degree', 'Degree Centrality', 'Betweenness Centrality',
        'Closeness Centrality', 'Eigenvector Centrality', 'Constraint',
        'Brokerage Score', 'Community'
    ]

    if len(keep) == 0:
        df = pd.DataFrame(columns=empty_cols)
        pbx.collaboration_brokerage_table = df
        pbx.ask_gpt_collaboration_brokerage = df
        if verbose:
            print('[collaboration_brokerage] no entities passed the filters')
        return df

    degree: Dict[str, int] = {n: 0 for n in keep}
    weighted_degree: Dict[str, float] = {n: 0.0 for n in keep}
    degree_cent: Dict[str, float] = {n: 0.0 for n in keep}
    bet_cent: Dict[str, float] = {n: 0.0 for n in keep}
    close_cent: Dict[str, float] = {n: 0.0 for n in keep}
    eig_cent: Dict[str, float] = {n: 0.0 for n in keep}
    constraint: Dict[str, float] = {n: np.nan for n in keep}
    community: Dict[str, int] = {n: 0 for n in keep}

    try:
        import networkx as nx
        G = nx.Graph()
        for node in keep:
            G.add_node(node)
        for (a, b), w in edge_weights.items():
            if a in keep and b in keep:
                G.add_edge(a, b, weight=float(w))

        degree = dict(G.degree())
        weighted_degree = dict(G.degree(weight='weight'))
        if G.number_of_nodes() > 1:
            degree_cent = {k: float(v) for k, v in nx.degree_centrality(G).items()}
        if G.number_of_edges() > 0:
            # Unweighted betweenness is intentionally used here: larger edge
            # weight means stronger collaboration, not longer graph distance.
            bet_cent = {k: float(v) for k, v in nx.betweenness_centrality(G, normalized=True).items()}
            close_cent = {k: float(v) for k, v in nx.closeness_centrality(G).items()}
            try:
                eig_cent = {k: float(v) for k, v in nx.eigenvector_centrality_numpy(G, weight='weight').items()}
            except Exception:
                eig_cent = {n: 0.0 for n in G.nodes()}
            try:
                constraint = {k: float(v) for k, v in nx.constraint(G, weight='weight').items()}
            except Exception:
                constraint = {n: np.nan for n in G.nodes()}
            try:
                communities = list(nx.algorithms.community.greedy_modularity_communities(G, weight='weight'))
                for cid, nodes in enumerate(communities, start=1):
                    for n in nodes:
                        community[n] = cid
            except Exception:
                pass
    except ImportError:
        # Dependency-safe fallback: degree and weighted degree are still useful
        # brokerage proxies when NetworkX is not available.
        adjacency: Dict[str, Dict[str, float]] = defaultdict(dict)
        for (a, b), w in edge_weights.items():
            if a in keep and b in keep:
                adjacency[a][b] = adjacency[a].get(b, 0.0) + float(w)
                adjacency[b][a] = adjacency[b].get(a, 0.0) + float(w)
        denom = max(len(keep) - 1, 1)
        for n in keep:
            degree[n] = len(adjacency.get(n, {}))
            weighted_degree[n] = float(sum(adjacency.get(n, {}).values()))
            degree_cent[n] = float(degree[n] / denom)

    raw = pd.DataFrame({
        'Entity': list(keep),
        'Documents': [int(docs[n]) for n in keep],
        'Citations': [int(cits[n]) for n in keep],
        'Mean Citations': [_safe_div(float(cits[n]), float(docs[n]), 0.0) for n in keep],
        'Degree': [int(degree.get(n, 0)) for n in keep],
        'Weighted Degree': [float(weighted_degree.get(n, 0.0)) for n in keep],
        'Degree Centrality': [float(degree_cent.get(n, 0.0)) for n in keep],
        'Betweenness Centrality': [float(bet_cent.get(n, 0.0)) for n in keep],
        'Closeness Centrality': [float(close_cent.get(n, 0.0)) for n in keep],
        'Eigenvector Centrality': [float(eig_cent.get(n, 0.0)) for n in keep],
        'Constraint': [float(constraint.get(n, np.nan)) if pd.notna(constraint.get(n, np.nan)) else np.nan for n in keep],
        'Community': [int(community.get(n, 0)) for n in keep],
    })

    if raw['Constraint'].notna().any():
        inv_constraint = 1.0 - _minmax(raw['Constraint'].fillna(raw['Constraint'].max()))
    else:
        inv_constraint = pd.Series([0.5] * len(raw), index=raw.index, dtype=float)

    raw['Brokerage Score'] = (
        0.45 * _minmax(raw['Betweenness Centrality']) +
        0.25 * _minmax(raw['Weighted Degree']) +
        0.20 * inv_constraint +
        0.10 * _minmax(raw['Documents'])
    ).astype(float)

    df = raw.sort_values(
        ['Brokerage Score', 'Betweenness Centrality', 'Weighted Degree', 'Documents'],
        ascending=False,
    ).reset_index(drop=True)

    if topn is not None:
        try:
            df = df.head(int(topn)).reset_index(drop=True)
        except Exception:
            pass

    pbx.collaboration_brokerage_table = df
    pbx.ask_gpt_collaboration_brokerage = df

    if verbose:
        print(f'[collaboration_brokerage] entity={alias} | returned={len(df)} | min_documents={min_documents}')

    if view is not None and not df.empty:
        fig = _collaboration_brokerage_figure(df, _ENTITY_PRETTY.get(alias, alias))
        pbx.collaboration_brokerage_figure = fig
        _render_figure(fig, view)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# §4  Burst Detection (Kleinberg 2-state discrete)
# ══════════════════════════════════════════════════════════════════════════════

def burst_detection(
    pbx,
    source: str = 'kwa',
    method: str = 'kleinberg',
    min_frequency: int = 5,
    s: float = 2.0,
    gamma: float = 1.0,
    topn: Optional[int] = 30,
    drop_unknown: bool = True,
    start: Optional[int] = None,
    end: Optional[int] = None,
    view: Optional[str] = 'browser',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Kleinberg 2-state discrete burst detection over year-binned per-term
    document frequencies.

    For each term t the model takes:
      r_t = number of papers in year t that contain the term
      d_t = total number of papers in year t
    A two-state Markov chain (baseline rate p0 vs burst rate p1 = s * p0)
    is fit by minimum-cost Viterbi decoding. Bursts are returned as
    consecutive runs of the burst state with associated strength.

    Parameters
    ----------
    source : str
        Term source ('kwa', 'kwp', 'aut', 'cout', 'inst', 'jou', 'ref', 'lan').
    method : str
        'kleinberg' (the only method currently implemented).
    min_frequency : int
        Drop terms whose total count is below this threshold.
    s : float
        Burst-state rate multiplier (p1 = min(s * p0, 0.999)). Must be > 1.
    gamma : float
        Cost of transitioning into a burst state. Larger values produce fewer
        but longer / stronger bursts.
    topn : int or None
        Cap on the number of returned bursts (sorted by strength descending).
    start, end : int or None
        Restrict the temporal window (defaults to the corpus span).

    Returns
    -------
    pandas.DataFrame
        Columns: term, start_year, end_year, length, burst_strength, total_count.
    """
    method_lc = str(method).strip().lower()
    if method_lc not in ('kleinberg', 'kleinberg_discrete'):
        raise ValueError(f"Only method='kleinberg' is implemented; got {method!r}")
    if not (s > 1.0):
        raise ValueError(f"s must be > 1.0 (burst rate multiplier); got {s}")

    alias = _resolve_entity_alias(source)
    per_paper, u_list, counts, _ = _get_entity_data(pbx, alias)
    years = _resolve_years(pbx)
    y_lo, y_hi = _year_bounds(pbx, start, end)
    if y_hi < y_lo:
        return pd.DataFrame(columns=['term', 'start_year', 'end_year', 'length',
                                     'burst_strength', 'total_count'])

    year_range = list(range(int(y_lo), int(y_hi) + 1))
    nyears     = len(year_range)
    y_idx: Dict[int, int] = {y: k for k, y in enumerate(year_range)}

    # papers per year
    d_per_year = np.zeros(nyears, dtype=float)
    for y in years:
        yy = int(y)
        if yy in y_idx:
            d_per_year[y_idx[yy]] += 1.0

    # filter terms by min_frequency
    candidates: List[Tuple[int, str, int]] = []
    for i, name in enumerate(u_list):
        if drop_unknown and _is_unknown(name):
            continue
        cnt = int(counts[i]) if i < len(counts) else 0
        if cnt >= int(min_frequency):
            candidates.append((i, name, cnt))
    if not candidates:
        if verbose:
            print(f'[burst_detection] no terms meet min_frequency={min_frequency}')
        try:
            setattr(pbx, 'burst_detection_table',
                    pd.DataFrame(columns=['term', 'start_year', 'end_year',
                                          'length', 'burst_strength', 'total_count']))
        except Exception:
            pass
        return pd.DataFrame(columns=['term', 'start_year', 'end_year', 'length',
                                     'burst_strength', 'total_count'])

    cand_rank: Dict[str, int] = {nm: rank for rank, (_, nm, _) in enumerate(candidates)}
    n_terms = len(candidates)
    R = np.zeros((n_terms, nyears), dtype=float)

    npapers = min(len(per_paper), len(years))
    for p in range(npapers):
        yy = int(years[p])
        if yy not in y_idx:
            continue
        y_pos = y_idx[yy]
        seen_t: set = set()
        for it in (per_paper[p] or []):
            rank = cand_rank.get(it)
            if rank is None or rank in seen_t:
                continue
            seen_t.add(rank)
            R[rank, y_pos] += 1.0

    rows = []
    for rank, (i, name, total) in enumerate(candidates):
        bursts = _kleinberg_2state(R[rank], d_per_year, s=float(s), gamma=float(gamma))
        for (a, b, strength) in bursts:
            rows.append({
                'term':           name,
                'start_year':     int(year_range[a]),
                'end_year':       int(year_range[b]),
                'length':         int(b - a + 1),
                'burst_strength': round(float(strength), 4),
                'total_count':    int(total),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        if verbose:
            print('[burst_detection] no bursts found (try lowering s or gamma)')
        try:
            setattr(pbx, 'burst_detection_table', df.copy())
        except Exception:
            pass
        return df

    df = df.sort_values(['burst_strength', 'start_year'], ascending=[False, True]).reset_index(drop=True)
    if topn is not None:
        try:
            n = int(topn)
            if n > 0:
                df = df.head(n).reset_index(drop=True)
        except Exception:
            pass

    if verbose:
        print(f'[burst_detection] terms_with_burst={df["term"].nunique()} | '
              f'total bursts={len(df)} | window={y_lo}-{y_hi}')

    try:
        setattr(pbx, 'burst_detection_table', df.copy())
    except Exception:
        pass

    if view is not None:
        fig = _burst_figure(df, year_range, _ENTITY_PRETTY.get(alias, alias))
        _render_figure(fig, view)

    return df


def _kleinberg_2state(r: np.ndarray, d: np.ndarray,
                      s: float = 2.0, gamma: float = 1.0) -> List[Tuple[int, int, float]]:
    """
    Discrete two-state Kleinberg burst detection on year-binned counts.

    Parameters
    ----------
    r : array of per-period event counts (papers containing the term in year t).
    d : array of per-period trials      (total papers in year t).
    s : burst-state rate multiplier (must be > 1).
    gamma : transition cost into the burst state.

    Returns
    -------
    list of (start_idx, end_idx, strength) tuples. Strength is the sum over the
    burst window of (cost_baseline - cost_burst) under the binomial model.
    """
    r = np.asarray(r, dtype=float)
    d = np.asarray(d, dtype=float)
    n = len(r)
    if n == 0:
        return []
    sum_d = float(d.sum())
    sum_r = float(r.sum())
    if sum_d <= 0 or sum_r <= 0:
        return []
    p0 = sum_r / sum_d
    p1 = min(0.999, float(s) * p0)
    if p1 <= p0 or p0 <= 0 or p0 >= 1:
        return []

    eps = 1e-12
    p0c = min(max(p0, eps), 1 - eps)
    p1c = min(max(p1, eps), 1 - eps)

    # negative log-likelihood (binomial; combinatorial term is state-independent)
    c0 = -r * math.log(p0c) - (d - r) * math.log(1 - p0c)
    c1 = -r * math.log(p1c) - (d - r) * math.log(1 - p1c)
    trans = float(gamma)

    INF = float('inf')
    f0 = np.full(n, INF)
    f1 = np.full(n, INF)
    bp0 = np.zeros(n, dtype=int)
    bp1 = np.zeros(n, dtype=int)
    f0[0] = c0[0]
    f1[0] = c1[0] + trans

    for t in range(1, n):
        # to state 0 (no transition cost going down)
        a = f0[t - 1] + c0[t]
        b = f1[t - 1] + c0[t]
        if a <= b:
            f0[t] = a; bp0[t] = 0
        else:
            f0[t] = b; bp0[t] = 1
        # to state 1 (transition cost going up)
        a = f0[t - 1] + trans + c1[t]
        b = f1[t - 1] + c1[t]
        if a <= b:
            f1[t] = a; bp1[t] = 0
        else:
            f1[t] = b; bp1[t] = 1

    # backtrace
    states = [0] * n
    states[n - 1] = 0 if f0[n - 1] <= f1[n - 1] else 1
    for t in range(n - 1, 0, -1):
        prev = bp0[t] if states[t] == 0 else bp1[t]
        states[t - 1] = int(prev)

    bursts: List[Tuple[int, int, float]] = []
    i = 0
    while i < n:
        if states[i] == 1:
            j = i
            while j < n and states[j] == 1:
                j += 1
            strength = float(np.sum(c0[i:j] - c1[i:j]))
            bursts.append((i, j - 1, strength))
            i = j
        else:
            i += 1
    return bursts


def _burst_figure(df: pd.DataFrame, year_range: List[int], pretty: str):
    go, _ = _lazy_plotly()
    y_min, y_max = int(year_range[0]), int(year_range[-1])
    df2 = (df.sort_values(['start_year', 'burst_strength'], ascending=[True, False])
             .reset_index(drop=True))
    n = len(df2)
    terms = df2['term'].tolist()

    smin = float(df2['burst_strength'].min())
    smax = float(df2['burst_strength'].max())
    rng  = (smax - smin) if smax > smin else 1.0

    fig = go.Figure()

    # Alternating row stripes for readability
    for i in range(n):
        if i % 2 == 1:
            fig.add_shape(type='rect',
                          x0=y_min - 1, x1=y_max + 3,
                          y0=i - 0.5,   y1=i + 0.5,
                          fillcolor='#f8fafc',
                          line=dict(width=0), layer='below')

    # Subtle vertical year guides
    for y in range(y_min, y_max + 1):
        fig.add_shape(type='line',
                      x0=y, x1=y, y0=-0.5, y1=n - 0.5,
                      line=dict(color=PALETTE['grid'], width=0.5),
                      layer='below')

    # Bursts: thick line + circular end caps; color = strength gradient
    for idx, r in df2.iterrows():
        norm  = (float(r['burst_strength']) - smin) / rng
        color = _interp_rgb(PALETTE['burst_low'], PALETTE['burst_high'], norm)
        hover_text = (
            f"<b>{r['term']}</b><br>"
            f"<span style='color:{PALETTE['text_faint']}'>──────────</span><br>"
            f"{int(r['start_year'])} – {int(r['end_year'])} · "
            f"length <b>{int(r['length'])}y</b><br>"
            f"Strength: <b>{r['burst_strength']:.2f}</b><br>"
            f"Total occurrences: {int(r['total_count'])}"
        )
        fig.add_trace(go.Scatter(
            x=[r['start_year'], r['end_year']],
            y=[idx, idx],
            mode='lines',
            line=dict(color=color, width=13),
            showlegend=False,
            hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=[r['start_year'], r['end_year']],
            y=[idx, idx],
            mode='markers',
            marker=dict(size=15, color=color, line=dict(width=2.5, color='white')),
            hovertext=[hover_text, hover_text],
            hoverinfo='text',
            showlegend=False,
        ))
        # Strength label trailing the bar
        fig.add_annotation(
            x=r['end_year'] + 0.4, y=idx,
            text=f"<b>{r['burst_strength']:.1f}</b>",
            showarrow=False, xanchor='left', yanchor='middle',
            font=dict(family=FONT_FAMILY, size=9, color=PALETTE['text_soft']),
        )

    tick_step = max(1, (y_max - y_min) // 12)
    layout = _base_layout(
        title=f'Burst Detection — {pretty}',
        subtitle=f'Kleinberg 2-state model · {n} bursts · {y_min}–{y_max}',
        height=max(440, 34 * n + 200),
        width=1100,
    )
    xa = _clean_axis('Year', show_grid=False)
    xa['range'] = [y_min - 0.8, y_max + 2.5]
    xa['dtick'] = tick_step
    layout.update(
        xaxis=xa,
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(n)),
            ticktext=[_format_label(t, 50) for t in terms],
            autorange='reversed',
            showgrid=False, zeroline=False, showline=False,
            ticks='', ticklen=0,
            tickfont=dict(family=FONT_FAMILY, size=10, color=PALETTE['text_strong']),
        ),
        margin=dict(l=260, r=100, t=100, b=70),
    )
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# §5  Knowledge Diffusion (temporal | citation)
# ══════════════════════════════════════════════════════════════════════════════

def knowledge_diffusion(
    pbx,
    source_entity: str = 'cout',
    target_entity: str = 'cout',
    concept_field: str = 'kwa',
    mechanism: str = 'temporal',
    topn_concepts: Optional[int] = 12,
    topn_entities: Optional[int] = 15,
    min_concept_count: int = 5,
    drop_unknown: bool = True,
    viz: str = 'auto',
    view: Optional[str] = 'browser',
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Track how concepts spread across entities over time.

    Mechanisms
    ----------
    'temporal'
        For each (entity, concept), record the earliest year of use.
        The earliest year across entities defines the concept's origin;
        adoption_delay_years measures the lag for each later adopter.
        Visualization: heatmap of first-adoption years.

    'citation'
        For each pair (paper A, paper B) such that B cites A, year(A) < year(B),
        and both papers share at least one concept C, register a directional
        diffusion edge: source_entity(A)  →  target_entity(B), weighted by the
        number of such citation-with-shared-concept events.
        Visualizations: chord (default) or sankey.

    Parameters
    ----------
    source_entity, target_entity : str
        Origin and destination entity aliases. For 'temporal' they should match
        (the function treats them as the same axis); for 'citation' they may
        differ (e.g. source='aut', target='cout' to study author-to-country flow).
    concept_field : str
        Concept axis alias ('kwa', 'kwp', 'ref', ...).
    topn_concepts : int or None
        Restrict to the top-N concepts by total count.
    topn_entities : int or None
        Restrict to the top-N entities by total count.
    min_concept_count : int
        Drop concepts below this raw count before any matrix building.
    viz : 'auto' | 'heatmap' | 'chord' | 'sankey'
        Visualization style. 'auto' picks 'heatmap' for the temporal mechanism
        and 'chord' for the citation mechanism. 'sankey' is the original Sankey
        diagram (citation only). 'heatmap' is the first-adoption matrix
        (temporal only).

    Returns
    -------
    pandas.DataFrame
        For 'temporal': columns concept, entity, first_year, origin_year,
                        adoption_delay_years, origin_entities.
        For 'citation': columns concept, source, target, weight, mean_delay_years.
    """
    mech = str(mechanism).strip().lower()
    if mech not in ('temporal', 'citation'):
        raise ValueError(f"mechanism must be 'temporal' or 'citation'; got {mechanism!r}")

    viz_lc = str(viz).strip().lower()
    if viz_lc not in ('auto', 'heatmap', 'chord', 'sankey'):
        raise ValueError(f"viz must be 'auto' | 'heatmap' | 'chord' | 'sankey'; got {viz!r}")

    src_alias = _resolve_entity_alias(source_entity)
    tgt_alias = _resolve_entity_alias(target_entity)
    cfield    = _resolve_entity_alias(concept_field)
    if cfield == src_alias or cfield == tgt_alias:
        raise ValueError("concept_field must differ from source_entity and target_entity")

    src_per, src_u, src_cnt, _ = _get_entity_data(pbx, src_alias)
    if tgt_alias == src_alias:
        tgt_per, tgt_u, tgt_cnt = src_per, src_u, src_cnt
    else:
        tgt_per, tgt_u, tgt_cnt, _ = _get_entity_data(pbx, tgt_alias)
    c_per, c_u, c_cnt, _ = _get_entity_data(pbx, cfield)
    years = _resolve_years(pbx)

    # concept filter
    concept_items: List[Tuple[str, int]] = []
    for i, name in enumerate(c_u):
        if drop_unknown and _is_unknown(name):
            continue
        cnt = int(c_cnt[i]) if i < len(c_cnt) else 0
        if cnt >= int(min_concept_count):
            concept_items.append((name, cnt))
    concept_items.sort(key=lambda t: -t[1])
    if topn_concepts is not None:
        try:
            n = int(topn_concepts)
            if n > 0:
                concept_items = concept_items[:n]
        except Exception:
            pass
    if not concept_items:
        if verbose:
            print('[knowledge_diffusion] no concepts pass filter')
        empty = (pd.DataFrame(columns=['concept', 'entity', 'first_year', 'origin_year',
                                       'adoption_delay_years', 'origin_entities'])
                 if mech == 'temporal'
                 else pd.DataFrame(columns=['concept', 'source', 'target', 'weight', 'mean_delay_years']))
        try:
            setattr(pbx, 'knowledge_diffusion_table', empty.copy())
        except Exception:
            pass
        return empty
    concept_set = {c for c, _ in concept_items}

    def _pick_top(u_list, cnts, topn):
        items = []
        for i, nm in enumerate(u_list):
            if drop_unknown and _is_unknown(nm):
                continue
            items.append((nm, cnts[i] if i < len(cnts) else 0))
        items.sort(key=lambda t: -t[1])
        if topn is not None:
            try:
                k = int(topn)
                if k > 0:
                    items = items[:k]
            except Exception:
                pass
        return [nm for nm, _ in items]

    src_keep = _pick_top(src_u, src_cnt, topn_entities)
    tgt_keep = _pick_top(tgt_u, tgt_cnt, topn_entities) if tgt_alias != src_alias else src_keep
    src_keep_set = set(src_keep)
    tgt_keep_set = set(tgt_keep)

    # ───────────────────── temporal ─────────────────────
    if mech == 'temporal':
        # for temporal we focus on a single axis; if user passed differing src/tgt,
        # we still proceed but use src_entity for the matrix.
        first_year: Dict[Tuple[str, str], int] = {}
        n_papers = min(len(src_per), len(c_per), len(years))
        for p in range(n_papers):
            y = int(years[p])
            if y < 0:
                continue
            ents = [e for e in (src_per[p] or []) if e in src_keep_set]
            if not ents:
                continue
            cons = [c for c in (c_per[p] or []) if c in concept_set]
            if not cons:
                continue
            for e in ents:
                for c in cons:
                    key = (e, c)
                    prev = first_year.get(key)
                    if prev is None or y < prev:
                        first_year[key] = y

        long_rows = []
        for e in src_keep:
            for c, _ in concept_items:
                y = first_year.get((e, c))
                long_rows.append({
                    'entity':     e,
                    'concept':    c,
                    'first_year': int(y) if y is not None else np.nan,
                })
        long_df = pd.DataFrame(long_rows)

        delay_rows = []
        for c, _ in concept_items:
            sub = long_df[(long_df['concept'] == c) & long_df['first_year'].notna()].copy()
            if sub.empty:
                continue
            origin_year = int(sub['first_year'].min())
            origins = sub[sub['first_year'] == origin_year]['entity'].tolist()
            for _, r in sub.iterrows():
                delay_rows.append({
                    'concept':              c,
                    'entity':               r['entity'],
                    'first_year':           int(r['first_year']),
                    'origin_year':          origin_year,
                    'adoption_delay_years': int(r['first_year']) - origin_year,
                    'origin_entities':      '; '.join(origins),
                })
        delay_df = (pd.DataFrame(delay_rows)
                    .sort_values(['concept', 'first_year'])
                    .reset_index(drop=True))

        if verbose:
            print(f'[knowledge_diffusion mechanism=temporal] concepts={len(concept_items)} '
                  f'entities={len(src_keep)} | rows={len(delay_df)}')

        try:
            setattr(pbx, 'knowledge_diffusion_table', delay_df.copy())
        except Exception:
            pass

        if view is not None and not delay_df.empty:
            if viz_lc == 'sankey' or viz_lc == 'chord':
                if verbose:
                    print(f"[knowledge_diffusion] viz={viz_lc!r} ignored for "
                          f"mechanism='temporal'; using 'heatmap' instead")
            fig = _diffusion_temporal_figure(
                long_df, src_keep, [c for c, _ in concept_items],
                _ENTITY_PRETTY.get(src_alias, src_alias),
                _ENTITY_PRETTY.get(cfield, cfield),
            )
            _render_figure(fig, view)

        return delay_df

    # ───────────────────── citation ─────────────────────
    if not hasattr(pbx, '_internal_reference_doc_ids'):
        raise RuntimeError(
            "citation mechanism requires pbx._internal_reference_doc_ids(); "
            "this pbx instance does not provide it."
        )
    internal_refs: List[List[int]] = pbx._internal_reference_doc_ids()
    n_papers = min(len(internal_refs), len(years), len(src_per), len(tgt_per), len(c_per))

    # edges[concept][(src_entity, tgt_entity)] = {'weight': int, 'delay': int}
    edges: Dict[str, Dict[Tuple[str, str], Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {'weight': 0, 'delay': 0}))

    for citing_p in range(n_papers):
        y_b = int(years[citing_p])
        if y_b < 0:
            continue
        b_concepts = set(c for c in (c_per[citing_p] or []) if c in concept_set)
        if not b_concepts:
            continue
        b_tgt_entities = [e for e in (tgt_per[citing_p] or []) if e in tgt_keep_set]
        if not b_tgt_entities:
            continue
        for cited_p in internal_refs[citing_p]:
            if cited_p is None:
                continue
            try:
                cp = int(cited_p)
            except Exception:
                continue
            if cp < 0 or cp >= n_papers or cp == citing_p:
                continue
            y_a = int(years[cp])
            if y_a < 0 or y_a >= y_b:
                continue
            a_concepts = set(c for c in (c_per[cp] or []) if c in concept_set)
            shared = a_concepts.intersection(b_concepts)
            if not shared:
                continue
            a_src_entities = [e for e in (src_per[cp] or []) if e in src_keep_set]
            if not a_src_entities:
                continue
            delta = y_b - y_a
            for con in shared:
                for a_e in a_src_entities:
                    for b_e in b_tgt_entities:
                        if a_e == b_e and src_alias == tgt_alias:
                            continue  # skip same-node self-loops
                        cell = edges[con][(a_e, b_e)]
                        cell['weight'] += 1
                        cell['delay']  += delta

    rows = []
    for con, ed in edges.items():
        for (a, b), info in ed.items():
            w = int(info.get('weight', 0))
            d = int(info.get('delay', 0))
            if w <= 0:
                continue
            rows.append({
                'concept':          con,
                'source':           a,
                'target':           b,
                'weight':           w,
                'mean_delay_years': round(d / w, 2),
            })
    df = (pd.DataFrame(rows)
          .sort_values(['concept', 'weight'], ascending=[True, False])
          .reset_index(drop=True))

    if verbose:
        n_con = df['concept'].nunique() if not df.empty else 0
        print(f'[knowledge_diffusion mechanism=citation] edges={len(df)} | concepts={n_con}')

    try:
        setattr(pbx, 'knowledge_diffusion_table', df.copy())
    except Exception:
        pass

    if view is not None and not df.empty:
        chosen = viz_lc
        if chosen == 'auto':
            chosen = 'chord'
        if chosen == 'heatmap':
            if verbose:
                print(f"[knowledge_diffusion] viz='heatmap' invalid for "
                      f"mechanism='citation'; using 'chord' instead")
            chosen = 'chord'
        figure_fn = _diffusion_chord_figure if chosen == 'chord' else _diffusion_sankey_figure
        fig = figure_fn(
            df,
            _ENTITY_PRETTY.get(src_alias, src_alias),
            _ENTITY_PRETTY.get(tgt_alias, tgt_alias),
            _ENTITY_PRETTY.get(cfield,    cfield),
        )
        _render_figure(fig, view)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Reference diversity and disruption analytics
# ──────────────────────────────────────────────────────────────────────────────

def reference_diversity(pbx, paper_ids = None):
    """
    Compute reference-diversity indicators for selected papers.

    Parameters
    ----------
    pbx : pybibx pbx instance
        Active PyBibX object containing the bibliographic dataframe and parsed
        reference structures.
    paper_ids : list[int] | tuple[int] | set[int] | int | None, optional
        Paper/document IDs to analyze. Use None to analyze all papers.

    Returns
    -------
    pandas.DataFrame
        Table with reference counts, internal-reference share, reference-age
        statistics, source entropy, and self-citation rate.
    """
    selected = pbx._safe_doc_indices(paper_ids) if (paper_ids is not None) else list(range(pbx.data.shape[0]))
    years = pd.to_numeric(pbx.data['year'], errors = 'coerce').fillna(np.nan).tolist()
    internal_refs = pbx._internal_reference_doc_ids()
    source_col = 'abbrev_source_title' if ('abbrev_source_title' in pbx.data.columns) else 'journal'
    rows = []

    for idx in selected:
        paper_year = years[idx]
        refs = [ref for ref in pbx.ref[idx] if str(ref).strip().upper() != 'UNKNOWN']
        internal_ids = [ref_idx for ref_idx in internal_refs[idx] if 0 <= ref_idx < pbx.data.shape[0]]
        ref_ages = []
        internal_sources = []
        self_cites = 0
        focal_authors = set([str(a).strip().lower() for a in pbx.aut[idx] if str(a).strip().upper() != 'UNKNOWN'])

        for ref_idx in internal_ids:
            ref_year = pd.to_numeric(pbx.data.iloc[ref_idx].get('year', np.nan), errors = 'coerce')
            if (pd.notna(paper_year) and pd.notna(ref_year)):
                ref_ages.append(float(paper_year) - float(ref_year))
            src_name = str(pbx.data.iloc[ref_idx].get(source_col, 'UNKNOWN')).strip().lower()
            if (src_name and src_name != 'unknown'):
                internal_sources.append(src_name)
            ref_authors = set([str(a).strip().lower() for a in pbx.aut[ref_idx] if str(a).strip().upper() != 'UNKNOWN'])
            if (len(focal_authors.intersection(ref_authors)) > 0):
                self_cites += 1

        source_entropy = np.nan
        source_entropy_norm = np.nan
        unique_sources = len(set(internal_sources))
        if (len(internal_sources) > 0 and unique_sources > 0):
            counts = pd.Series(internal_sources).value_counts(normalize = True).values
            source_entropy = float(-(counts * np.log(counts)).sum())
            if (unique_sources > 1):
                source_entropy_norm = float(source_entropy / np.log(unique_sources))
            else:
                source_entropy_norm = 0.0

        row = {
            'Paper ID': str(idx),
            'Year': int(paper_year) if pd.notna(paper_year) else np.nan,
            'N References': int(len(refs)),
            'Unique References': int(len(set(refs))),
            'N Internal References': int(len(internal_ids)),
            'Internal Reference Share': float(len(internal_ids) / len(refs)) if len(refs) > 0 else np.nan,
            'Mean Reference Age': float(np.mean(ref_ages)) if len(ref_ages) > 0 else np.nan,
            'Median Reference Age': float(np.median(ref_ages)) if len(ref_ages) > 0 else np.nan,
            'Std Reference Age': float(np.std(ref_ages)) if len(ref_ages) > 0 else np.nan,
            'Min Reference Age': float(np.min(ref_ages)) if len(ref_ages) > 0 else np.nan,
            'Max Reference Age': float(np.max(ref_ages)) if len(ref_ages) > 0 else np.nan,
            'Unique Internal Sources': int(unique_sources),
            'Source Entropy': source_entropy,
            'Normalized Source Entropy': source_entropy_norm,
            'Self Citation Rate': float(self_cites / len(internal_ids)) if len(internal_ids) > 0 else np.nan,
        }
        rows.append(row)

    results = pd.DataFrame(rows)
    pbx.reference_diversity_table = results
    return results


def disruption_index(pbx, paper_ids = None, strict_future = True, min_future_citers = 1):
    """
    Compute the disruption index for selected papers.

    Parameters
    ----------
    pbx : pybibx pbx instance
        Active PyBibX object containing the bibliographic dataframe and parsed
        internal citation network.
    paper_ids : list[int] | tuple[int] | set[int] | int | None, optional
        Paper/document IDs to analyze. Use None to analyze all papers.
    strict_future : bool, default=True
        If True, only citing papers published after the focal paper are counted.
        If False, same-year citation relations are allowed where applicable.
    min_future_citers : int, default=1
        Minimum number of future citers required for a valid disruption score.

    Returns
    -------
    pandas.DataFrame
        Table with disruption components N_i, N_j, N_k and the final disruption
        index for each selected paper.
    """
    selected = pbx._safe_doc_indices(paper_ids) if (paper_ids is not None) else list(range(pbx.data.shape[0]))
    years = pd.to_numeric(pbx.data['year'], errors = 'coerce').fillna(-1).astype(int).tolist()
    internal_refs = pbx._internal_reference_doc_ids()

    citers_of = defaultdict(set)
    for citing_idx, refs in enumerate(internal_refs):
        citing_year = years[citing_idx]
        for cited_idx in refs:
            if (citing_idx == cited_idx or cited_idx < 0 or cited_idx >= len(years)):
                continue
            cited_year = years[cited_idx]
            if (strict_future == True and citing_year <= cited_year):
                continue
            if (strict_future == False and citing_year < cited_year):
                continue
            citers_of[cited_idx].add(citing_idx)

    rows = []
    for focal_idx in selected:
        focal_year = years[focal_idx]
        focal_refs = set()
        for ref_idx in internal_refs[focal_idx]:
            if (0 <= ref_idx < len(years)):
                if (strict_future == True and years[ref_idx] < focal_year):
                    focal_refs.add(ref_idx)
                elif (strict_future == False and years[ref_idx] <= focal_year):
                    focal_refs.add(ref_idx)

        focal_citers = set([j for j in citers_of.get(focal_idx, set()) if years[j] > focal_year])
        ref_citers = set()
        for ref_idx in focal_refs:
            ref_citers.update([j for j in citers_of.get(ref_idx, set()) if years[j] > focal_year and j != focal_idx])

        n_i = 0
        n_j = 0
        for citer_idx in focal_citers:
            citer_refs = set(internal_refs[citer_idx])
            if (len(citer_refs.intersection(focal_refs)) > 0):
                n_j += 1
            else:
                n_i += 1
        n_k = len(ref_citers.difference(focal_citers))
        denom = n_i + n_j + n_k
        score = float((n_i - n_j) / denom) if denom > 0 else np.nan
        valid = bool(len(focal_citers) >= int(min_future_citers) and denom > 0)
        if (valid == False and denom > 0 and len(focal_citers) < int(min_future_citers)):
            score = np.nan

        rows.append({
            'Paper ID': str(focal_idx),
            'Year': int(focal_year) if focal_year >= 0 else np.nan,
            'N Internal References': int(len(focal_refs)),
            'N Future Citers': int(len(focal_citers)),
            'N_i': int(n_i),
            'N_j': int(n_j),
            'N_k': int(n_k),
            'Disruption Index': score,
            'Valid': valid,
        })

    results = pd.DataFrame(rows)
    pbx.disruption_index_table = results
    return results


def _diffusion_temporal_figure(long_df: pd.DataFrame, e_labels: List[str],
                               c_labels: List[str], e_pretty: str, c_pretty: str):
    go, _ = _lazy_plotly()
    pivot = long_df.pivot_table(index='entity', columns='concept',
                                values='first_year', aggfunc='min')
    pivot = pivot.reindex(index=e_labels, columns=c_labels)
    Z = pivot.values.astype(float)
    fig = go.Figure(data=go.Heatmap(
        z=Z,
        x=[_format_label(c, 30) for c in c_labels],
        y=[_format_label(e, 30) for e in e_labels],
        colorscale=_TIME_SCALE,
        xgap=2, ygap=2,
        colorbar=dict(
            title=dict(text='<b>First adoption year</b>',
                       font=dict(family=FONT_FAMILY, size=11, color=PALETTE['text_strong'])),
            thickness=14, len=0.7, x=1.02, xanchor='left',
            tickfont=dict(family=FONT_FAMILY, size=10, color=PALETTE['text_soft']),
            outlinewidth=0,
        ),
        hovertemplate=(f"<b>%{{y}}</b><br>"
                       f"<span style='color:{PALETTE['text_faint']}'>──────────</span><br>"
                       f"{c_pretty}: %{{x}}<br>First year: <b>%{{z}}</b><extra></extra>"),
        hoverongaps=False,
    ))
    layout = _base_layout(
        title=f'Knowledge Diffusion (Temporal) — {c_pretty}',
        subtitle=f'First adoption of {c_pretty.lower()} by {e_pretty.lower()}',
        height=max(480, 30 * len(e_labels) + 220),
        width =max(760, 30 * len(c_labels) + 360),
    )
    xa = _clean_axis(c_pretty, show_grid=False)
    xa['tickangle'] = -40
    ya = _clean_axis(e_pretty, show_grid=False)
    layout.update(xaxis=xa, yaxis=ya, margin=dict(l=160, r=120, t=100, b=120))
    fig.update_layout(**layout)
    return fig


def _diffusion_sankey_figure(df: pd.DataFrame, src_pretty: str, tgt_pretty: str, c_pretty: str):
    go, _ = _lazy_plotly()
    src_nodes = sorted(df['source'].astype(str).unique())
    tgt_nodes = sorted(df['target'].astype(str).unique())
    src_idx = {n: i for i, n in enumerate(src_nodes)}
    tgt_idx = {n: len(src_nodes) + i for i, n in enumerate(tgt_nodes)}

    labels = ([f'{n}' for n in src_nodes] +
              [f'{n}' for n in tgt_nodes])
    node_colors = ([PALETTE['stars']]    * len(src_nodes) +
                   [PALETTE['emerging']] * len(tgt_nodes))

    concepts = sorted(df['concept'].astype(str).unique())
    palette  = PALETTE['concept_cycle']
    cc_color = {c: palette[i % len(palette)] for i, c in enumerate(concepts)}

    def _hex_to_rgba(h, alpha=0.45):
        h = h.lstrip('#')
        return f'rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})'

    sources, targets, values, colors, link_labels = [], [], [], [], []
    for _, r in df.iterrows():
        sources.append(src_idx[str(r['source'])])
        targets.append(tgt_idx[str(r['target'])])
        values.append(int(r['weight']))
        colors.append(_hex_to_rgba(cc_color[str(r['concept'])], 0.42))
        link_labels.append(f"{r['concept']}  ·  mean delay {r['mean_delay_years']}y")

    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            label=labels, pad=18, thickness=18,
            color=node_colors,
            line=dict(color='white', width=1),
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=colors,
            label=link_labels,
            hovertemplate=('<b>%{label}</b><br>'
                           '%{source.label} → %{target.label}<br>'
                           'weight: <b>%{value}</b><extra></extra>'),
        ),
    )])
    layout = _base_layout(
        title=f'Knowledge Diffusion (Sankey) — {c_pretty}',
        subtitle=f'Citation flow · {src_pretty} → {tgt_pretty} · {len(df)} edges',
        height=max(580, 22 * (len(src_nodes) + len(tgt_nodes)) + 220),
        width=1120,
    )
    fig.update_layout(**layout)
    return fig


def _diffusion_chord_figure(df: pd.DataFrame, src_pretty: str, tgt_pretty: str, c_pretty: str):
    """
    Circular flow diagram — alternative to Sankey for citation diffusion.

    When src and tgt are the same entity type, all entities share one ring
    and curves connect them through the origin. When the types differ, sources
    occupy the right hemisphere and targets the left.
    """
    go, _ = _lazy_plotly()

    src_set = set(df['source'].astype(str))
    tgt_set = set(df['target'].astype(str))
    if min(len(src_set), len(tgt_set)) > 0:
        overlap = len(src_set & tgt_set)
        same_type = overlap >= 0.5 * min(len(src_set), len(tgt_set))
    else:
        same_type = False

    if same_type:
        nodes = sorted(src_set | tgt_set)
        n = len(nodes)
        angles = [np.pi / 2 - 2 * np.pi * i / max(n, 1) for i in range(n)]
        node_role = {nm: 'shared' for nm in nodes}
    else:
        src_nodes = sorted(src_set)
        tgt_nodes = sorted(tgt_set)
        nodes = src_nodes + tgt_nodes
        if len(src_nodes) > 0:
            a_s = np.linspace(np.pi/2 - 0.05*np.pi, -np.pi/2 + 0.05*np.pi, len(src_nodes))
        else:
            a_s = np.array([])
        if len(tgt_nodes) > 0:
            a_t = np.linspace(np.pi/2 + 0.05*np.pi, 3*np.pi/2 - 0.05*np.pi, len(tgt_nodes))
        else:
            a_t = np.array([])
        angles = list(a_s) + list(a_t)
        node_role = ({nm: 'src' for nm in src_nodes} |
                     {nm: 'tgt' for nm in tgt_nodes})

    pos = {nm: (np.cos(a), np.sin(a)) for nm, a in zip(nodes, angles)}
    angle_of = dict(zip(nodes, angles))

    concepts = sorted(df['concept'].astype(str).unique())
    palette  = PALETTE['concept_cycle']
    concept_color = {c: palette[i % len(palette)] for i, c in enumerate(concepts)}

    fig = go.Figure()

    # Background ring (dotted guide)
    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode='lines',
        line=dict(color=PALETTE['text_faint'], width=1, dash='dot'),
        hoverinfo='skip', showlegend=False,
    ))

    # Edges: quadratic Bézier with control at origin — pulls curves through
    # the center, giving the classic chord appearance
    max_w = float(df['weight'].max()) if not df.empty else 1.0
    df_sorted = df.sort_values('weight').reset_index(drop=True)

    # Render edges grouped by concept (one Plotly trace per concept → cleaner legend)
    for concept in concepts:
        sub = df_sorted[df_sorted['concept'].astype(str) == concept]
        if sub.empty:
            continue
        xs: List[Optional[float]] = []
        ys: List[Optional[float]] = []
        hovers: List[Optional[str]] = []
        edge_widths: List[float] = []
        for _, r in sub.iterrows():
            src, tgt = str(r['source']), str(r['target'])
            if src not in pos or tgt not in pos:
                continue
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            t = np.linspace(0, 1, 30)
            bx = (1 - t)**2 * x0 + 2 * (1 - t) * t * 0.0 + t**2 * x1
            by = (1 - t)**2 * y0 + 2 * (1 - t) * t * 0.0 + t**2 * y1
            xs.extend(bx.tolist() + [None])
            ys.extend(by.tolist() + [None])
            ht = (f"<b>{concept}</b><br>"
                  f"<span style='color:{PALETTE['text_faint']}'>──────────</span><br>"
                  f"{src} → {tgt}<br>"
                  f"weight: <b>{int(r['weight'])}</b> · "
                  f"avg delay: {r['mean_delay_years']:.1f}y")
            hovers.extend([ht] * len(t) + [None])
            edge_widths.append(1.4 + 6.0 * (float(r['weight']) / max_w))

        line_w = float(np.median(edge_widths)) if edge_widths else 2.0
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(width=line_w, color=concept_color[concept], shape='spline'),
            opacity=0.62,
            name=concept,
            hovertext=hovers, hoverinfo='text',
            legendgroup=concept,
        ))

    # Node markers (colored by role when src/tgt differ)
    node_x = [pos[nm][0] for nm in nodes]
    node_y = [pos[nm][1] for nm in nodes]
    node_colors = [
        PALETTE['stars']    if node_role[nm] == 'src' else
        PALETTE['emerging'] if node_role[nm] == 'tgt' else
        PALETTE['text_strong']
        for nm in nodes
    ]
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers',
        marker=dict(size=14, color=node_colors,
                    line=dict(width=2, color='white')),
        hoverinfo='text', hovertext=nodes,
        showlegend=False,
    ))

    # External labels with smart anchoring (left/right hemisphere flips)
    annotations = []
    for nm in nodes:
        a = angle_of[nm]
        r_label = 1.09
        annotations.append(dict(
            x=r_label * np.cos(a), y=r_label * np.sin(a),
            text=_format_label(nm, 22),
            showarrow=False,
            xanchor='left' if np.cos(a) >= 0 else 'right',
            yanchor='middle',
            font=dict(family=FONT_FAMILY, size=10, color=PALETTE['text_strong']),
        ))

    if same_type:
        subtitle = (f'Citation flow ({src_pretty} ↔ {tgt_pretty}) · '
                    f'{len(df)} edges · {len(concepts)} concepts')
    else:
        subtitle = (f'Citation flow ({src_pretty} → {tgt_pretty}) · '
                    f'{len(df)} edges · {len(concepts)} concepts')

    layout = _base_layout(
        title=f'Knowledge Diffusion (Chord) — {c_pretty}',
        subtitle=subtitle,
        height=780, width=1020,
    )
    layout.update(
        annotations=annotations,
        xaxis=dict(visible=False, range=[-1.55, 1.55],
                   scaleanchor='y', scaleratio=1),
        yaxis=dict(visible=False, range=[-1.45, 1.45]),
        legend=dict(
            title=dict(text='<b>Concept</b>',
                       font=dict(family=FONT_FAMILY, size=11,
                                 color=PALETTE['text_strong'])),
            orientation='v', y=0.5, x=1.02, xanchor='left', yanchor='middle',
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor=PALETTE['text_faint'], borderwidth=1,
            font=dict(family=FONT_FAMILY, size=10),
            itemsizing='constant',
        ),
        showlegend=True,
        margin=dict(l=60, r=200, t=100, b=60),
    )
    fig.update_layout(**layout)
    return fig
