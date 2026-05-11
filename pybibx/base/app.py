"""
PyBibX Web App — Interactive browser-based interface.

    @staticmethod
    def web_app():
        from pybibx.base.webapp import launch
        launch()

Or run standalone:
    python app.py
"""

import os, sys, json, base64, threading, webbrowser, tempfile, traceback, re
from io import BytesIO, StringIO
from contextlib import redirect_stdout

# ── Flask / Werkzeug ──────────────────────────────────────────────────────────
try:
    from flask import Flask, request, jsonify, render_template_string
    from werkzeug.serving import make_server
except ImportError:
    os.system(f"{sys.executable} -m pip install flask werkzeug -q")
    from flask import Flask, request, jsonify, render_template_string
    from werkzeug.serving import make_server

import plotly.graph_objects as go
import plotly.io as pio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
STATE = {'pbx': None, 'db': None, 'filepath': None, 'topic_model': None, 'sleeping_beauties': None, 'original_data': None}
_FIGS_PLOTLY = []
_FIGS_MPL    = []
_SERVER = None
_SERVER_THREAD = None
_SERVER_URL = None

# ── Intercept plotly show ─────────────────────────────────────────────────────
_orig_fig_show = go.Figure.show
def _capture_plotly(self, *a, **kw):
    _FIGS_PLOTLY.append(self.to_json())
go.Figure.show = _capture_plotly

# ── Intercept matplotlib show ─────────────────────────────────────────────────
_orig_plt_show = plt.show
def _capture_mpl(*a, **kw):
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for f in figs:
        buf = BytesIO()
        f.savefig(buf, format='png', bbox_inches='tight', dpi=130)
        buf.seek(0)
        _FIGS_MPL.append(base64.b64encode(buf.read()).decode())
    plt.close('all')
plt.show = _capture_mpl

# ── Run helper ────────────────────────────────────────────────────────────────


def _runtime_artifact_title(attr_name):
    name = attr_name
    for prefix in ('ask_gpt_', 'ask_'):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    tokens = [tok for tok in name.replace('__', '_').split('_') if tok]
    pieces = []
    for tok in tokens:
        pieces.append(tok.upper() if len(tok) <= 3 else tok.capitalize())
    return ' '.join(pieces) or 'Artifact'


def _is_runtime_artifact_value(value):
    import pandas as pd
    if value is None:
        return False
    if isinstance(value, pd.DataFrame):
        return not value.empty
    if isinstance(value, str):
        return value.strip() != ''
    if isinstance(value, dict):
        return len(value) > 0
    if isinstance(value, (list, tuple, set)):
        return len(value) > 0
    if isinstance(value, (int, float)):
        return value != -1
    return True


def _runtime_artifact_signature(value):
    import pandas as pd
    if isinstance(value, pd.DataFrame):
        preview = value.astype(str).head(50)
        return ('dataframe', tuple(value.shape), tuple(map(str, value.columns)), preview.to_json(orient='split', default_handler=str))
    if isinstance(value, dict):
        return ('dict', json.dumps(value, sort_keys=True, default=str))
    if isinstance(value, (list, tuple, set)):
        return ('sequence', json.dumps(list(value), default=str))
    return ('scalar', repr(value))


def _snapshot_runtime_artifacts(owner):
    if owner is None:
        return {}
    snap = {}
    for attr in dir(owner):
        if not (attr.startswith('ask_gpt_') or attr.startswith('ask_')):
            continue
        if attr.endswith('_t'):
            continue
        try:
            value = getattr(owner, attr)
        except Exception:
            continue
        if not _is_runtime_artifact_value(value):
            continue
        try:
            snap[attr] = _runtime_artifact_signature(value)
        except Exception:
            snap[attr] = ('unavailable', attr)
    return snap


def _runtime_artifact_payload(attr_name, value):
    import pandas as pd
    title = _runtime_artifact_title(attr_name)
    if isinstance(value, pd.DataFrame):
        return {'title': title, 'type': 'table', 'html': _df_html(value, index=False)}
    if isinstance(value, dict):
        if value and all(not isinstance(v, (dict, list, tuple, set)) for v in value.values()):
            df = pd.DataFrame({'Item': list(value.keys()), 'Value': list(value.values())})
            return {'title': title, 'type': 'table', 'html': _df_html(df, index=False)}
        return {'title': title, 'type': 'text', 'value': json.dumps(value, indent=2, default=str)}
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        if seq and all(isinstance(item, dict) for item in seq):
            try:
                df = pd.DataFrame(seq)
                return {'title': title, 'type': 'table', 'html': _df_html(df, index=False)}
            except Exception:
                pass
        if seq and all(isinstance(item, (list, tuple)) and not isinstance(item, str) for item in seq):
            try:
                df = pd.DataFrame(seq)
                return {'title': title, 'type': 'table', 'html': _df_html(df, index=False)}
            except Exception:
                pass
        return {'title': title, 'type': 'text', 'value': json.dumps(seq, indent=2, default=str)}
    return {'title': title, 'type': 'text', 'value': str(value)}


def _collect_runtime_artifacts(owner, before=None):
    if owner is None:
        return []
    before = before or {}
    artifacts = []
    for attr in dir(owner):
        if not (attr.startswith('ask_gpt_') or attr.startswith('ask_')):
            continue
        if attr.endswith('_t'):
            continue
        try:
            value = getattr(owner, attr)
        except Exception:
            continue
        if not _is_runtime_artifact_value(value):
            continue
        try:
            signature = _runtime_artifact_signature(value)
        except Exception:
            signature = ('unavailable', attr)
        if before.get(attr) == signature:
            continue
        try:
            artifacts.append(_runtime_artifact_payload(attr, value))
        except Exception:
            artifacts.append({'title': _runtime_artifact_title(attr), 'type': 'text', 'value': str(value)})
    return artifacts

def run_fn(func, *args, **kwargs):
    global _FIGS_PLOTLY, _FIGS_MPL
    _FIGS_PLOTLY, _FIGS_MPL = [], []
    plt.close('all')
    out = StringIO()
    result_data = None
    owner = getattr(func, '__self__', None)
    before_artifacts = _snapshot_runtime_artifacts(owner)
    try:
        with redirect_stdout(out):
            ret = func(*args, **kwargs)
    except Exception as e:
        return {'ok': False, 'error': str(e), 'trace': traceback.format_exc()}

    if plt.get_fignums():
        _capture_mpl()

    try:
        import pandas as pd
        if isinstance(ret, pd.DataFrame):
            result_data = {'type': 'dataframe', 'html': _df_html(
                ret, index=True, limit=50,
                float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else x
            )}
        elif isinstance(ret, (str, int, float)):
            result_data = {'type': 'text', 'value': str(ret)}
        elif isinstance(ret, dict):
            result_data = {'type': 'dict', 'value': json.dumps(ret, default=str, indent=2)}
        elif isinstance(ret, list) and len(ret) > 0:
            result_data = {'type': 'text', 'value': str(ret)}
    except Exception:
        pass

    artifacts = _collect_runtime_artifacts(owner, before_artifacts)

    return {
        'ok': True,
        'stdout': out.getvalue(),
        'plotly': _FIGS_PLOTLY[:],
        'images': _FIGS_MPL[:],
        'result': result_data,
        'artifacts': artifacts
    }


def _df_to_csv_text(df, index=False):
    buf = StringIO()
    df.to_csv(buf, index=index)
    return buf.getvalue()


def _df_html(df, index=False, limit=None, float_format=None):
    import pandas as pd
    if df is None:
        return '<div class="callout warn">No data available.</div>'
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return f'<pre class="text-out">{str(df)}</pre>'

    truncated = limit is not None and len(df) > limit
    shown = df.head(limit) if truncated else df
    note = ''
    if truncated:
        note = (
            f'<div class="callout info" style="margin-bottom:10px;">'
            f'Showing first {limit:,} of {len(df):,} rows. '
            f'Use <strong>Show all rows</strong> to expand the full table.'
            f'</div>'
        )

    table_html = shown.to_html(classes='result-table', border=0, index=index, float_format=float_format)
    payload = {
        'csv_b64': base64.b64encode(_df_to_csv_text(df, index=index).encode('utf-8')).decode('ascii'),
        'total_rows': int(len(df)),
        'shown_rows': int(len(shown)),
        'index': bool(index),
        'truncated': bool(truncated),
    }
    if truncated:
        full_html = _df_html(df, index=index, limit=None, float_format=float_format)
        payload['full_html_b64'] = base64.b64encode(full_html.encode('utf-8')).decode('ascii')

    extra_class = ' is-truncated' if truncated else ''
    return (
        f'<div class="df-artifact{extra_class}" '
        f'data-truncated="{1 if truncated else 0}" '
        f'data-total-rows="{len(df)}" '
        f'data-shown-rows="{len(shown)}">'
        f'<script type="application/json" class="df-payload">{json.dumps(payload)}</script>'
        f'{note}{table_html}'
        f'</div>'
    )


def _df_grid_payload(df, limit=None, page=1):
    import math
    import pandas as pd
    if df is None:
        df = pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            df = pd.DataFrame()
    total_rows = int(df.shape[0])
    total_cols = int(df.shape[1])
    try:
        limit = max(1, int(limit if limit is not None else 20))
    except Exception:
        limit = 20
    total_pages = max(1, math.ceil(total_rows / limit)) if total_rows > 0 else 1
    try:
        page = int(page)
    except Exception:
        page = 1
    page = min(max(1, page), total_pages)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    shown = df.iloc[start_idx:end_idx].copy()
    shown = shown.fillna('').astype(str)
    return {
        'columns': list(map(str, df.columns.tolist())),
        'rows': shown.values.tolist(),
        'visible_rows': int(shown.shape[0]),
        'total_rows': total_rows,
        'total_cols': total_cols,
        'page': page,
        'total_pages': total_pages,
        'page_size': limit,
        'start_row_index': start_idx,
        'end_row_index': max(start_idx, start_idx + int(shown.shape[0]) - 1) if int(shown.shape[0]) else start_idx,
    }




def _df_grid_full_payload(df):
    import pandas as pd
    if df is None:
        df = pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            df = pd.DataFrame()
    full = df.fillna('').astype(str)
    return {
        'columns': list(map(str, full.columns.tolist())),
        'rows': full.values.tolist(),
        'total_rows': int(full.shape[0]),
        'total_cols': int(full.shape[1]),
    }


def _normalize_grid_rows(columns, rows):
    columns = list(map(str, columns or []))
    width = len(columns)
    normalized = []
    for row in rows or []:
        seq = row if isinstance(row, (list, tuple)) else [row]
        seq = ['' if value is None else str(value) for value in seq]
        if len(seq) < width:
            seq.extend([''] * (width - len(seq)))
        elif len(seq) > width:
            seq = seq[:width]
        normalized.append(seq)
    return columns, normalized

def _split_csv(value, cast=None):
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [x.strip() for x in str(value).split(',') if x.strip()]
    if cast is None:
        return items
    out = []
    for item in items:
        try:
            out.append(cast(item))
        except Exception:
            pass
    return out


def _safe_int(value, default=None):
    if value is None:
        return default
    s = str(value).strip()
    if s == '':
        return default
    try:
        return int(float(s))
    except Exception:
        return default


def _safe_float(value, default=None):
    if value is None:
        return default
    s = str(value).strip()
    if s == '':
        return default
    try:
        return float(s)
    except Exception:
        return default


def _stop_words_arg(value):
    s = '' if value is None else str(value).strip()
    return [s] if s else []


def _objects_payload(pbx):
    import pandas as pd
    sections = []

    def add(title, df):
        try:
            count = int(df.shape[0])
        except Exception:
            count = 0
        sections.append({
            'title': title,
            'count': count,
            'html': _df_html(df, index=False),
            'grid': _df_grid_payload(df, limit=max(1, int(df.shape[0]) if getattr(df, 'shape', [0])[0] else 1), page=1)
        })

    add('Authors and IDs', pbx.table_id_aut)
    add('Authors Keywords and IDs', pbx.table_id_kwa)
    add('Countries and IDs', pbx.table_id_ctr)
    add('Documents and IDs', pbx.table_id_doc)
    add('Documents by Type', pbx.id_doc_types())
    add('Institutions and IDs', pbx.table_id_uni)
    add('Keywords Plus and IDs', pbx.table_id_kwp)
    ref_df = pd.DataFrame({'ID': pbx.u_ref_id, 'Reference': pbx.u_ref})
    add('References and IDs', ref_df)
    add('Sources and IDs', pbx.table_id_jou)
    sections.sort(key=lambda item: item['title'].lower())
    return sections


def _dataset_sep(value, default='\t'):
    token = default if value is None else str(value)
    lookup = {
        '\\t': '\t', 'tab': '\t', '\t': '\t',
        ',': ',', 'comma': ',',
        ';': ';', 'semicolon': ';',
        '|': '|', 'pipe': '|'}
    return lookup.get(token, token if len(token) == 1 else default)


def _coerce_form_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _save_uploaded_file(file_storage):
    suffix = os.path.splitext(getattr(file_storage, 'filename', '') or '')[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()
    file_storage.save(tmp_path)
    return tmp_path


def _dataset_to_text(df, sep='\t'):
    buf = StringIO()
    df.to_csv(buf, index=False, sep=sep)
    return buf.getvalue()


def _dataset_payload(df, sep='	', preview_limit=20, page=1):
    preview_limit = max(1, int(preview_limit or 20))
    return {
        'rows': int(df.shape[0]),
        'cols': int(df.shape[1]),
        'columns': list(map(str, df.columns.tolist())),
        'preview_html': _df_html(df, index=False, limit=preview_limit),
        'grid': _df_grid_payload(df, limit=preview_limit, page=page),
        'full_grid': _df_grid_full_payload(df),
        'csv_text': _dataset_to_text(df, sep=sep),
        'sep': '\t' if sep == '	' else sep,
        'preview_limit': preview_limit
    }


def _load_dataset_into_state(df, db=None, preserve_original=False, filepath=None):
    try:
        from .pbx import pbx_probe
    except ImportError:
        from pybibx.base.pbx import pbx_probe

    db = (db or STATE.get('db') or 'scopus').lower()
    if STATE.get('pbx') is None:
        STATE['pbx'] = pbx_probe.from_dataframe(df, db=db)
    else:
        STATE['pbx'].load_database_df(df)
        STATE['pbx'].database = db
    STATE['db'] = db
    if filepath is not None:
        STATE['filepath'] = filepath
    if preserve_original or STATE.get('original_data') is None:
        STATE['original_data'] = STATE['pbx'].data.copy(deep=True)
    _invalidate_topic_state()
    return STATE['pbx']


def _invalidate_topic_state():
    STATE['topic_model'] = None
    pbx = STATE.get('pbx')
    if pbx is None:
        return
    for attr in ('topic_model', 'topic_info', 'topics', 'probs', 'topic_corpus', 'embds', 'model_wv', 'model_cp', 'model_vc'):
        if hasattr(pbx, attr):
            try:
                setattr(pbx, attr, None)
            except Exception:
                pass


def _topic_model_ready():
    pbx = STATE.get('pbx')
    if pbx is None:
        return False
    return getattr(pbx, 'topic_model', None) is not None and getattr(pbx, 'topic_info', None) is not None


def _word_embeddings_ready():
    pbx = STATE.get('pbx')
    if pbx is None:
        return False
    return getattr(pbx, 'model_wv', None) is not None and getattr(pbx, 'model_vc', None) is not None

# ═════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/status')
def status():
    loaded = STATE['pbx'] is not None
    info = {}
    if loaded:
        pbx = STATE['pbx']
        try:
            info = {
                'docs': int(pbx.data.shape[0]),
                'db': STATE['db'],
                'filepath': os.path.basename(STATE['filepath'] or ''),
                'topic_ready': _topic_model_ready(),
                'word_embeddings_ready': _word_embeddings_ready(),
            }
        except Exception:
            pass
    return jsonify({'loaded': loaded, 'info': info})

@app.route('/api/objects')
def objects():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    try:
        return jsonify({'ok': True, 'sections': _objects_payload(STATE['pbx'])})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})

@app.route('/api/dataset/current')
def dataset_current():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    try:
        sep = _dataset_sep(request.args.get('sep'), '	')
        preview_limit = _safe_int(request.args.get('preview_limit'), 20)
        page = _safe_int(request.args.get('page'), 1)
        return jsonify({'ok': True, **_dataset_payload(STATE['pbx'].data, sep=sep, preview_limit=preview_limit, page=page)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})


@app.route('/api/dataset/export')
def dataset_export():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    try:
        sep = _dataset_sep(request.args.get('sep'), '\t')
        return jsonify({'ok': True, 'filename': 'data.csv', 'sep': '\\t' if sep == '\t' else sep, 'csv_text': _dataset_to_text(STATE['pbx'].data, sep=sep)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})

@app.route('/api/dataset/save', methods=['POST'])
def dataset_save():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d = request.json or {}
    text = d.get('csv_text', '')
    if not str(text).strip():
        return jsonify({'ok': False, 'error': 'Dataset editor is empty'})
    sep = _dataset_sep(d.get('sep'), '	')
    preview_limit = _safe_int(d.get('preview_limit'), 20)
    page = _safe_int(d.get('page'), 1)
    try:
        df = pd.read_csv(StringIO(text), dtype=str, sep=sep, keep_default_na=False)
        expected = list(map(str, STATE['pbx'].data.columns.tolist()))
        received = list(map(str, df.columns.tolist()))
        if expected != received:
            return jsonify({'ok': False, 'error': 'Edited dataset must keep the same columns and order as the current dataset.'})
        _load_dataset_into_state(df, db=STATE.get('db'), preserve_original=True, filepath=STATE.get('filepath'))
        return jsonify({'ok': True, 'message': 'Dataset changes applied and dataset reloaded.', 'docs': int(STATE['pbx'].data.shape[0]), 'db': STATE.get('db'), **_dataset_payload(STATE['pbx'].data, sep=sep, preview_limit=preview_limit, page=page)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})




@app.route('/api/dataset/save_grid', methods=['POST'])
def dataset_save_grid():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d = request.json or {}
    sep = _dataset_sep(d.get('sep'), '	')
    preview_limit = _safe_int(d.get('preview_limit'), 20)
    page = _safe_int(d.get('page'), 1)
    columns = d.get('columns') or []
    rows = d.get('rows') or []
    try:
        expected = list(map(str, STATE['pbx'].data.columns.tolist()))
        columns, rows = _normalize_grid_rows(columns, rows)
        if columns != expected:
            return jsonify({'ok': False, 'error': 'Edited dataset must keep the same columns and order as the current dataset.'})
        df = pd.DataFrame(rows, columns=columns, dtype=str).fillna('')
        STATE['pbx'].load_database_df(df)
        _invalidate_topic_state()
        return jsonify({'ok': True, 'message': 'Dataset grid changes applied and recomputed.', 'docs': int(STATE['pbx'].data.shape[0]), 'db': STATE.get('db'), **_dataset_payload(STATE['pbx'].data, sep=sep, preview_limit=preview_limit, page=page)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})

@app.route('/api/dataset/apply', methods=['POST'])
def dataset_apply():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d = request.json or {}
    sep = _dataset_sep(d.get('sep'), '	')
    preview_limit = _safe_int(d.get('preview_limit'), 20)
    page = _safe_int(d.get('page'), 1)
    text = d.get('csv_text', '')
    if not str(text).strip():
        return jsonify({'ok': False, 'error': 'Dataset editor is empty'})
    try:
        STATE['pbx'].load_database_text(text, sep = sep)
        _invalidate_topic_state()
        return jsonify({'ok': True, 'message': 'Dataset applied, bibfile.data replaced, and all objects recomputed.', 'docs': int(STATE['pbx'].data.shape[0]), 'db': STATE.get('db'), **_dataset_payload(STATE['pbx'].data, sep=sep, preview_limit=preview_limit, page=page)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})

@app.route('/api/dataset/load_csv', methods=['POST'])
def dataset_load_csv():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    f = request.files.get('file')
    sep = _dataset_sep(request.form.get('sep'), '	')
    preview_limit = _safe_int(request.form.get('preview_limit'), 20)
    page = _safe_int(request.form.get('page'), 1)
    csv_text = request.form.get('csv_text', '')
    if f is None and not str(csv_text).strip():
        return jsonify({'ok': False, 'error': 'No edited dataset file provided'})
    try:
        if f is not None:
            raw = f.read()
            text = raw.decode('utf-8-sig', errors='replace')
            filename = f.filename or 'edited_dataset.csv'
        else:
            text = csv_text
            filename = 'edited_dataset.csv'
        STATE['pbx'].load_database_text(text, sep = sep)
        STATE['filepath'] = filename
        _invalidate_topic_state()
        return jsonify({'ok': True, 'message': 'Edited dataset reapplied, bibfile.data replaced, and all objects recomputed.', 'docs': int(STATE['pbx'].data.shape[0]), 'db': STATE.get('db'), **_dataset_payload(STATE['pbx'].data, sep=sep, preview_limit=preview_limit, page=page)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})


@app.route('/api/dataset/reset', methods=['POST'])
def dataset_reset():
    if not STATE['pbx'] or STATE.get('original_data') is None:
        return jsonify({'ok': False, 'error': 'No original dataset snapshot available'})
    try:
        payload = request.json or {} if request.is_json else {}
        sep = _dataset_sep(payload.get('sep'), '	')
        preview_limit = _safe_int(payload.get('preview_limit'), 20)
        page = _safe_int(payload.get('page'), 1)
        _load_dataset_into_state(STATE['original_data'].copy(deep=True), db=STATE.get('db'), preserve_original=True, filepath=STATE.get('filepath'))
        return jsonify({'ok': True, 'message': 'Dataset restored to the original loaded snapshot.', 'docs': int(STATE['pbx'].data.shape[0]), 'db': STATE.get('db'), **_dataset_payload(STATE['pbx'].data, sep=sep, preview_limit=preview_limit, page=page)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})


@app.route('/api/upload', methods=['POST'])
def upload():
    f   = request.files.get('file')
    db  = request.form.get('db', 'scopus')
    dup = _coerce_form_bool(request.form.get('del_duplicated', 'true'), default=True)
    expand_refs = _coerce_form_bool(request.form.get('expand_references', 'false'), default=False)
    if not f:
        return jsonify({'ok': False, 'error': 'No file provided'})
    tmp_path = _save_uploaded_file(f)
    STATE['filepath'] = tmp_path
    STATE['db'] = db
    try:
        try:
            from .pbx import pbx_probe
        except ImportError:
            from pybibx.base.pbx import pbx_probe
        out = StringIO()
        with redirect_stdout(out):
            STATE['pbx'] = pbx_probe(tmp_path, db=db, del_duplicated=dup, expand_references=expand_refs if str(db).lower() == 'openalex' else False)
        pbx = STATE['pbx']
        STATE['original_data'] = pbx.data.copy(deep=True)
        _invalidate_topic_state()
        return jsonify({
            'ok': True,
            'docs': int(pbx.data.shape[0]),
            'db': db,
            'stdout': out.getvalue(),
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})


@app.route('/api/upload_merge', methods=['POST'])
def upload_merge():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'Load a dataset before merging another file'})
    f   = request.files.get('file')
    db  = request.form.get('db', STATE.get('db') or 'scopus')
    dup = _coerce_form_bool(request.form.get('del_duplicated', 'true'), default=True)
    expand_refs = _coerce_form_bool(request.form.get('expand_references', 'false'), default=False)
    if not f:
        return jsonify({'ok': False, 'error': 'No file provided'})
    tmp_path = _save_uploaded_file(f)
    try:
        out = StringIO()
        with redirect_stdout(out):
            STATE['pbx'].merge_database(file_bib=tmp_path, db=db, del_duplicated=dup, expand_references=expand_refs if str(db).lower() == 'openalex' else False)
        STATE['original_data'] = STATE['pbx'].data.copy(deep=True)
        _invalidate_topic_state()
        return jsonify({
            'ok': True,
            'docs': int(STATE['pbx'].data.shape[0]),
            'db': STATE.get('db'),
            'stdout': out.getvalue(),
            **_dataset_payload(STATE['pbx'].data, sep='\t', preview_limit=20, page=1)
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})

@app.route('/api/eda', methods=['POST'])
def eda():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    return jsonify(run_fn(STATE['pbx'].eda_bib))

@app.route('/api/health', methods=['POST'])
def health():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    return jsonify(run_fn(STATE['pbx'].health_bib))


@app.route('/api/filter', methods=['POST'])
def filter_bib():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d = request.json or {}
    kwargs = {}
    docs = _split_csv(d.get('documents', ''), int)
    if docs:
        kwargs['documents'] = docs
    doc_types = _split_csv(d.get('doc_type', ''))
    if doc_types:
        kwargs['doc_type'] = doc_types
    year_str = _safe_int(d.get('year_str'), -1)
    year_end = _safe_int(d.get('year_end'), -1)
    kwargs['year_str'] = year_str if year_str is not None else -1
    kwargs['year_end'] = year_end if year_end is not None else -1
    sources = _split_csv(d.get('sources', ''))
    if sources:
        kwargs['sources'] = sources
    core = _safe_int(d.get('core'), -1)
    kwargs['core'] = core if core is not None else -1
    country = _split_csv(d.get('country', ''))
    if country:
        kwargs['country'] = country
    language = _split_csv(d.get('language', ''))
    if language:
        kwargs['language'] = language
    kwargs['abstract'] = bool(d.get('abstract', False))
    res = run_fn(STATE['pbx'].filter_bib, **kwargs)
    if res['ok']:
        _invalidate_topic_state()
        try:
            res['docs'] = int(STATE['pbx'].data.shape[0])
        except Exception:
            pass
    return jsonify(res)


@app.route('/api/wordcloud', methods=['POST'])
def wordcloud():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d = request.json or {}
    return jsonify(run_fn(STATE['pbx'].word_cloud_plot,
        entry=d.get('entry','kwp'),
        wordsn=int(d.get('wordsn', 300)),
        rmv_custom_words=[x.strip() for x in d.get('rmv_words','').split(',') if x.strip()]
    ))

@app.route('/api/ngrams', methods=['POST'])
def ngrams():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d = request.json or {}
    ngrams_val = _safe_int(d.get('ngrams', 1), None)
    if ngrams_val is None:
        return jsonify({'ok': False, 'error': 'ngrams argument must be an integer'})
    wordsn_val = _safe_int(d.get('wordsn', 20), 20)
    return jsonify(run_fn(STATE['pbx'].get_top_ngrams,
        entry=d.get('entry','kwp'),
        ngrams=ngrams_val,
        wordsn=wordsn_val,
        stop_words=_stop_words_arg(d.get('lang')),
        rmv_custom_words=[x.strip() for x in d.get('rmv_words','').split(',') if x.strip()]
    ))

@app.route('/api/treemap', methods=['POST'])
def treemap():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d = request.json or {}
    return jsonify(run_fn(STATE['pbx'].tree_map,
        entry=d.get('entry','kwp'),
        topn=int(d.get('topn',25))
    ))

@app.route('/api/bars', methods=['POST'])
def bars():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d = request.json or {}
    return jsonify(run_fn(STATE['pbx'].plot_bars,
        statistic=d.get('statistic','dpy'),
        topn=int(d.get('topn',20))
    ))

@app.route('/api/evolution', methods=['POST'])
def evolution():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d = request.json or {}
    lang = d.get('lang', '')
    return jsonify(run_fn(STATE['pbx'].plot_evolution_year,
        key=d.get('key','kwp'),
        topn=int(d.get('topn',10)),
        start=int(d.get('start',2010)),
        end=int(d.get('end',2025)),
        stop_words=_stop_words_arg(lang),
        rmv_custom_words=_split_csv(d.get('rmv_words','')),
        txt_font_size=int(d.get('txt_font_size', 12))
    ))


@app.route('/api/term_growth', methods=['POST'])
def term_growth():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d = request.json or {}
    lang = d.get('lang', '')
    source = d.get('source', 'kwa')
    if isinstance(source, str):
        source = [x.strip() for x in source.split(',') if x.strip()]
    if not source:
        source = ['kwa']
    return jsonify(run_fn(STATE['pbx'].term_growth,
        source=source,
        topn=int(d.get('topn', 10)),
        cumulative=bool(d.get('cumulative', True)),
        stop_words=_stop_words_arg(lang),
        rmv_custom_words=_split_csv(d.get('rmv_words', '')),
        start=int(d.get('start', -1)),
        end=int(d.get('end', -1)),
        line=bool(d.get('line', True)),
        bubble=bool(d.get('bubble', True)),
        view='browser'
    ))


@app.route('/api/sankey', methods=['POST'])
def sankey():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d = request.json or {}
    layers = d.get('layers') or []
    entry = []
    topn = []
    if isinstance(layers, list) and layers:
        for layer in layers:
            code = str((layer or {}).get('entry', '')).strip()
            if not code:
                continue
            entry.append(code)
            n = _safe_int((layer or {}).get('topn'), None)
            if len(entry) > 1:
                topn.append(n)
    else:
        raw = d.get('entry', 'aut,cout,jou')
        entry = [x.strip() for x in str(raw).split(',') if x.strip()]
        raw_topn = str(d.get('topn', '')).strip()
        if raw_topn:
            if ',' in raw_topn:
                topn = [_safe_int(x.strip(), None) for x in raw_topn.split(',') if x.strip()]
            else:
                single_topn = _safe_int(raw_topn, None)
                if single_topn is not None and len(entry) >= 2:
                    topn = [single_topn] * (len(entry) - 1)
    if len(entry) < 2:
        return jsonify({'ok': False, 'error': 'Sankey requires at least two layers.'})
    if topn:
        topn = [15 if n is None else n for n in topn]
        if len(topn) < len(entry) - 1:
            topn.extend([topn[-1]] * (len(entry) - 1 - len(topn)))
        topn = topn[:len(entry) - 1]
    else:
        topn = None
    return jsonify(run_fn(STATE['pbx'].sankey_diagram,
        entry=entry,
        topn=topn,
        rmv_unknowns=bool(d.get('rmv_unknowns', True))
    ))

@app.route('/api/productivity', methods=['POST'])
def productivity():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d    = request.json or {}
    kind = d.get('kind', 'authors')
    topn = int(d.get('topn', 20))
    pbx  = STATE['pbx']
    fn   = {
        'authors':     pbx.authors_productivity,
        'countries':   pbx.countries_productivity,
        'institution': pbx.institution_productivity,
        'source':      pbx.source_productivity,
    }.get(kind, pbx.authors_productivity)
    if kind == 'countries':
        return jsonify(run_fn(fn))
    return jsonify(run_fn(fn, topn=topn))

@app.route('/api/network', methods=['POST'])
def network():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d    = request.json or {}
    kind = d.get('kind', 'adj')
    pbx  = STATE['pbx']
    if kind == 'adj':
        return jsonify(run_fn(pbx.network_adj,
            adj_type=d.get('adj_type','aut'),
            min_count=int(d.get('min_count',2)),
            node_labels=bool(d.get('node_labels', False)),
            node_size=int(d.get('node_size', -1)),
            label_type=d.get('label_type', 'id'),
            centrality=d.get('centrality') or None
        ))
    elif kind == 'map':
        return jsonify(run_fn(pbx.network_adj_map,
            connections=bool(d.get('connections', True)),
            country_lst=_split_csv(d.get('country_lst',''))
        ))
    elif kind == 'sim':
        res = run_fn(pbx.network_sim,
            sim_type=d.get('sim_type','coup'),
            node_size=int(d.get('node_size', -1)),
            node_labels=bool(d.get('node_labels', False)),
            cut_coup=float(d.get('cut_coup',0.3)),
            cut_cocit=int(d.get('cut_cocit',5))
        )
        if res.get('ok') and getattr(pbx, 'sim_table', None) is not None:
            res['tables'] = {'similarity_values': _df_html(pbx.sim_table, index=False)}
        return jsonify(res)
    elif kind == 'hist':
        res = run_fn(pbx.network_hist,
            min_links=int(d.get('min_links',1)),
            chain=_split_csv(d.get('chain',''), int),
            path=bool(d.get('path', False)),
            node_size=int(d.get('node_size',20)),
            font_size=int(d.get('font_size',10)),
            node_labels=bool(d.get('node_labels', True)),
            dist=float(d.get('dist',0.7)),
            dist_pad=float(d.get('dist_pad',1.0))
        )
        return jsonify(res)
    elif kind == 'main_path':
        try:
            result = pbx.main_path_analysis(
                method=d.get('method', 'spc'),
                min_path_size=int(d.get('min_path_size', 2)),
                strict_year=bool(d.get('strict_year', True))
            )
            path_df = result.get('path_table')
            edges_df = result.get('edge_weights')
            main_path = result.get('path', [])
            return jsonify({'ok': True, 'stdout': '', 'plotly': [], 'images': [], 'result': {'type': 'text', 'value': 'Main path: ' + (' → '.join(map(str, main_path)) if main_path else 'No main path found')}, 'tables': {
                'main_path': _df_html(path_df, index=False),
                'main_path_edge_weights': _df_html(edges_df, index=False)
            }})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})
    elif kind == 'adj_dir':
        res = run_fn(pbx.network_adj_dir,
            min_count=int(d.get('min_count',7)),
            node_labels=bool(d.get('node_labels', True)),
            local_nodes=bool(d.get('local_nodes', False)),
            node_size=int(d.get('node_size', 20)),
            font_size=int(d.get('font_size', 10))
        )
        return jsonify(res)
    elif kind == 'find_dir':
        return jsonify(run_fn(pbx.find_nodes_dir,
            article_ids=_split_csv(d.get('article_ids',''), int),
            ref_ids=_split_csv(d.get('ref_ids','')),
            node_size=int(d.get('node_size',20)),
            font_size=int(d.get('font_size',10))
        ))
    elif kind == 'salsa':
        try:
            result, top_by_decade_a, top_by_decade_h = pbx.salsa(
                max_iter=int(d.get('max_iter',150)),
                tol=float(d.get('tol',1e-6)),
                topn_decade=int(d.get('topn_decade',5))
            )
            import pandas as pd
            decades = []
            for decade in sorted(top_by_decade_h.keys()):
                for node_id, score in top_by_decade_h[decade]:
                    decades.append({'Decade': f'{decade}s', 'Role': 'Hub', 'Node ID': node_id, 'Score': score})
            for decade in sorted(top_by_decade_a.keys()):
                for node_id, score in top_by_decade_a[decade]:
                    decades.append({'Decade': f'{decade}s', 'Role': 'Authority', 'Node ID': node_id, 'Score': score})
            top_nodes = pd.DataFrame({
                'Node ID': result['node_ids'],
                'Hub Score': result['hubs'],
                'Authority Score': result['authorities'],
            }).sort_values(['Hub Score','Authority Score'], ascending=False).reset_index(drop=True)
            return jsonify({'ok': True, 'stdout': '', 'plotly': [], 'images': [], 'tables': {
                'salsa_top_nodes': _df_html(top_nodes.head(50), index=False),
                'salsa_by_decade': _df_html(pd.DataFrame(decades), index=False)
            }})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})
    return jsonify({'ok': False, 'error': 'Unknown network kind'})

@app.route('/api/refs', methods=['POST'])
def refs():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d    = request.json or {}
    kind = d.get('kind', 'top_refs')
    pbx  = STATE['pbx']

    if kind == 'top_refs':
        return jsonify(run_fn(pbx.plot_top_refs, topn=int(d.get('topn',15)), font_size=int(d.get('font_size',10)), use_ref_id=bool(d.get('use_ref_id', False)), date_start=int(d['date_start']) if d.get('date_start') else None, date_end=int(d['date_end']) if d.get('date_end') else None))
    elif kind == 'rpys':
        return jsonify(run_fn(pbx.plot_rpys, peaks_only=bool(d.get('peaks_only', False))))
    elif kind == 'trajectory':
        ids = _split_csv(d.get('ref_ids',''))
        names = _split_csv(d.get('ref_names',''))
        return jsonify(run_fn(pbx.plot_citation_trajectory, ref_names=names, ref_ids=ids))
    elif kind == 'ref_matrix':
        ids = _split_csv(d.get('ref_ids',''))
        return jsonify(run_fn(pbx.ref_citation_matrix, tgt_ref_id=ids, date_start=int(d['date_start']) if d.get('date_start') else None, date_end=int(d['date_end']) if d.get('date_end') else None))
    elif kind == 'co_refs':
        return jsonify(run_fn(pbx.top_cited_co_references, group=int(d.get('group',2)), topn=int(d.get('topn',10))))
    elif kind == 'co_citation_network':
        res = run_fn(pbx.plot_co_citation_network, target_ref_id=d.get('target_ref_id',''), topn=int(d.get('topn',20)))
        if res.get('ok') and getattr(pbx, 'top_co_c', None) is not None:
            res['tables'] = {'co_citation_network': _df_html(pbx.top_co_c, index=False)}
        return jsonify(res)
    elif kind == 'sleeping_beauties':
        res = run_fn(pbx.detect_sleeping_beauties, topn=int(d.get('topn',10)), min_count=int(d.get('min_count',5)))
        if res.get('ok') and res.get('result', {}).get('type') == 'dataframe':
            try:
                import pandas as pd
                STATE['sleeping_beauties'] = pd.read_html(res['result']['html'])[0]
            except Exception:
                STATE['sleeping_beauties'] = None
        return jsonify(res)
    elif kind == 'princes':
        metrics = STATE.get('sleeping_beauties')
        if metrics is None:
            metrics = pbx.detect_sleeping_beauties(topn=int(d.get('topn',10)), min_count=int(d.get('min_count',5)))
            STATE['sleeping_beauties'] = metrics
        return jsonify(run_fn(pbx.detect_princes, metrics=metrics))
    elif kind == 'reference_diversity':
        try:
            result = pbx.reference_diversity(paper_ids=_split_csv(d.get('paper_ids',''), int) or None)
            return jsonify({'ok': True, 'stdout': '', 'plotly': [], 'images': [], 'result': {'type': 'text', 'value': f'Reference diversity computed for {len(result):,} paper(s).'}, 'tables': {
                'reference_diversity': _df_html(result, index=False)
            }})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})
    elif kind == 'disruption_index':
        try:
            result = pbx.disruption_index(
                paper_ids=_split_csv(d.get('paper_ids',''), int) or None,
                strict_future=bool(d.get('strict_future', True)),
                min_future_citers=int(d.get('min_future_citers', 1))
            )
            return jsonify({'ok': True, 'stdout': '', 'plotly': [], 'images': [], 'result': {'type': 'text', 'value': f'Disruption index computed for {len(result):,} paper(s).'}, 'tables': {
                'disruption_index': _df_html(result, index=False)
            }})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})
    return jsonify({'ok': False, 'error': 'Unknown refs kind'})

@app.route('/api/cross', methods=['POST'])
def cross():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d = request.json or {}
    kind = d.get('kind', 'count_y_x')
    pbx = STATE['pbx']
    if kind == 'count_y_x':
        res = run_fn(pbx.plot_count_y_per_x,
            rmv_unknowns=bool(d.get('rmv_unknowns', True)),
            x=d.get('x', 'cout'),
            y=d.get('y', 'aut'),
            topn_x=int(d.get('topn_x', 5)),
            topn_y=int(d.get('topn_y', 5)),
            text_font_size=int(d.get('text_font_size', 12)),
            x_angle=int(d.get('x_angle', -90))
        )
        if res.get('ok') and getattr(pbx, 'top_y_x', None) is not None:
            res['tables'] = {'count_y_per_x': _df_html(pbx.top_y_x, index=False)}
        return jsonify(res)
    elif kind == 'heatmap_y_x':
        res = run_fn(pbx.plot_heatmap_y_per_x,
            x=d.get('x', 'kwa'),
            y=d.get('y', 'aut'),
            topn_x=int(d.get('topn_x', 15)),
            topn_y=int(d.get('topn_y', 5)),
            element_x=_split_csv(d.get('element_x', '')),
            element_y=_split_csv(d.get('element_y', '')),
            rmv_unknowns=bool(d.get('rmv_unknowns', True))
        )
        if res.get('ok') and getattr(pbx, 'heat_y_x', None) is not None:
            res['tables'] = {'heatmap_y_per_x': _df_html(pbx.heat_y_x, index=False)}
        return jsonify(res)
    return jsonify({'ok': False, 'error': 'Unknown cross kind'})

@app.route('/api/projection', methods=['POST'])
def projection():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d = request.json or {}
    return jsonify(run_fn(STATE['pbx'].docs_projection,
        corpus_type=d.get('corpus_type','abs'),
        stop_words=_stop_words_arg(d.get('lang')),
        rmv_custom_words=_split_csv(d.get('rmv_words','')),
        n_components=int(d.get('n_components',2)),
        n_clusters=int(d.get('n_clusters',5)),
        node_labels=bool(d.get('node_labels', True)),
        node_size=int(d.get('node_size',20)),
        node_font_size=int(d.get('node_font_size',8)),
        tf_idf=bool(d.get('tf_idf', False)),
        embeddings=bool(d.get('embeddings', False)),
        model=d.get('model','allenai/scibert_scivocab_uncased'),
        method=d.get('method','tsvd'),
        showlegend=bool(d.get('showlegend', True)),
        cluster_method=d.get('cluster_method','kmeans'),
        min_size=int(d.get('min_size',5)),
        max_size=int(d.get('max_size',50))
    ))

@app.route('/api/topics', methods=['POST'])
def topics():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d = request.json or {}
    res = run_fn(STATE['pbx'].topics_creation,
        stop_words=_stop_words_arg(d.get('lang')),
        rmv_custom_words=_split_csv(d.get('rmv_words','')),
        embeddings=bool(d.get('embeddings', False)),
        model=d.get('model', 'allenai/scibert_scivocab_uncased')
    )
    if res.get('ok'):
        STATE['topic_model'] = getattr(STATE['pbx'], 'topic_model', None)
    return jsonify(res)

@app.route('/api/topics_graph', methods=['POST'])
def topics_graph():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    if not _topic_model_ready():
        return jsonify({'ok': False, 'error': 'Run Create Topics first to build the topic model.'})
    d    = request.json or {}
    kind = d.get('kind', 'topics')
    pbx  = STATE['pbx']
    fn   = {
        'topics':       pbx.graph_topics,
        'distribution': pbx.graph_topics_distribution,
        'projection':   pbx.graph_topics_projection,
        'heatmap':      pbx.graph_topics_heatmap,
        'time':         pbx.graph_topics_time,
    }.get(kind, pbx.graph_topics)
    return jsonify(run_fn(fn))

@app.route('/api/ai_tools', methods=['POST'])
def ai_tools():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d = request.json or {}
    kind = d.get('kind', 'topics_authors')
    pbx = STATE['pbx']
    if kind in ('topics_authors', 'topics_representatives', 'topics_words') and not _topic_model_ready():
        return jsonify({'ok': False, 'error': 'Run Create Topics first to build the topic model.'})
    if kind == 'topics_authors':
        return jsonify(run_fn(pbx.topics_authors, topn=int(d.get('topn', 15))))
    elif kind == 'topics_representatives':
        return jsonify(run_fn(pbx.topics_representatives))
    elif kind == 'topics_words':
        return jsonify(run_fn(pbx.topics_words, doc_id=int(d.get('doc_id', 0))))
    elif kind == 'summarize':
        mode = str(d.get('mode', 'ext')).strip().lower()
        article_ids = _split_csv(d.get('article_ids', ''))
        model_name = d.get('model') or None
        if mode == 'abs':
            kwargs = {'article_ids': article_ids}
            if model_name:
                kwargs['model_name'] = model_name
            return jsonify(run_fn(pbx.summarize_abst_peg, **kwargs))
        kwargs = {'article_ids': article_ids}
        if model_name:
            kwargs['model_name'] = model_name
        return jsonify(run_fn(pbx.summarize_ext_bert, **kwargs))
    elif kind == 'create_embeddings':
        try:
            import pandas as pd

            pbx.create_embeddings(
                stop_words=_stop_words_arg(d.get('lang')),
                rmv_custom_words=_split_csv(d.get('rmv_words','')),
                corpus_type=d.get('corpus_type', 'abs'),
                model=d.get('model', 'allenai/scibert_scivocab_uncased')
            )

            emb = getattr(pbx, 'embds', None)
            if emb is None:
                return jsonify({
                    'ok': True,
                    'stdout': '',
                    'plotly': [],
                    'images': [],
                    'result': {'type': 'text', 'value': 'Embeddings created, but no embds object was found.'}
                })

            df = pd.DataFrame(emb)
            return jsonify({
                'ok': True,
                'stdout': '',
                'plotly': [],
                'images': [],
                'result': {'type': 'text', 'value': f'Embeddings created successfully. Shape: {df.shape}'},
                'artifacts': [
                    {'title': 'Embeddings', 'type': 'table', 'html': _df_html(df, index=False)}
                ]
            })

        except Exception as e:
            return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})
    elif kind == 'word_embeddings':
        res = run_fn(
            pbx.word_embeddings,
            stop_words=_stop_words_arg(d.get('lang')),
            rmv_custom_words=_split_csv(d.get('rmv_words', '')),
            vector_size=_safe_int(d.get('vector_size', 100), 100),
            window=_safe_int(d.get('window', 5), 5),
            min_count=_safe_int(d.get('min_count', 1), 1),
            epochs=_safe_int(d.get('epochs', 10), 10),
        )
        if res.get('ok'):
            import pandas as pd
            vocab = list(getattr(pbx, 'model_vc', []) or [])
            res['result'] = {'type': 'text', 'value': f'Word embeddings created successfully. Vocabulary size: {len(vocab)}'}
            if vocab:
                artifacts = list(res.get('artifacts') or [])
                artifacts.append({'title': 'Vocabulary', 'type': 'table', 'html': _df_html(pd.DataFrame({'word': vocab}), index=False, limit=500)})
                res['artifacts'] = artifacts
        return jsonify(res)
    elif kind in ('word_embeddings_sim', 'word_embeddings_operations', 'plot_word_embeddings') and not _word_embeddings_ready():
        return jsonify({'ok': False, 'error': 'Run Word Embeddings first to build the embedding model.'})
    elif kind == 'word_embeddings_sim':
        word_1 = str(d.get('word_1', '')).strip()
        word_2 = str(d.get('word_2', '')).strip()
        if not word_1 or not word_2:
            return jsonify({'ok': False, 'error': 'Provide both words to compute similarity.'})
        try:
            similarity = pbx.word_embeddings_sim(word_1=word_1, word_2=word_2)
            return jsonify({
                'ok': True,
                'stdout': '',
                'plotly': [],
                'images': [],
                'result': {'type': 'text', 'value': str(similarity)}
            })
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})
    elif kind == 'word_embeddings_operations':
        positive = _split_csv(d.get('positive', ''))
        negative = _split_csv(d.get('negative', ''))
        if not positive and not negative:
            return jsonify({'ok': False, 'error': 'Provide at least one positive or negative word.'})
        try:
            import pandas as pd
            result = pbx.word_embeddings_operations(
                positive=positive,
                negative=negative,
                topn=_safe_int(d.get('topn', 10), 10),
            )
            df = pd.DataFrame(result, columns=['word', 'score'])
            return jsonify({
                'ok': True,
                'stdout': '',
                'plotly': [],
                'images': [],
                'result': {'type': 'dataframe', 'html': _df_html(df, index=False)}
            })
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})
    elif kind == 'plot_word_embeddings':
        positive = _split_csv(d.get('positive', ''))
        negative = _split_csv(d.get('negative', ''))
        if not positive:
            return jsonify({'ok': False, 'error': 'Provide at least one positive word to plot.'})
        return jsonify(run_fn(
            pbx.plot_word_embeddings,
            positive=[positive],
            negative=[negative],
            topn=_safe_int(d.get('topn', 5), 5),
            node_size=_safe_int(d.get('node_size', 10), 10),
            font_size=_safe_int(d.get('font_size', 14), 14),
        ))
    return jsonify({'ok': False, 'error': 'Unknown AI tool kind'})

@app.route('/api/profiling', methods=['POST'])
def profiling():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d    = request.json or {}
    kind = d.get('kind', 'author')
    name = d.get('name') or None
    pid  = d.get('id') or None
    topn = int(d.get('topn', 5))
    pbx  = STATE['pbx']
    fn   = {
        'author':      pbx.profiling_author,
        'affiliation': pbx.profiling_affiliation,
        'country':     pbx.profiling_country,
        'journal':     pbx.profiling_journal,
        'keyword':     pbx.profiling_keyword,
        'kwp':         pbx.profiling_keyword_plus,
    }.get(kind, pbx.profiling_author)
    if kind == 'reference':
        return jsonify(run_fn(pbx.profiling_reference, label_id=pid, topn=topn))
    return jsonify(run_fn(fn, label_name=name, label_id=pid, topn=topn))

@app.route('/api/hindex', methods=['POST'])
def hindex():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d = request.json or {}
    import pandas as pd
    pbx = STATE['pbx']
    year = _safe_int(d.get('year'), None)
    try:
        data = {
            'Author': pbx.u_aut,
            'H-index': pbx.h_index(),
            'G-index': pbx.g_index(),
            'E-index': pbx.e_index(),
            'J-index': pbx.j_index(),
        }
        if year is not None:
            data['M-index'] = pbx.m_index(current_year=year)
        df = pd.DataFrame(data)
        df = df.sort_values(['H-index', 'G-index', 'E-index', 'J-index'], ascending=False).reset_index(drop=True)
        return jsonify(run_fn(lambda: df))
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})


# ═════════════════════════════════════════════════════════════════════════════
#  HTML FRONTEND
# ═════════════════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PyBibX · Web App</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
:root{
  --bg:#0b0d12;--bg2:#11141c;--bg3:#171b27;--sur:#1c2030;--sur2:#232840;
  --bdr:rgba(255,255,255,.07);--bdr2:rgba(255,255,255,.14);
  --am:#f5a623;--am2:#ffd080;--am-d:rgba(245,166,35,.12);
  --cy:#3de3c8;--cy2:#7af0dc;--cy-d:rgba(61,227,200,.1);
  --vi:#a78bfa;--red:#f87171;--green:#34d399;
  --tx:#e8eaf2;--tx2:#9aa0b8;--tx3:#4c5270;
  --mono:'IBM Plex Mono',monospace;--sans:'Syne',sans-serif;--body:'Inter',sans-serif;
  --r:10px;--sw:260px;
}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--tx);font-family:var(--body);font-size:14px;display:flex;height:100vh;overflow:hidden;}

/* SIDEBAR */
#sidebar{width:var(--sw);min-width:var(--sw);background:var(--bg2);border-right:1px solid var(--bdr);display:flex;flex-direction:column;overflow-y:auto;}
#sidebar::-webkit-scrollbar{width:3px;}
#sidebar::-webkit-scrollbar-thumb{background:var(--bdr2);}
.logo{
  height:110px;              /* tighter container */
  padding:10px 0 6px 0;      /* less empty space around logo */
  display:flex;
  align-items:center;        /* vertical center */
  justify-content:center;    /* horizontal center */
  text-align:center;
  border-bottom:1px solid var(--bdr);
}

.logo img{
  display:block;
  width:150px;               /* fixed logo size */
  height:auto;
  margin:0 auto;
  object-fit:contain;
}
.logo-title{font-family:'DM Serif Display',serif;font-size:24px;background:linear-gradient(135deg,var(--am),var(--cy));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.logo-sub{font-family:var(--mono);font-size:10px;color:var(--tx3);letter-spacing:.1em;text-transform:uppercase;margin-top:2px;}
.db-pill{margin:10px 18px;background:var(--sur);border:1px solid var(--bdr2);border-radius:6px;padding:6px 10px;font-family:var(--mono);font-size:11px;color:var(--tx3);display:none;}
.db-pill.show{display:block;}
.db-pill span{color:var(--am2);}
.nav-sec{padding:10px 0 2px;}
.nav-lbl{font-family:var(--mono);font-size:9.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--tx3);padding:0 18px 5px;}
.nav-item{display:flex;align-items:center;gap:9px;padding:7px 18px;cursor:pointer;border-left:2px solid transparent;color:var(--tx2);font-size:13px;transition:all .18s;user-select:none;}
.nav-item:hover{background:var(--am-d);color:var(--am2);border-left-color:rgba(245,166,35,.4);}
.nav-item.active{background:var(--am-d);color:var(--am);border-left-color:var(--am);}
.nav-item.disabled{opacity:.35;pointer-events:none;}
.nav-ico{font-size:14px;min-width:18px;text-align:center;}

/* MAIN */
#main{flex:1;display:flex;flex-direction:column;overflow:hidden;}
#topbar{background:rgba(11,13,18,.9);backdrop-filter:blur(20px);border-bottom:1px solid var(--bdr);padding:10px 28px;display:flex;align-items:center;gap:12px;flex-shrink:0;}
.tb-title{font-family:var(--sans);font-size:15px;font-weight:700;color:var(--tx);}
.tb-sub{font-size:12px;color:var(--tx3);font-family:var(--mono);}
.tb-status{margin-left:auto;display:flex;align-items:center;gap:7px;font-size:12px;font-family:var(--mono);}
.dot{width:7px;height:7px;border-radius:50%;background:var(--tx3);}
.dot.green{background:var(--green);box-shadow:0 0 8px rgba(52,211,153,.5);}

/* TAB BAR */
#tab-bar{display:flex;align-items:center;gap:2px;padding:0 20px;background:var(--bg2);border-bottom:1px solid var(--bdr);flex-shrink:0;}
.tab-btn{background:none;border:none;border-bottom:2px solid transparent;color:var(--tx3);font-family:var(--mono);font-size:12px;padding:10px 18px;cursor:pointer;transition:all .18s;letter-spacing:.04em;display:flex;align-items:center;gap:6px;}
.tab-btn:hover{color:var(--tx2);}
.tab-btn.active{color:var(--am);border-bottom-color:var(--am);}
.tab-badge{background:var(--am-d);color:var(--am2);border-radius:10px;padding:1px 6px;font-size:10px;min-width:18px;text-align:center;}
#clear-btn{margin-left:auto;background:none;border:1px solid var(--bdr2);border-radius:5px;color:var(--tx3);font-size:11px;font-family:var(--mono);padding:3px 10px;cursor:pointer;transition:all .18s;}
#clear-btn:hover{color:var(--red);border-color:var(--red);}

/* BODY */
#body{flex:1;overflow:hidden;position:relative;}

/* TAB PANES */
.tab-pane{display:none;height:100%;overflow-y:scroll;padding:28px;scrollbar-gutter:stable;}
.tab-pane.active{display:block;}
.tab-pane::-webkit-scrollbar{width:4px;}
.tab-pane::-webkit-scrollbar-thumb{background:var(--bdr2);}

/* OUTPUT TAB */
#tab-results{padding:16px;}
#tab-tsg{padding:16px;}
#output-body{height:100%;}
#tsg-body{height:100%;}
#empty-tsg{display:flex;flex-direction:column;align-items:center;justify-content:center;height:300px;color:var(--tx3);gap:10px;}
#empty-tsg .e-icon{font-size:32px;opacity:.4;}
#empty-tsg p{font-size:12px;font-family:var(--mono);}
.tsg-shell{display:flex;flex-direction:column;gap:10px;}
.tsg-stats{font-size:12px;font-family:var(--mono);color:var(--tx2);padding:10px 12px;border:1px solid var(--bdr);border-radius:14px;background:linear-gradient(180deg,var(--sur2),var(--sur));}
#output-body::-webkit-scrollbar{width:4px;}
#output-body::-webkit-scrollbar-thumb{background:var(--bdr2);}

/* OUTPUT ITEMS */
.out-item{margin-bottom:16px;border-radius:var(--r);border:1px solid var(--bdr);overflow:hidden;}
.out-item-hdr{background:var(--sur);padding:8px 14px;display:flex;align-items:center;gap:8px;font-family:var(--mono);font-size:11px;color:var(--tx3);}
.out-item-hdr .tag{background:var(--am-d);color:var(--am2);padding:1px 7px;border-radius:4px;font-size:10px;}
.out-item-tabs{display:flex;gap:8px;flex-wrap:wrap;margin-left:10px;}
.out-tab{background:transparent;border:1px solid var(--bdr2);color:var(--tx2);border-radius:999px;padding:4px 10px;cursor:pointer;font-family:var(--mono);font-size:11px;}
.out-tab.active{background:var(--cy);border-color:transparent;color:#041016;}
.out-item-body{background:var(--bg);padding:0;}
.out-panel-tools{display:flex;justify-content:flex-end;gap:8px;padding:10px 12px 0 12px;background:var(--bg);}
.out-download-btn{background:transparent;border:1px solid var(--bdr2);color:var(--tx2);border-radius:999px;padding:5px 10px;cursor:pointer;font-family:var(--mono);font-size:11px;transition:all .18s;}
.out-download-btn:hover{border-color:var(--am);color:var(--am2);background:var(--am-d);}
.out-panel{display:none;padding:0;}
.out-panel.active{display:block;}
.plotly-out{width:100%;height:520px;}
.html-out-frame{display:block;width:100%;height:calc(100vh - 190px);min-height:640px;border:0;background:#060b16;}
.html-stats{padding:10px 12px;background:var(--bg);color:var(--tx2);font-family:var(--mono);font-size:11px;border-bottom:1px solid var(--bdr);}
.img-out{display:block;width:100%;border-radius:0;}
#spinner-overlay{position:fixed;inset:0;background:rgba(2,6,23,.62);backdrop-filter:blur(4px);display:none;align-items:center;justify-content:center;z-index:9999;padding:24px;}
#spinner-overlay.show{display:flex;}
.spinner-card{min-width:260px;max-width:420px;background:linear-gradient(180deg,var(--sur),var(--bg2));border:1px solid var(--bdr2);border-radius:20px;box-shadow:0 28px 80px rgba(0,0,0,.45);padding:26px 24px;display:flex;align-items:center;gap:16px;}
.spinner-orbit{width:42px;height:42px;border-radius:999px;border:3px solid rgba(255,255,255,.14);border-top-color:var(--am);border-right-color:#ffd082;animation:spin360 .9s linear infinite;flex:0 0 auto;}
.spinner-copy{display:flex;flex-direction:column;gap:5px;}
.spinner-title{font-family:var(--sans);font-size:15px;font-weight:700;color:var(--tx);}
.spinner-sub{font-family:var(--mono);font-size:11px;color:var(--tx3);text-transform:uppercase;letter-spacing:.08em;}
@keyframes spin360{to{transform:rotate(360deg);}}
.text-out{padding:12px 14px;font-family:var(--mono);font-size:12px;color:var(--cy2);white-space:pre-wrap;line-height:1.7;max-height:400px;overflow-y:auto;}
.table-wrap{overflow-x:auto;max-height:400px;overflow-y:auto;}
.error-out{padding:12px 14px;font-family:var(--mono);font-size:12px;color:var(--red);white-space:pre-wrap;}
.spin{display:none;width:100%;padding:40px;text-align:center;color:var(--tx3);font-family:var(--mono);font-size:12px;}
.spin.show{display:block;}
#empty-out{display:flex;flex-direction:column;align-items:center;justify-content:center;height:300px;color:var(--tx3);gap:10px;}
#empty-out .e-icon{font-size:32px;opacity:.4;}
#empty-out p{font-size:12px;font-family:var(--mono);}

/* PARAMS PANE */
#tab-parameters{max-width:860px;}
#params-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:280px;color:var(--tx3);gap:12px;text-align:center;}
#params-empty .pe-icon{font-size:40px;opacity:.35;}
#params-empty p{font-size:13px;font-family:var(--mono);}
#objects-body{display:flex;flex-direction:column;gap:12px;padding:18px;}
#empty-objects{display:flex;flex-direction:column;align-items:center;justify-content:center;height:280px;color:var(--tx3);gap:12px;text-align:center;}
#empty-objects .e-icon{font-size:40px;opacity:.35;}
.obj-item{border:1px solid var(--bdr);border-radius:14px;background:var(--bg2);overflow:hidden;}
.obj-head{display:flex;align-items:center;gap:10px;padding:14px 16px;cursor:pointer;user-select:none;background:linear-gradient(180deg,var(--sur),var(--bg2));}
.obj-head:hover{background:linear-gradient(180deg,var(--sur2),var(--sur));}
.obj-title{font-family:var(--sans);font-weight:700;}
.obj-count{margin-left:auto;color:var(--am2);font-family:var(--mono);font-size:12px;}
.obj-arrow{color:var(--tx2);font-size:12px;transition:transform .15s ease;}
.obj-item.open .obj-arrow{transform:rotate(90deg);}
.obj-body{display:none;padding:12px 14px 16px;}
.obj-item.open .obj-body{display:block;}
.obj-help{color:var(--tx2);font-size:12px;font-family:var(--mono);}
.params-section{display:none;}

/* UPLOAD SECTION */
.upload-wrap{max-width:560px;margin:0 auto;padding:20px 0;}
.upload-title{font-family:'DM Serif Display',serif;font-size:32px;margin-bottom:8px;}
.upload-title em{font-style:italic;color:var(--am2);}
.upload-desc{color:var(--tx2);font-size:13.5px;margin-bottom:18px;line-height:1.7;}
.upload-mode-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:0 0 14px;}
.upload-mode-card{background:var(--sur);border:1px solid var(--bdr);border-radius:var(--r);padding:14px;cursor:pointer;transition:all .2s;}
.upload-mode-card:hover{border-color:var(--am);background:var(--am-d);}
.upload-mode-card.active{border-color:var(--am);background:var(--am-d);box-shadow:0 0 0 1px rgba(244,162,97,.18) inset;}
.upload-mode-name{font-family:var(--sans);font-size:13px;font-weight:700;color:var(--am2);margin-bottom:4px;}
.upload-mode-desc{font-size:11.5px;color:var(--tx2);line-height:1.5;}
.upload-mode-note{font-family:var(--mono);font-size:11px;color:var(--tx3);margin:0 0 14px;}
.upload-mode-requirement{font-family:var(--mono);font-size:11px;color:var(--am2);margin:-4px 0 14px;display:none;}
.drop-zone{border:2px dashed var(--bdr2);border-radius:var(--r);padding:36px;text-align:center;cursor:pointer;transition:all .2s;margin-bottom:20px;}
.drop-zone:hover,.drop-zone.drag{border-color:var(--am);background:var(--am-d);}
.drop-zone input{display:none;}
.drop-icon{font-size:36px;margin-bottom:10px;}
.drop-main{font-family:var(--sans);font-size:15px;font-weight:600;color:var(--tx);margin-bottom:4px;}
.drop-sub{font-size:12px;color:var(--tx3);font-family:var(--mono);}
.drop-file-name{font-family:var(--mono);font-size:12px;color:var(--am2);margin-top:8px;}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px;}
label.fl{font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--tx3);display:block;margin-bottom:5px;}
select,input[type=text],input[type=number]{width:100%;background:var(--sur);border:1px solid var(--bdr2);border-radius:7px;color:var(--tx);font-family:var(--body);font-size:13px;padding:8px 12px;outline:none;transition:border .18s;}
select:focus,input:focus{border-color:var(--am);}
select option{background:var(--sur);}
.checkbox-row{display:flex;align-items:center;gap:8px;margin-bottom:16px;}
.checkbox-row input[type=checkbox]{accent-color:var(--am);width:15px;height:15px;}
.checkbox-row span{font-size:13px;color:var(--tx2);}
.load-btn{width:100%;background:linear-gradient(135deg,var(--am),#e8890a);border:none;border-radius:8px;color:#1a0d00;font-family:var(--sans);font-size:14px;font-weight:700;padding:12px;cursor:pointer;letter-spacing:.04em;transition:opacity .2s;}
.load-btn:hover{opacity:.88;}
.load-btn:disabled{opacity:.4;cursor:not-allowed;}

/* PANEL VIEWS */
.view{display:none;}
.view.active{display:block;}
.view-title{font-family:var(--sans);font-size:20px;font-weight:700;color:var(--tx);margin-bottom:6px;}
.view-desc{font-size:13px;color:var(--tx2);margin-bottom:22px;line-height:1.6;max-width:600px;}
.fn-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-bottom:8px;}
.fn-card{background:var(--sur);border:1px solid var(--bdr);border-radius:var(--r);padding:16px;cursor:pointer;transition:all .2s;}
.fn-card:hover{border-color:var(--am);background:var(--am-d);}
.fn-card.active{border-color:var(--am);background:var(--am-d);}
.fn-card.disabled{opacity:.48;cursor:not-allowed;pointer-events:none;filter:saturate(.7);}
.fn-card.disabled:hover{border-color:var(--bdr);background:var(--sur);}
.fn-card-icon{font-size:22px;margin-bottom:8px;}
.fn-card-name{font-family:var(--sans);font-size:13px;font-weight:600;color:var(--am2);margin-bottom:4px;}
.fn-card-desc{font-size:11.5px;color:var(--tx2);line-height:1.5;}
.fn-card-meta{margin-top:8px;font-family:var(--mono);font-size:10px;color:var(--tx3);text-transform:uppercase;letter-spacing:.08em;}
.fn-hint{font-family:var(--mono);font-size:11px;color:var(--tx3);margin-top:10px;}
.ai-group{display:none;margin-bottom:18px;}
.ai-group.active{display:block;}
.ai-group-title{font-family:var(--sans);font-size:16px;font-weight:700;color:var(--tx);margin-bottom:4px;}
.ai-group-desc{font-size:12px;color:var(--tx2);margin-bottom:14px;line-height:1.6;max-width:680px;}
.run-btn:disabled{opacity:.48;cursor:not-allowed;}

/* PARAMS BOX */
.params-box{background:var(--sur);border:1px solid var(--bdr);border-radius:var(--r);padding:20px;margin-bottom:16px;}
.params-title{font-family:var(--mono);font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--tx3);margin-bottom:14px;}
.params-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;}
.param-item{display:flex;flex-direction:column;gap:5px;}
.run-btn{background:var(--am-d);border:1px solid rgba(245,166,35,.4);border-radius:8px;color:var(--am2);font-family:var(--sans);font-size:13.5px;font-weight:700;padding:10px 28px;cursor:pointer;transition:all .2s;display:inline-flex;align-items:center;gap:8px;}
.run-btn:hover{background:rgba(245,166,35,.22);border-color:var(--am);}
.run-btn.loading{opacity:.6;cursor:wait;}
.run-btn svg{animation:spin .8s linear infinite;display:none;}
.run-btn.loading svg{display:block;}
@keyframes spin{to{transform:rotate(360deg)}}

/* TABLES */
.result-table{width:100%;border-collapse:collapse;font-size:12px;font-family:var(--mono);}
.result-table th{background:var(--sur2);color:var(--am2);padding:6px 10px;text-align:left;border-bottom:1px solid var(--bdr2);position:sticky;top:0;cursor:pointer;user-select:none;}
.result-table td{padding:5px 10px;border-bottom:1px solid rgba(255,255,255,.04);color:var(--tx2);}
.result-table tr:hover td{background:rgba(255,255,255,.03);}
.table-tools{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:10px;padding:10px 12px;border:1px solid var(--bdr);border-radius:12px;background:rgba(255,255,255,.02);}
.table-stat{font-family:var(--mono);font-size:11px;color:var(--tx3);margin-right:auto;}
.table-tools button{background:var(--sur2);color:var(--tx);border:1px solid var(--bdr2);border-radius:8px;padding:6px 10px;font-family:var(--mono);font-size:11px;cursor:pointer;}
.table-tools button:hover{border-color:var(--am);color:var(--am2);}
.result-table thead tr.filter-row th{position:sticky;top:33px;background:rgba(10,16,28,.98);padding:6px 8px;cursor:default;}
.result-table thead tr.filter-row input{width:100%;padding:6px 8px;border-radius:7px;font-size:11px;font-family:var(--mono);background:var(--bg2);border:1px solid var(--bdr2);color:var(--tx);}
.result-table thead tr.filter-row input:focus{border-color:var(--am);}
.sort-ind{display:inline-flex;align-items:center;justify-content:center;min-width:16px;margin-left:6px;color:var(--tx3);font-size:11px;}

/* STATS CARDS */
.stats-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px;}
.stat-c{background:var(--sur);border:1px solid var(--bdr);border-radius:var(--r);padding:16px;}
.stat-n{font-family:var(--sans);font-size:26px;font-weight:700;color:var(--am);}
.stat-l{font-size:11px;color:var(--tx3);font-family:var(--mono);margin-top:3px;}

/* CALLOUT */
.callout{border-radius:8px;padding:11px 14px;margin:10px 0;font-size:12.5px;border-left:3px solid;line-height:1.6;}
.callout.info{background:var(--cy-d);border-color:var(--cy);color:#a7f3ea;}
.callout.warn{background:var(--am-d);border-color:var(--am);color:#fcd68a;}

/* ENTRY REFERENCE */
.entry-ref{display:inline-grid;grid-template-columns:60px 1fr;gap:4px 10px;background:var(--bg);border:1px solid var(--bdr);border-radius:8px;padding:10px 14px;margin-bottom:14px;font-size:12px;}
.ek{font-family:var(--mono);color:var(--am2);}
.ev{color:var(--tx2);}

/* TOAST */
#toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:var(--sur2);border:1px solid var(--bdr2);border-radius:8px;padding:10px 18px;font-family:var(--mono);font-size:12px;color:var(--tx);z-index:9999;opacity:0;pointer-events:none;transition:opacity .25s;}

#toast.show{opacity:1;}
#toast.err{border-color:var(--red);color:var(--red);}
#toast.ok{border-color:var(--green);color:var(--green);}
.help-tip{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;border:1px solid var(--bdr2);color:var(--am2);font-family:var(--mono);font-size:10px;cursor:help;margin-left:6px;position:relative;vertical-align:middle;background:var(--sur2);}
.help-tip:hover::after,.help-tip:focus::after{content:attr(data-tip);position:absolute;left:0;top:22px;min-width:240px;max-width:360px;white-space:normal;padding:10px 12px;background:var(--sur2);border:1px solid var(--bdr2);border-radius:8px;color:var(--tx);font-family:var(--body);font-size:12px;line-height:1.5;z-index:50;box-shadow:0 10px 30px rgba(0,0,0,.35);}
.sankey-builder{display:flex;flex-direction:column;gap:12px;margin-bottom:12px;}
.sankey-layer{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(180px,.9fr) auto;gap:12px;align-items:end;overflow:visible;}
.sankey-layer .param-item{margin-bottom:0;}
.sankey-remove{background:none;border:1px solid var(--bdr2);color:var(--tx2);border-radius:8px;padding:8px 10px;cursor:pointer;height:38px;}
.sankey-remove:hover{border-color:var(--red);color:var(--red);}
.sankey-add{background:none;border:1px dashed var(--bdr2);color:var(--am2);border-radius:8px;padding:10px 12px;cursor:pointer;font-family:var(--mono);}
.sankey-add:hover{border-color:var(--am);}
.params-box .help-note{margin-top:10px;}

.dataset-shell{display:flex;flex-direction:column;gap:14px;}
.dataset-toolbar{display:flex;flex-wrap:wrap;align-items:end;gap:10px;padding:12px 14px;border:1px solid var(--bdr);border-radius:14px;background:var(--bg2);}
.dataset-toolbar button,.dataset-upload-label,.dataset-inline-btn{background:var(--sur2);color:var(--tx);border:1px solid var(--bdr2);border-radius:8px;padding:7px 11px;font-family:var(--mono);font-size:11px;cursor:pointer;}
.dataset-toolbar button:hover,.dataset-upload-label:hover,.dataset-inline-btn:hover{border-color:var(--am);color:var(--am2);}
.dataset-upload-label input{display:none;}
.dataset-meta{display:flex;flex-wrap:wrap;gap:10px;font-family:var(--mono);font-size:11px;color:var(--tx3);}
.dataset-meta span{padding:4px 8px;border:1px solid var(--bdr);border-radius:999px;background:rgba(255,255,255,.02);}
.dataset-workbench{display:flex;flex-wrap:wrap;gap:14px;align-items:stretch;}
.dataset-workbench.raw-hidden .dataset-pane-side{display:none;}
.dataset-workbench.raw-hidden .dataset-pane-main{flex:1 1 100%;}
.dataset-pane{flex:1 1 320px;min-width:320px;min-height:380px;border:1px solid var(--bdr);border-radius:14px;background:var(--bg2);padding:12px;resize:both;overflow:auto;}
.dataset-pane-main{flex:2 1 720px;}
.dataset-pane-side{flex:1 1 320px;}
.dataset-grid-toolbar{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:8px;flex-wrap:wrap;}
.dataset-grid-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.dataset-grid-pager{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding-top:10px;}
.dataset-grid-pager .page-label{font-family:var(--mono);font-size:11px;color:var(--tx3);}
.dataset-grid-pager input{width:84px;}
.dataset-pane-side.is-hidden{display:none;}
.dataset-grid-note{font-family:var(--mono);font-size:11px;color:var(--tx3);}
.dataset-editor{width:100%;min-height:320px;resize:vertical;background:var(--bg);border:1px solid var(--bdr2);border-radius:12px;color:var(--tx);font-family:var(--mono);font-size:12px;line-height:1.55;padding:14px;outline:none;}
.dataset-editor:focus{border-color:var(--am);}
.dataset-section-title{font-family:var(--sans);font-size:14px;font-weight:700;color:var(--tx);margin-bottom:8px;}
#dataset-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:280px;color:var(--tx3);gap:12px;text-align:center;}
#dataset-empty .e-icon{font-size:40px;opacity:.35;}
.grid-shell{width:100%;min-height:300px;border:1px solid var(--bdr2);border-radius:12px;background:var(--bg);overflow:auto;}
.object-grid-wrap{width:100%;min-height:260px;border:1px solid var(--bdr2);border-radius:12px;background:var(--bg);overflow:auto;resize:both;}
.grid-table{width:max-content;min-width:100%;border-collapse:separate;border-spacing:0;table-layout:fixed;font-size:12px;font-family:var(--mono);}
.grid-table th,.grid-table td{position:relative;min-width:160px;max-width:420px;border-bottom:1px solid rgba(255,255,255,.05);border-right:1px solid rgba(255,255,255,.04);}
.grid-table th{position:sticky;top:0;z-index:3;background:var(--sur2);color:var(--am2);padding:9px 12px;text-align:left;white-space:nowrap;}
.grid-table td{height:36px;padding:7px 10px;color:var(--tx2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;background:transparent;}
.grid-table td.row-index,.grid-table th.row-index{position:sticky;left:0;z-index:4;min-width:74px;max-width:74px;width:74px;text-align:center;background:var(--bg2);color:var(--am2);}
.grid-table th.row-index{z-index:6;background:var(--sur2);}
.grid-table tr.is-selected td{background:rgba(245,166,35,.08);} 
.grid-table tr.is-selected td.row-index{background:rgba(245,166,35,.16);}
.grid-table .row-selector{cursor:pointer;font-family:var(--mono);font-size:11px;}
.grid-table tbody tr:hover td{background:rgba(255,255,255,.03);}
.grid-table td[contenteditable="true"]{cursor:text;}
.grid-table td[contenteditable="true"]:focus{outline:2px solid var(--am);outline-offset:-2px;background:rgba(245,166,35,.08);color:var(--tx);}
.grid-table.is-readonly td{cursor:default;}
.grid-table .cell-content{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.grid-table .col-resizer{position:absolute;top:0;right:-2px;width:8px;height:100%;cursor:col-resize;z-index:5;}
.grid-empty{display:flex;align-items:center;justify-content:center;min-height:180px;color:var(--tx3);font-family:var(--mono);font-size:12px;padding:16px;text-align:center;}
.grid-caption{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:10px 12px;border-bottom:1px solid var(--bdr);font-family:var(--mono);font-size:11px;color:var(--tx3);background:rgba(255,255,255,.02);}
.grid-caption strong{color:var(--am2);font-weight:500;}
.obj-item{border:1px solid var(--bdr);border-radius:14px;background:var(--bg2);overflow:hidden;}
.obj-head{display:flex;align-items:center;gap:10px;padding:14px 16px;cursor:pointer;user-select:none;background:linear-gradient(180deg,var(--sur),var(--bg2));}
.obj-head:hover{background:linear-gradient(180deg,var(--sur2),var(--sur));}
.obj-title{font-family:var(--sans);font-weight:700;}
.obj-count{margin-left:auto;color:var(--am2);font-family:var(--mono);font-size:12px;}
.obj-arrow{color:var(--tx2);font-size:12px;transition:transform .15s ease;}
.obj-item.open .obj-arrow{transform:rotate(90deg);}
.obj-body{display:none;padding:12px 14px 16px;}
.obj-item.open .obj-body{display:block;}
.obj-help{color:var(--tx2);font-size:12px;font-family:var(--mono);}

</style>
</head>
<body>

<!-- ══════════════════ SIDEBAR ══════════════════ -->
<nav id="sidebar">
  <div class="logo">
    <img src="https://github.com/Valdecy/Datasets/raw/master/Data%20Science/logo.png" alt="Logo">
  </div>
  <div class="db-pill" id="db-pill">Loaded: <span id="db-info">–</span></div>

  <div class="nav-sec">
    <div class="nav-lbl">Setup</div>
    <div class="nav-item active" data-view="upload" onclick="showView('upload',this)">
      <span class="nav-ico">📂</span> Upload &amp; Load
    </div>
  </div>

  <div class="nav-sec">
    <div class="nav-lbl">Analysis</div>
    <div class="nav-item disabled" data-view="dataset" onclick="showView('dataset',this)">
      <span class="nav-ico">📊</span> Dataset &amp; Reports
    </div>
    <div class="nav-item disabled" data-view="visuals" onclick="showView('visuals',this)">
      <span class="nav-ico">🎨</span> Visualizations
    </div>
    <div class="nav-item disabled" data-view="network" onclick="showView('network',this)">
      <span class="nav-ico">🌐</span> Networks
    </div>
    <div class="nav-item disabled" data-view="temporal" onclick="showView('temporal',this)">
      <span class="nav-ico">🧭</span> Temporal Scholarly Graph
    </div>
    <div class="nav-item disabled" data-view="references" onclick="showView('references',this)">
      <span class="nav-ico">📚</span> References
    </div>
    <div class="nav-item disabled" data-view="profiling" onclick="showView('profiling',this)">
      <span class="nav-ico">🔍</span> Profiling
    </div>
    <div class="nav-item disabled" data-view="ai" data-ai-area="topics" onclick="showView('ai',this,'topics')">
      <span class="nav-ico">🧠</span> AI - Topics
    </div>
    <div class="nav-item disabled" data-view="ai" data-ai-area="words" onclick="showView('ai',this,'words')">
      <span class="nav-ico">🧬</span> AI - Words
    </div>
    <div class="nav-item disabled" data-view="ai" data-ai-area="summarize" onclick="showView('ai',this,'summarize')">
      <span class="nav-ico">📝</span> AI - Summarize
    </div>
  </div>
</nav>

<!-- ══════════════════ MAIN ══════════════════ -->
<div id="main">
  <!-- TOPBAR -->
  <div id="topbar">
    <div>
      <div class="tb-title" id="view-title">Upload Data</div>
      <div class="tb-sub" id="view-sub">Select your database export file to begin</div>
    </div>
    <div class="tb-status">
      <div class="dot" id="status-dot"></div>
      <span id="status-txt" style="color:var(--tx3)">No dataset loaded</span>
    </div>
  </div>

  <!-- TAB BAR -->
  <div id="tab-bar">
    <button class="tab-btn active" id="tbtn-functions" onclick="switchTab('functions')">⚡ Functions</button>
    <button class="tab-btn" id="tbtn-parameters" onclick="switchTab('parameters')">⚙ Parameters</button>
    <button class="tab-btn" id="tbtn-dataset" onclick="switchTab('dataset')" style="display:none">🧾 Dataset</button>
    <button class="tab-btn" id="tbtn-objects" onclick="switchTab('objects')" style="display:none">🗂 Objects &amp; IDs</button>
    <button class="tab-btn" id="tbtn-results" onclick="switchTab('results')">📊 Results <span class="tab-badge" id="results-badge" style="display:none">0</span></button>
    <button class="tab-btn" id="tbtn-tsg" onclick="switchTab('tsg')">🧭 TSG Viewer</button>
    <button id="clear-btn" onclick="clearOutput()">clear results</button>
  </div>

  <!-- BODY -->
  <div id="body">

    <!-- ════ TAB: FUNCTIONS ════ -->
    <div id="tab-functions" class="tab-pane active">

      <!-- ── UPLOAD ── -->
      <div class="view active" id="view-upload">
        <div class="upload-wrap">
          <h2 class="upload-title">Load your <em>dataset</em></h2>
          <p class="upload-desc">Export your search results from Scopus, Web of Science, or PubMed and drop the file below. PyBibX will parse, deduplicate, and prepare it for analysis.</p>

          <div class="upload-mode-grid">
            <div class="upload-mode-card active" data-upload-mode="load" onclick="setUploadMode('load', this)">
              <div class="upload-mode-name">Load dataset</div>
              <div class="upload-mode-desc">Start a new working dataset from a database export file.</div>
            </div>
            <div class="upload-mode-card" data-upload-mode="merge" onclick="setUploadMode('merge', this)">
              <div class="upload-mode-name">Merge file</div>
              <div class="upload-mode-desc">Append a new export into the current dataset and deduplicate it.</div>
            </div>
            <div class="upload-mode-card" data-upload-mode="edited" onclick="setUploadMode('edited', this)">
              <div class="upload-mode-name">Load edited dataset</div>
              <div class="upload-mode-desc">Replace the current working dataset with an edited table file.</div>
            </div>
          </div>
          <div class="upload-mode-note" id="upload-mode-note">Load a fresh export as a new working dataset.</div>
          <div class="upload-mode-requirement" id="upload-mode-requirement"></div>

          <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
            <input type="file" id="file-input" accept=".bib,.csv,.txt,.json" onchange="handleFile(this)"/>
            <div class="drop-icon">📄</div>
            <div class="drop-main" id="drop-main">Click or drag &amp; drop your file</div>
            <div class="drop-sub" id="drop-sub">.bib · .csv · .txt · .json</div>
            <div class="drop-file-name" id="file-label">No file selected</div>
          </div>

          <div class="form-row" id="upload-db-row">
            <div>
              <label class="fl">Database</label>
              <select id="db-select">
                <option value="scopus">Scopus (.bib or .csv)</option>
                <option value="wos">Web of Science (.bib)</option>
                <option value="pubmed">PubMed (.txt)</option>
                <option value="openalex">OpenAlex (.csv or .json)</option>
              </select>
            </div>
            <div style="display:flex;flex-direction:column;justify-content:flex-end;gap:8px;">
              <div class="checkbox-row">
                <input type="checkbox" id="dup-check" checked/>
                <span>Remove duplicates automatically</span>
              </div>
              <div class="checkbox-row" id="openalex-expand-row" style="display:none;">
                <input type="checkbox" id="openalex-expand-check"/>
                <span>Expand OpenAlex cited references from JSON / CR_OPENALEX (slower)</span>
              </div>
            </div>
          </div>

          <div class="form-row" id="upload-edited-row" style="display:none;grid-template-columns:1fr;">
            <div>
              <label class="fl">Edited dataset separator</label>
              <select id="upload-edited-sep">
                <option value="tab">Tab (TSV)</option>
                <option value="comma">Comma (,)</option>
                <option value="semicolon">Semicolon (;)</option>
                <option value="pipe">Pipe (|)</option>
              </select>
            </div>
          </div>

          <button class="load-btn" id="load-btn" onclick="loadData()" disabled>Load Dataset</button>
        </div>
      </div>

      <!-- ── DATASET ── -->
      <div class="view" id="view-dataset">
        <div class="view-title">Dataset &amp; Reports</div>
        <div class="view-desc">Explore your loaded dataset, run quality checks, and filter records.</div>
        <div class="stats-row" id="stats-row">
          <div class="stat-c"><div class="stat-n" id="stat-docs">–</div><div class="stat-l">Documents</div></div>
          <div class="stat-c"><div class="stat-n" id="stat-db">–</div><div class="stat-l">Database</div></div>
        </div>
        <div class="fn-grid">
          <div class="fn-card" onclick="runEda(this)">
            <div class="fn-card-icon">📊</div><div class="fn-card-name">EDA Report</div>
            <div class="fn-card-desc">Full exploratory stats: authors, citations, H-index, timespan</div>
          </div>
          <div class="fn-card" onclick="runHealth(this)">
            <div class="fn-card-icon">🩺</div><div class="fn-card-name">Health Check</div>
            <div class="fn-card-desc">Data quality report — which fields are missing and by how much</div>
          </div>
        </div>
        <div class="fn-hint">Click EDA or Health to run immediately.</div>
      </div>

      <!-- ── VISUALIZATIONS ── -->
      <div class="view" id="view-visuals">
        <div class="view-title">Visualizations</div>
        <div class="view-desc">Interactive plots for keywords, n-grams, trends, and document projections.</div>
        <div class="fn-grid">
          <div class="fn-card" onclick="selectParams('wc','visuals',this)"><div class="fn-card-icon">☁️</div><div class="fn-card-name">Word Cloud</div><div class="fn-card-desc">Visual word frequency from any text field</div></div>
          <div class="fn-card" onclick="selectParams('ngrams','visuals',this)"><div class="fn-card-icon">📊</div><div class="fn-card-name">N-Grams</div><div class="fn-card-desc">Top unigrams, bigrams, or trigrams</div></div>
          <div class="fn-card" onclick="selectParams('treemap','visuals',this)"><div class="fn-card-icon">🗂</div><div class="fn-card-name">TreeMap</div><div class="fn-card-desc">Proportional area map for keywords/authors</div></div>
          <div class="fn-card" onclick="selectParams('bars','visuals',this)"><div class="fn-card-icon">📈</div><div class="fn-card-name">Bar Charts</div><div class="fn-card-desc">Documents per year, Lotka's law, Bradford's law</div></div>
          <!-- <div class="fn-card" onclick="selectParams('evolution','visuals',this)"><div class="fn-card-icon">🌊</div><div class="fn-card-name">Evolution Plot</div><div class="fn-card-desc">Topic/keyword trends over time</div></div> -->
          <div class="fn-card" onclick="selectParams('term_growth','visuals',this)"><div class="fn-card-icon">📈</div><div class="fn-card-name">Term Growth</div><div class="fn-card-desc">Yearly growth for keywords, title terms, or abstract terms</div></div>
          <div class="fn-card" onclick="selectParams('sankey','visuals',this)"><div class="fn-card-icon">🔀</div><div class="fn-card-name">Sankey Diagram</div><div class="fn-card-desc">Flow between authors, countries, keywords</div></div>
          <div class="fn-card" onclick="selectParams('productivity','visuals',this)"><div class="fn-card-icon">📅</div><div class="fn-card-name">Productivity</div><div class="fn-card-desc">Yearly output per author/country/journal</div></div>
          <div class="fn-card" onclick="selectParams('projection','visuals',this)"><div class="fn-card-icon">🔵</div><div class="fn-card-name">Doc Projection</div><div class="fn-card-desc">2D document clustering by TF-IDF or embeddings</div></div>
          <div class="fn-card" onclick="selectParams('countyx','visuals',this)"><div class="fn-card-icon">🧱</div><div class="fn-card-name">Count Y per X</div><div class="fn-card-desc">Stacked counts between any two entity types</div></div>
          <div class="fn-card" onclick="selectParams('heatmapxy','visuals',this)"><div class="fn-card-icon">🔥</div><div class="fn-card-name">Heatmap Y per X</div><div class="fn-card-desc">Entity heatmap between keywords, authors, countries</div></div>
        </div>
        <div class="fn-hint">Click any function card to configure its parameters →</div>
      </div>

      <!-- ── NETWORK ── -->
      <div class="view" id="view-network">
        <div class="view-title">Network Analysis</div>
        <div class="view-desc">Co-authorship, keyword co-occurrence, similarity, and historiograph networks.</div>
        <div class="fn-grid">
          <div class="fn-card" onclick="selectParams('adj','network',this)"><div class="fn-card-icon">👥</div><div class="fn-card-name">Collaboration / Adjacency</div><div class="fn-card-desc">Co-authorship, co-keywords, co-countries</div></div>
          <div class="fn-card" onclick="selectParams('map','network',this)"><div class="fn-card-icon">🗺</div><div class="fn-card-name">World Map</div><div class="fn-card-desc">Country collaboration on a geographic map</div></div>
          <div class="fn-card" onclick="selectParams('sim','network',this)"><div class="fn-card-icon">🔗</div><div class="fn-card-name">Similarity Network</div><div class="fn-card-desc">Bibliographic coupling or co-citation</div></div>
          <div class="fn-card" onclick="selectParams('hist','network',this)"><div class="fn-card-icon">🕰</div><div class="fn-card-name">Historiograph</div><div class="fn-card-desc">Direct citation chain over time</div></div>

          <div class="fn-card" onclick="selectParams('mainpath','network',this)"><div class="fn-card-icon">🛤️</div><div class="fn-card-name">Main Path Analysis</div><div class="fn-card-desc">Core knowledge-flow path in the citation network</div></div>
          <div class="fn-card" onclick="selectParams('adjdir','network',this)"><div class="fn-card-icon">🧭</div><div class="fn-card-name">Directed Citation Network</div><div class="fn-card-desc">Local vs cited references in a directed citation graph</div></div>
          <div class="fn-card" onclick="selectParams('finddir','network',this)"><div class="fn-card-icon">🎯</div><div class="fn-card-name">Highlight Citation Nodes</div><div class="fn-card-desc">Focus on specific article IDs or reference IDs</div></div>
          <div class="fn-card" onclick="selectParams('salsa','network',this)"><div class="fn-card-icon">💃</div><div class="fn-card-name">SALSA</div><div class="fn-card-desc">Hub and authority analysis across decades</div></div>
        </div>
        <div class="fn-hint">Click any function card to configure its parameters →</div>
      </div>

      <!-- ── TEMPORAL SG ── -->
      <div class="view" id="view-temporal">
        <div class="view-title">Temporal Scholarly Graph</div>
        <div class="view-desc">Paper-centred temporal explorer with citations, authors, keywords, references, and multi-lens navigation.</div>
        <div class="fn-grid">
          <div class="fn-card" onclick="selectParams('temporal_sg','temporal',this)"><div class="fn-card-icon">🧭</div><div class="fn-card-name">TSG Builder</div><div class="fn-card-desc">Open the dedicated temporal graph workspace in its own tab</div></div>
        </div>
        <div class="fn-hint">Configure the graph parameters →</div>
      </div>

      <!-- ── REFERENCES ── -->
      <div class="view" id="view-references">
        <div class="view-title">Reference Analysis</div>
        <div class="view-desc">Explore citation patterns, RPYS, and reference trajectories.</div>
        <div class="fn-grid">
          <div class="fn-card" onclick="selectParams('top_refs','references',this)"><div class="fn-card-icon">🏆</div><div class="fn-card-name">Top References</div><div class="fn-card-desc">Most cited references in the dataset</div></div>
          <div class="fn-card" onclick="selectParams('rpys','references',this)"><div class="fn-card-icon">📡</div><div class="fn-card-name">RPYS</div><div class="fn-card-desc">Reference publication year spectroscopy — spot paradigm shifts</div></div>
          <div class="fn-card" onclick="selectParams('trajectory','references',this)"><div class="fn-card-icon">🚀</div><div class="fn-card-name">Citation Trajectory</div><div class="fn-card-desc">Yearly citations for specific references</div></div>
          <div class="fn-card" onclick="selectParams('refmatrix','references',this)"><div class="fn-card-icon">🧮</div><div class="fn-card-name">Reference Citation Matrix</div><div class="fn-card-desc">See which citing articles reference which target references</div></div>
          <div class="fn-card" onclick="selectParams('corefs','references',this)"><div class="fn-card-icon">🧷</div><div class="fn-card-name">Co-References</div><div class="fn-card-desc">Most frequent co-cited reference groups</div></div>
          <div class="fn-card" onclick="selectParams('cocitnet','references',this)"><div class="fn-card-icon">🕸</div><div class="fn-card-name">Co-Citation Network</div><div class="fn-card-desc">Network around a focal reference</div></div>
          <div class="fn-card" onclick="selectParams('sleep','references',this)"><div class="fn-card-icon">😴</div><div class="fn-card-name">Sleeping Beauties</div><div class="fn-card-desc">Late-blooming references with delayed recognition</div></div>
          <div class="fn-card" onclick="selectParams('princes','references',this)"><div class="fn-card-icon">👑</div><div class="fn-card-name">Princes</div><div class="fn-card-desc">Likely awakening papers associated with sleeping beauties</div></div>
          <div class="fn-card" onclick="selectParams('refdiv','references',this)"><div class="fn-card-icon">📚</div><div class="fn-card-name">Reference Diversity</div><div class="fn-card-desc">Breadth and age profile of each paper's references</div></div>
          <div class="fn-card" onclick="selectParams('disruption','references',this)"><div class="fn-card-icon">⚡</div><div class="fn-card-name">Disruption Index</div><div class="fn-card-desc">Whether papers disrupt or consolidate prior work</div></div>
        </div>
        <div class="fn-hint">Click any function card to configure its parameters →</div>
      </div>

      <!-- ── PROFILING ── -->
      <div class="view" id="view-profiling">
        <div class="view-title">Entity Profiling</div>
        <div class="view-desc">Detailed statistics for any individual author, journal, country, institution, keyword, or reference, plus author-level bibliometric indices.</div>
        <div class="fn-grid">
          <div class="fn-card active" onclick="selectParams('profiling','profiling',this)">
            <div class="fn-card-icon">🔍</div><div class="fn-card-name">Profile Entity</div>
            <div class="fn-card-desc">Author · Affiliation · Country · Journal · Keyword · Reference</div>
          </div>
          <div class="fn-card" onclick="selectParams('metrics','profiling',this)">
            <div class="fn-card-icon">📐</div><div class="fn-card-name">Compute Indices</div>
            <div class="fn-card-desc">H · G · E · J · M for every author in the corpus</div>
          </div>
        </div>
        <div class="fn-hint">Click a card to configure parameters →</div>
      </div>

      <!-- ── AI ── -->
      <div class="view" id="view-ai">
        <div class="view-title">AI &amp; Tools</div>
        <div class="view-desc">Topic modelling, word-level analysis, and concise topic summaries split into focused work areas.</div>
        <div class="callout warn" style="margin-bottom:16px;">⚡ Using <code>bertopic</code>, <code>sentence-transformers</code>, and <code>gensim</code>. </div>

        <div class="ai-group active" id="ai-group-topics" data-ai-group="topics">
          <div class="ai-group-title">AI - Topics</div>
          <div class="ai-group-desc">Build the BERTopic model, inspect topic structure, and explore topic-level charts.</div>
          <div class="fn-grid">
            <div class="fn-card" onclick="selectParams('ai-create','ai',this)"><div class="fn-card-icon">🧠</div><div class="fn-card-name">Create Topics</div><div class="fn-card-desc">Build BERTopic model from abstracts</div></div>
            <div class="fn-card" data-topic-required="1" onclick="selectAIGraph('graph_topics',this)"><div class="fn-card-icon">📊</div><div class="fn-card-name">Topic Words</div><div class="fn-card-desc">Top words per topic (bar chart)</div><div class="fn-card-meta">Requires Create Topics</div></div>
            <div class="fn-card" data-topic-required="1" onclick="selectAIGraph('distribution',this)"><div class="fn-card-icon">🔢</div><div class="fn-card-name">Distribution</div><div class="fn-card-desc">Document count per topic</div><div class="fn-card-meta">Requires Create Topics</div></div>
            <div class="fn-card" data-topic-required="1" onclick="selectAIGraph('projection',this)"><div class="fn-card-icon">🔵</div><div class="fn-card-name">Topic Projection</div><div class="fn-card-desc">2D scatter by topic cluster</div><div class="fn-card-meta">Requires Create Topics</div></div>
            <div class="fn-card" data-topic-required="1" onclick="selectAIGraph('heatmap',this)"><div class="fn-card-icon">🌡</div><div class="fn-card-name">Topic Heatmap</div><div class="fn-card-desc">Topic–keyword co-occurrence</div><div class="fn-card-meta">Requires Create Topics</div></div>
            <div class="fn-card" data-topic-required="1" onclick="selectAIGraph('time',this)"><div class="fn-card-icon">⏳</div><div class="fn-card-name">Topics over Time</div><div class="fn-card-desc">Topic frequency evolution</div><div class="fn-card-meta">Requires Create Topics</div></div>
            <div class="fn-card" data-topic-required="1" onclick="selectParams('ai-authors','ai',this)"><div class="fn-card-icon">👩‍🔬</div><div class="fn-card-name">Authors per Topic</div><div class="fn-card-desc">Top authors contributing to each topic</div><div class="fn-card-meta">Requires Create Topics</div></div>
            <div class="fn-card" data-topic-required="1" onclick="selectParams('ai-representatives','ai',this)"><div class="fn-card-icon">📄</div><div class="fn-card-name">Representative Docs</div><div class="fn-card-desc">Most representative documents for each topic</div><div class="fn-card-meta">Requires Create Topics</div></div>
            <div class="fn-card" data-topic-required="1" onclick="selectParams('ai-doc-topic','ai',this)"><div class="fn-card-icon">📝</div><div class="fn-card-name">Doc - Topic Alignment</div><div class="fn-card-desc">Topic alignment for words in one document</div><div class="fn-card-meta">Requires Create Topics</div></div>
          </div>
          <div class="fn-hint">Run “Create Topics” first to unlock the topic-dependent cards.</div>
        </div>

        <div class="ai-group" id="ai-group-words" data-ai-group="words">
          <div class="ai-group-title">AI - Words</div>
          <div class="ai-group-desc">Build a local word-embedding model and explore semantic similarity, analogies, and 2D plots.</div>
          <div class="fn-grid">
            <div class="fn-card" onclick="selectParams('ai-word-embeddings','ai',this)"><div class="fn-card-icon">🧬</div><div class="fn-card-name">Word Embeddings</div><div class="fn-card-desc">Train a FastText model from abstracts</div></div>
            <div class="fn-card" data-word-emb-required="1" onclick="selectParams('ai-word-sim','ai',this)"><div class="fn-card-icon">📏</div><div class="fn-card-name">Word Similarity</div><div class="fn-card-desc">Cosine similarity between two words</div><div class="fn-card-meta">Requires Word Embeddings</div></div>
            <div class="fn-card" data-word-emb-required="1" onclick="selectParams('ai-word-ops','ai',this)"><div class="fn-card-icon">⚙️</div><div class="fn-card-name">Word Operations</div><div class="fn-card-desc">Vector arithmetic with positive and negative words</div><div class="fn-card-meta">Requires Word Embeddings</div></div>
            <div class="fn-card" data-word-emb-required="1" onclick="selectParams('ai-word-plot','ai',this)"><div class="fn-card-icon">🗺️</div><div class="fn-card-name">Plot Word Embeddings</div><div class="fn-card-desc">Project related words in 2D</div><div class="fn-card-meta">Requires Word Embeddings</div></div>
          </div>
          <div class="fn-hint">Run “Word Embeddings” first to unlock the embedding-dependent cards.</div>
        </div>

        <div class="ai-group" id="ai-group-summarize" data-ai-group="summarize">
          <div class="ai-group-title">AI - Summarize</div>
          <div class="ai-group-desc">Generate extractive or abstractive summaries from the current abstracts.</div>
          <div class="fn-grid">
            <div class="fn-card" onclick="selectParams('ai-summarize','ai',this)"><div class="fn-card-icon">✍️</div><div class="fn-card-name">Summarize Abstracts</div><div class="fn-card-desc">Switch between extractive and abstractive models</div></div>
          </div>
          <div class="fn-hint">The model field auto-fills a recommended default for each method, but you can overwrite it manually.</div>
        </div>
      </div>

    </div><!-- /tab-functions -->

    <!-- ════ TAB: PARAMETERS ════ -->
    <div id="tab-parameters" class="tab-pane">

      <div id="params-empty">
        <div class="pe-icon">⚙️</div>
        <p>Select a function from the <strong>Functions</strong> tab<br>to configure its parameters here.</p>
      </div>

      <!-- FILTER -->
      <div class="params-section" id="params-filter">
        <div class="params-box">
          <div class="params-title">Filter Dataset Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Documents (comma-sep positions)</label><input type="text" id="f-docs" placeholder="e.g. 0, 5, 12"/></div>
            <div class="param-item"><label class="fl">Document types</label><input type="text" id="f-dtype" placeholder="e.g. Article, Review"/></div>
            <div class="param-item"><label class="fl">Year start</label><input type="number" id="f-ystr" placeholder="optional"/></div>
            <div class="param-item"><label class="fl">Year end</label><input type="number" id="f-yend" placeholder="optional"/></div>
            <div class="param-item"><label class="fl">Sources</label><input type="text" id="f-sources" placeholder="comma-separated"/></div>
            <div class="param-item"><label class="fl">Bradford core</label><input type="number" id="f-core" placeholder="optional"/></div>
            <div class="param-item"><label class="fl">Countries</label><input type="text" id="f-country" placeholder="comma-separated"/></div>
            <div class="param-item"><label class="fl">Languages</label><input type="text" id="f-lang" placeholder="comma-separated"/></div>
          </div>
          <div class="checkbox-row">
            <input type="checkbox" id="f-abs"/>
            <span>Only documents with abstracts</span>
          </div>
          <div class="callout info help-note">All filter arguments from the notebook are exposed here: documents, doc_type, year range, sources, Bradford core, country, language, and abstract. Use exact names from the loaded dataset tables when possible.</div>
          <button class="run-btn" onclick="runFilter(this)">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run
          </button>
        </div>
      </div>

      <!-- WORD CLOUD -->
      <div class="params-section" id="params-wc">
        <div class="params-box">
          <div class="params-title">Word Cloud Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Entry</label><select id="wc-entry"><option value="kwp">Keywords Plus</option><option value="kwa">Authors Keywords</option><option value="abs">Abstracts</option><option value="title">Titles</option></select></div>
            <div class="param-item"><label class="fl">Max words</label><input type="number" id="wc-words" value="300"/></div>
            <div class="param-item"><label class="fl">Remove words (comma-sep)</label><input type="text" id="wc-rmv" placeholder="study, paper, analysis"/></div>
          </div>
          <button class="run-btn" onclick="runWordcloud(this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Generate</button>
        </div>
      </div>

      <!-- NGRAMS -->
      <div class="params-section" id="params-ngrams">
        <div class="params-box">
          <div class="params-title">N-Gram Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Entry</label><select id="ng-entry"><option value="kwp">Keywords Plus</option><option value="kwa">Authors Keywords</option><option value="abs">Abstracts</option><option value="title">Titles</option></select></div>
            <div class="param-item"><label class="fl">N-gram size</label><select id="ng-n"><option value="1">1 — Unigrams</option><option value="2">2 — Bigrams</option><option value="3">3 — Trigrams</option></select></div>
            <div class="param-item"><label class="fl">Top N words</label><input type="number" id="ng-topn" value="20"/></div>
            <div class="param-item"><label class="fl">Language stopwords</label><select id="ng-lang"><option value="ar">Arabic</option><option value="bn">Bengali</option><option value="bg">Bulgarian</option><option value="zh">Chinese</option><option value="cs">Czech</option><option value="en">English</option><option value="fi">Finnish</option><option value="fr">French</option><option value="de">German</option><option value="el">Greek</option><option value="he">Hebrew</option><option value="hi">Hindi</option><option value="hu">Hungarian</option><option value="it">Italian</option><option value="ja">Japanese</option><option value="ko">Korean</option><option value="mr">Marathi</option><option value="fa">Persian</option><option value="pl">Polish</option><option value="pt-br">Portuguese (Brazil)</option><option value="ro">Romanian</option><option value="ru">Russian</option><option value="sk">Slovak</option><option value="es">Spanish</option><option value="sv">Swedish</option><option value="th">Thai</option><option value="uk">Ukrainian</option></select></div>
            <div class="param-item"><label class="fl">Remove words (comma-sep)</label><input type="text" id="ng-rmv" placeholder="study, paper"/></div>
          </div>
          <button class="run-btn" onclick="runNgrams(this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- TREEMAP -->
      <div class="params-section" id="params-treemap">
        <div class="params-box">
          <div class="params-title">TreeMap Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Entry</label><select id="tm-entry"><option value="kwp">Keywords Plus</option><option value="kwa">Authors Keywords</option><option value="aut">Authors</option><option value="ctr">Countries</option><option value="inst">Institutions</option><option value="jou">Journals</option></select></div>
            <div class="param-item"><label class="fl">Top N</label><input type="number" id="tm-topn" value="25"/></div>
          </div>
          <button class="run-btn" onclick="runTreemap(this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- BARS -->
      <div class="params-section" id="params-bars">
        <div class="params-box">
          <div class="params-title">Bar Chart Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Statistic</label>
              <select id="bars-stat">
                <option value="dpy">Documents per Year</option>
                <option value="cpy">Citations per Year</option>
                <option value="ppy">Past Citations per Year</option>
                <option value="ltk">Lotka's Law</option>
                <option value="bdf_1">Bradford's Law — Core 1</option>
                <option value="bdf_2">Bradford's Law — Core 2</option>
                <option value="bdf_3">Bradford's Law — Core 3</option>
                <option value="spd">Sources per Documents</option>
                <option value="spc">Sources per Citations</option>
                <option value="apd">Authors per Documents</option>
                <option value="apc">Authors per Citations</option>
                <option value="aph">Authors by H-Index</option>
                <option value="apj">Authors by J-Index</option>
                <option value="ipd">Institutions per Documents</option>
                <option value="ipc">Institutions per Citations</option>
                <option value="cpd">Countries per Documents</option>
                <option value="cpc">Countries per Citations</option>
                <option value="lpd">Languages per Documents</option>
                <option value="kpd">Keywords Plus per Documents</option>
                <option value="kad">Authors' Keywords per Documents</option>
              </select>
            </div>
            <div class="param-item"><label class="fl">Top N</label><input type="number" id="bars-topn" value="20"/></div>
          </div>
          <button class="run-btn" onclick="runBars(this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- EVOLUTION -->
      <!--
      <div class="params-section" id="params-evolution">
        <div class="params-box">
          <div class="params-title">Evolution Plot Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Key</label><select id="ev-key"><option value="kwp">Keywords Plus</option><option value="kwa">Authors Keywords</option><option value="abs">Abstracts</option><option value="title">Titles</option><option value="jou">Sources</option></select></div>
            <div class="param-item"><label class="fl">Top N topics</label><input type="number" id="ev-topn" value="10"/></div>
            <div class="param-item"><label class="fl">Year start</label><input type="number" id="ev-start" value="2010"/></div>
            <div class="param-item"><label class="fl">Year end</label><input type="number" id="ev-end" value="2025"/></div>
            <div class="param-item"><label class="fl">Stopword language</label><select id="ev-lang"><option value="ar">Arabic</option><option value="bn">Bengali</option><option value="bg">Bulgarian</option><option value="zh">Chinese</option><option value="cs">Czech</option><option value="en">English</option><option value="fi">Finnish</option><option value="fr">French</option><option value="de">German</option><option value="el">Greek</option><option value="he">Hebrew</option><option value="hi">Hindi</option><option value="hu">Hungarian</option><option value="it">Italian</option><option value="ja">Japanese</option><option value="ko">Korean</option><option value="mr">Marathi</option><option value="fa">Persian</option><option value="pl">Polish</option><option value="pt-br">Portuguese (Brazil)</option><option value="ro">Romanian</option><option value="ru">Russian</option><option value="sk">Slovak</option><option value="es">Spanish</option><option value="sv">Swedish</option><option value="th">Thai</option><option value="uk">Ukrainian</option></select></div>
          </div>
          <div class="callout info" style="margin-bottom:10px;">Adjust the year range to match your dataset's publication span to avoid errors.</div>
          <button class="run-btn" onclick="runEvolution(this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>
      -->
      <!-- TERM GROWTH -->
      <div class="params-section" id="params-term_growth">
        <div class="params-box">
          <div class="params-title">Term Growth Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Source</label><select id="tg-source"><option value="kwa" selected>Authors Keywords</option><option value="kwp">Keywords Plus</option><option value="title">Title Terms</option><option value="abs">Abstract Terms</option></select></div>
            <div class="param-item"><label class="fl">Top N</label><input type="number" id="tg-topn" value="10"/></div>
            <div class="param-item"><label class="fl">Year start</label><input type="number" id="tg-start" value="-1"/></div>
            <div class="param-item"><label class="fl">Year end</label><input type="number" id="tg-end" value="-1"/></div>
            <div class="param-item"><label class="fl">Stopword language</label><select id="tg-lang"><option value="ar">Arabic</option><option value="bn">Bengali</option><option value="bg">Bulgarian</option><option value="zh">Chinese</option><option value="cs">Czech</option><option value="en" selected>English</option><option value="fi">Finnish</option><option value="fr">French</option><option value="de">German</option><option value="el">Greek</option><option value="he">Hebrew</option><option value="hi">Hindi</option><option value="hu">Hungarian</option><option value="it">Italian</option><option value="ja">Japanese</option><option value="ko">Korean</option><option value="mr">Marathi</option><option value="fa">Persian</option><option value="pl">Polish</option><option value="pt-br">Portuguese (Brazil)</option><option value="ro">Romanian</option><option value="ru">Russian</option><option value="sk">Slovak</option><option value="es">Spanish</option><option value="sv">Swedish</option><option value="th">Thai</option><option value="uk">Ukrainian</option></select></div>
            <div class="param-item"><label class="fl">Remove words (comma-sep)</label><input type="text" id="tg-rmv" placeholder="study, paper, analysis"/></div>
          </div>
          <div class="checkbox-row"><input type="checkbox" id="tg-cumulative" checked/><span>Cumulative</span></div>
          <div class="checkbox-row"><input type="checkbox" id="tg-line" checked/><span>Line plot</span></div>
          <div class="checkbox-row"><input type="checkbox" id="tg-bubble" checked/><span>Bubble timeline</span></div>
          <button class="run-btn" onclick="runTermGrowth(this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>


<!-- SANKEY -->
<div class="params-section" id="params-sankey">
  <div class="params-box">
    <div class="params-title">Sankey Diagram Parameters</div>
    <div id="sankey-builder" class="sankey-builder"></div>
    <button type="button" class="sankey-add" onclick="addSankeyLayer()">+ add another layer</button>
    <div class="callout info help-note">Define the top N items flow between the previous layer and the next one.</div>
    <div class="checkbox-row"><input type="checkbox" id="sk-rmv" checked/><span>Remove unknown entries</span></div>
    <button class="run-btn" onclick="runSankey(this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
  </div>
</div>

<!-- PRODUCTIVITY -->
      <div class="params-section" id="params-productivity">
        <div class="params-box">
          <div class="params-title">Productivity Plot Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Entity type</label><select id="pr-kind"><option value="authors">Authors</option><option value="countries">Countries</option><option value="institution">Institutions</option><option value="source">Journals/Sources</option></select></div>
            <div class="param-item"><label class="fl">Top N</label><input type="number" id="pr-topn" value="20"/></div>
          </div>
          <button class="run-btn" onclick="runProductivity(this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- PROJECTION -->
      <div class="params-section" id="params-projection">
        <div class="params-box">
          <div class="params-title">Document Projection Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Corpus</label><select id="pj-corpus"><option value="abs">Abstracts</option><option value="title">Titles</option><option value="kwa">Authors Keywords</option><option value="kwp">Keywords Plus</option></select></div>
            <div class="param-item"><label class="fl">Stop-word language</label><select id="pj-lang"><option value="ar">Arabic</option><option value="bn">Bengali</option><option value="bg">Bulgarian</option><option value="zh">Chinese</option><option value="cs">Czech</option><option value="en">English</option><option value="fi">Finnish</option><option value="fr">French</option><option value="de">German</option><option value="el">Greek</option><option value="he">Hebrew</option><option value="hi">Hindi</option><option value="hu">Hungarian</option><option value="it">Italian</option><option value="ja">Japanese</option><option value="ko">Korean</option><option value="mr">Marathi</option><option value="fa">Persian</option><option value="pl">Polish</option><option value="pt-br">Portuguese (Brazil)</option><option value="ro">Romanian</option><option value="ru">Russian</option><option value="sk">Slovak</option><option value="es">Spanish</option><option value="sv">Swedish</option><option value="th">Thai</option><option value="uk">Ukrainian</option></select></div>
            <div class="param-item"><label class="fl">Remove words (comma-sep)</label><input type="text" id="pj-rmv" placeholder="optional"/></div>
            <div class="param-item"><label class="fl">Dimensions</label><input type="number" id="pj-components" value="2" min="2"/></div>
            <div class="param-item"><label class="fl">N clusters</label><input type="number" id="pj-clusters" value="5"/></div>
            <div class="param-item"><label class="fl">Reduction method</label><select id="pj-method"><option value="tsvd">Truncated SVD (fast)</option><option value="umap">UMAP</option><option value="tsne">t-SNE</option></select></div>
            <div class="param-item"><label class="fl">Cluster method</label><select id="pj-cm"><option value="kmeans">K-Means</option><option value="hdbscan">HDBSCAN</option></select></div>
            <div class="param-item"><label class="fl">Node size</label><input type="number" id="pj-size" value="20" min="1"/></div>
            <div class="param-item"><label class="fl">Node font size</label><input type="number" id="pj-font" value="8" min="1"/></div>
            <div class="param-item"><label class="fl">Embedding model</label><input type="text" id="pj-model" value="allenai/scibert_scivocab_uncased"/></div>
            <div class="param-item"><label class="fl">HDBSCAN min size</label><input type="number" id="pj-minsize" value="5" min="2"/></div>
            <div class="param-item"><label class="fl">HDBSCAN max size</label><input type="number" id="pj-maxsize" value="50" min="2"/></div>
          </div>
          <div class="checkbox-row"><input type="checkbox" id="pj-labels" checked/><span>Show node labels</span></div>
          <div class="checkbox-row"><input type="checkbox" id="pj-tfidf"/><span>Use TF-IDF for clustering when embeddings are off</span></div>
          <div class="checkbox-row"><input type="checkbox" id="pj-legend" checked/><span>Show legend</span></div>
          <div class="checkbox-row"><input type="checkbox" id="pj-emb"/><span>Use SciBERT embeddings (slower, more accurate)</span></div>
          <button class="run-btn" onclick="runProjection(this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- NETWORK: ADJ -->
      <div class="params-section" id="params-adj">
        <div class="params-box">
          <div class="params-title">Collaboration / Adjacency Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Type</label><select id="adj-type"><option value="aut">Authors</option><option value="cout">Countries</option><option value="inst">Institutions</option><option value="kwa">Authors Keywords</option><option value="kwp">Keywords Plus</option></select></div>
            <div class="param-item"><label class="fl">Min co-occurrences</label><input type="number" id="adj-min" value="2"/></div>
            <div class="param-item"><label class="fl">Centrality coloring</label><select id="adj-centrality"><option value="">None</option><option value="degree">Degree</option><option value="load">Load</option><option value="betw">Betweenness</option><option value="close">Closeness</option><option value="eigen">Eigenvector</option><option value="katz">Katz</option><option value="harmonic">Harmonic</option><option value="pagerank">PageRank</option></select></div>
          </div>
          <div class="params-grid" style="margin-top:12px;"><div class="param-item"><label class="fl">Label type</label><select id="adj-labeltype"><option value="id">ID</option><option value="name">Name</option></select></div><div class="param-item"><label class="fl">Node size</label><input type="number" id="adj-size" value="-1"/></div></div>
          <div class="checkbox-row"><input type="checkbox" id="adj-labels"/><span>Show node labels</span></div>
          <button class="run-btn" onclick="runNetwork('adj',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- NETWORK: MAP -->
      <div class="params-section" id="params-map">
        <div class="params-box">
          <div class="params-title">World Map Parameters</div>
          <div class="checkbox-row"><input type="checkbox" id="map-conn" checked/><span>Show collaboration connections</span></div>
          <div class="param-item" style="margin:12px 0 14px;"><label class="fl">Highlight countries (comma-sep)</label><input type="text" id="map-countries" placeholder="optional"/></div>
          <button class="run-btn" onclick="runNetwork('map',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- NETWORK: SIM -->
      <div class="params-section" id="params-sim">
        <div class="params-box">
          <div class="params-title">Similarity Network Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Type</label><select id="sim-type"><option value="coup">Bibliographic Coupling</option><option value="cocit">Co-Citation</option></select></div>
            <div class="param-item"><label class="fl">Coupling threshold (0–1)</label><input type="number" id="sim-coup" value="0.3" step="0.05" min="0" max="1"/></div>
            <div class="param-item"><label class="fl">Co-citation min count</label><input type="number" id="sim-cocit" value="5" min="1"/></div>
          </div>
          <div class="params-grid" style="margin-top:12px;"><div class="param-item"><label class="fl">Node size</label><input type="number" id="sim-size" value="-1"/></div></div>
          <div class="checkbox-row"><input type="checkbox" id="sim-labels"/><span>Show node labels</span></div>
          <button class="run-btn" onclick="runNetwork('sim',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- NETWORK: HIST -->
      <div class="params-section" id="params-hist">
        <div class="params-box">
          <div class="params-title">Historiograph Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Min links</label><input type="number" id="hist-min" value="1" min="1"/></div>
            <div class="param-item"><label class="fl">Chain document IDs</label><input type="text" id="hist-chain" placeholder="optional, e.g. 3,8,56"/></div>
            <div class="param-item"><label class="fl">Node size</label><input type="number" id="hist-size" value="20" min="1"/></div>
            <div class="param-item"><label class="fl">Font size</label><input type="number" id="hist-font" value="10" min="1"/></div>
            <div class="param-item"><label class="fl">Distance</label><input type="number" id="hist-dist" value="0.7" step="0.1"/></div>
            <div class="param-item"><label class="fl">Distance pad</label><input type="number" id="hist-distpad" value="1.0" step="0.1"/></div>
          </div>
          <div class="checkbox-row"><input type="checkbox" id="hist-labels" checked/><span>Show node labels</span></div>
          <div class="checkbox-row"><input type="checkbox" id="hist-path"/><span>Show only the specified path/chain</span></div>
          <button class="run-btn" onclick="runNetwork('hist',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- REFS: TOP -->
      <div class="params-section" id="params-temporal_sg">
        <div class="params-box">
          <div class="params-title">Temporal Scholarly Graph Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Initial view</label><select id="tsg-view"><option value="timeline">Timeline</option><option value="force">Force</option><option value="ego">Ego</option></select></div>
            <div class="param-item"><label class="fl">Graph center</label><select id="tsg-center"><option value="paper">Paper</option><option value="journal">Journal</option><option value="author">Author</option><option value="institution">Institution</option><option value="country">Country</option><option value="reference">Reference</option><option value="author_keyword">Authors' keyword</option><option value="keyword_plus">Keyword Plus</option></select></div>
            <div class="param-item"><label class="fl">Max papers</label><input id="tsg-maxp" type="number" value="500" min="50" max="5000"/></div>
            <div class="param-item"><label class="fl">Max references</label><input id="tsg-maxr" type="number" value="300" min="20" max="3000"/></div>
            <div class="param-item"><label class="fl">Start year</label><input id="tsg-start" type="number" placeholder="auto"/></div>
            <div class="param-item"><label class="fl">End year</label><input id="tsg-end" type="number" placeholder="auto"/></div>
          </div>
          <button class="run-btn" onclick="runTemporalSG(this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M2 12h20"/><circle cx="12" cy="12" r="4"/></svg>Open Temporal Scholarly Graph</button>
        </div>
      </div>

      <div class="params-section" id="params-top_refs">
        <div class="params-box">
          <div class="params-title">Top References Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Top N references</label><input type="number" id="tr-topn" value="15" min="1"/></div>
            <div class="param-item"><label class="fl">Legend font size</label><input type="number" id="tr-font" value="10" min="1"/></div>
            <div class="param-item"><label class="fl">Date start</label><input type="number" id="tr-start" placeholder="optional"/></div>
            <div class="param-item"><label class="fl">Date end</label><input type="number" id="tr-end" placeholder="optional"/></div>
          </div>
          <div class="checkbox-row"><input type="checkbox" id="tr-useid"/><span>Use reference IDs instead of names</span></div>
          <button class="run-btn" onclick="runRefs('top_refs',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- REFS: RPYS -->
      <div class="params-section" id="params-rpys">
        <div class="params-box">
          <div class="params-title">RPYS Parameters</div>
          <div class="checkbox-row"><input type="checkbox" id="rpys-peaks"/><span>Show peaks only</span></div>
          <button class="run-btn" onclick="runRefs('rpys',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- REFS: TRAJECTORY -->
      <div class="params-section" id="params-trajectory">
        <div class="params-box">
          <div class="params-title">Citation Trajectory Parameters</div>
          <div class="param-item" style="margin-bottom:14px;"><label class="fl">Reference IDs (comma-sep, e.g. r_1,r_5)</label><input type="text" id="traj-ids" placeholder="r_1, r_5"/></div>
          <div class="param-item" style="margin-bottom:14px;"><label class="fl">Reference names (comma-sep)</label><input type="text" id="traj-names" placeholder="optional"/></div>
          <div class="callout info" style="margin-bottom:10px;">Use IDs from the Objects & IDs tab, or exact reference names.</div>
          <button class="run-btn" onclick="runRefs('trajectory',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- PROFILING -->
      <div class="params-section" id="params-profiling">
        <div class="params-box">
          <div class="params-title">Profiling Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Entity type</label><select id="prof-kind"><option value="author">Author</option><option value="affiliation">Affiliation</option><option value="country">Country</option><option value="journal">Journal</option><option value="keyword">Keyword (Authors)</option><option value="kwp">Keyword Plus (WoS)</option><option value="reference">Reference</option></select></div>
            <div class="param-item"><label class="fl">Name (or leave blank for ID)</label><input type="text" id="prof-name" placeholder="e.g. Smith J"/></div>
            <div class="param-item"><label class="fl">Entity ID (e.g. a_1)</label><input type="text" id="prof-id" placeholder="a_1"/></div>
            <div class="param-item"><label class="fl">Top N related</label><input type="number" id="prof-topn" value="5" min="1"/></div>
          </div>
          <div class="callout info" style="margin-bottom:10px;">Provide either a Name or an ID.</div>
          <button class="run-btn" onclick="runProfiling(this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- METRICS -->
      <div class="params-section" id="params-metrics">
        <div class="params-box">
          <div class="params-title">Bibliometric Index Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Current year (for M-index)</label><input type="number" id="mi-year" value="2026"/></div>
            <div class="param-item"><label class="fl">Included indices</label><input type="text" value="H, G, E, J, M" disabled/></div>
          </div>
          <button class="run-btn" onclick="runMetrics(this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Compute All Indices</button>
        </div>
      </div>

      <!-- AI: CREATE TOPICS -->
      <div class="params-section" id="params-ai-create">
        <div class="params-box">
          <div class="params-title">BERTopic Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Stop-word language</label><select id="ai-lang"><option value="ar">Arabic</option><option value="bn">Bengali</option><option value="bg">Bulgarian</option><option value="zh">Chinese</option><option value="cs">Czech</option><option value="en">English</option><option value="fi">Finnish</option><option value="fr">French</option><option value="de">German</option><option value="el">Greek</option><option value="he">Hebrew</option><option value="hi">Hindi</option><option value="hu">Hungarian</option><option value="it">Italian</option><option value="ja">Japanese</option><option value="ko">Korean</option><option value="mr">Marathi</option><option value="fa">Persian</option><option value="pl">Polish</option><option value="pt-br">Portuguese (Brazil)</option><option value="ro">Romanian</option><option value="ru">Russian</option><option value="sk">Slovak</option><option value="es">Spanish</option><option value="sv">Swedish</option><option value="th">Thai</option><option value="uk">Ukrainian</option></select></div>
          </div>
          <div class="checkbox-row"><input type="checkbox" id="ai-emb"/><span>Use SciBERT embeddings (slower, more accurate)</span></div>
          <button class="run-btn" onclick="runTopics(this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Build Model</button>
        </div>
      </div>

      <!-- CROSS: COUNT Y PER X -->
      <div class="params-section" id="params-countyx">
        <div class="params-box">
          <div class="params-title">Count Y per X Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">X axis</label><select id="cxy-x"><option value="cout">Countries</option><option value="aut">Authors</option><option value="inst">Institutions</option><option value="jou">Sources</option><option value="kwa">Authors Keywords</option><option value="kwp">Keywords Plus</option><option value="lan">Language</option></select></div>
            <div class="param-item"><label class="fl">Y axis</label><select id="cxy-y"><option value="aut">Authors</option><option value="cout">Countries</option><option value="inst">Institutions</option><option value="jou">Sources</option><option value="kwa">Authors Keywords</option><option value="kwp">Keywords Plus</option><option value="lan">Language</option></select></div>
            <div class="param-item"><label class="fl">Top N X</label><input type="number" id="cxy-topx" value="5" min="1"/></div>
            <div class="param-item"><label class="fl">Top N Y</label><input type="number" id="cxy-topy" value="5" min="1"/></div>
            <div class="param-item"><label class="fl">Text font size</label><input type="number" id="cxy-font" value="12" min="8"/></div>
            <div class="param-item"><label class="fl">X label angle</label><input type="number" id="cxy-angle" value="-90"/></div>
          </div>
          <div class="checkbox-row"><input type="checkbox" id="cxy-rmv" checked/><span>Remove unknowns</span></div>
          <button class="run-btn" onclick="runCross('count_y_x',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- CROSS: HEATMAP Y PER X -->
      <div class="params-section" id="params-heatmapxy">
        <div class="params-box">
          <div class="params-title">Heatmap Y per X Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">X axis</label><select id="hxy-x"><option value="kwa">Authors Keywords</option><option value="kwp">Keywords Plus</option><option value="aut">Authors</option><option value="cout">Countries</option><option value="inst">Institutions</option><option value="jou">Sources</option><option value="lan">Language</option></select></div>
            <div class="param-item"><label class="fl">Y axis</label><select id="hxy-y"><option value="aut">Authors</option><option value="cout">Countries</option><option value="inst">Institutions</option><option value="jou">Sources</option><option value="kwa">Authors Keywords</option><option value="kwp">Keywords Plus</option><option value="lan">Language</option></select></div>
            <div class="param-item"><label class="fl">Top N X</label><input type="number" id="hxy-topx" value="15" min="1"/></div>
            <div class="param-item"><label class="fl">Top N Y</label><input type="number" id="hxy-topy" value="5" min="1"/></div>
            <div class="param-item"><label class="fl">Specific X elements (comma-sep)</label><input type="text" id="hxy-ex" placeholder="optional"/></div>
            <div class="param-item"><label class="fl">Specific Y elements (comma-sep)</label><input type="text" id="hxy-ey" placeholder="optional"/></div>
          </div>
          <div class="checkbox-row"><input type="checkbox" id="hxy-rmv" checked/><span>Remove unknowns</span></div>
          <button class="run-btn" onclick="runCross('heatmap_y_x',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- NETWORK: MAIN PATH -->
      <div class="params-section" id="params-mainpath">
        <div class="params-box">
          <div class="params-title">Main Path Analysis Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Method</label><select id="mp-method"><option value="spc" selected>SPC</option><option value="splc">SPLC</option><option value="spnp">SPNP</option></select></div>
            <div class="param-item"><label class="fl">Min path size</label><input type="number" id="mp-minsize" value="2" min="1"/></div>
          </div>
          <div class="checkbox-row"><input type="checkbox" id="mp-strict" checked/><span>Enforce strict year ordering</span></div>
          <button class="run-btn" onclick="runNetwork('main_path',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- NETWORK: DIRECTED CITATION -->
      <div class="params-section" id="params-adjdir">
        <div class="params-box">
          <div class="params-title">Directed Citation Network Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Min citation count</label><input type="number" id="adir-min" value="7" min="1"/></div>
            <div class="param-item"><label class="fl">Node size</label><input type="number" id="adir-size" value="20" min="1"/></div>
            <div class="param-item"><label class="fl">Font size</label><input type="number" id="adir-font" value="10" min="6"/></div>
          </div>
          <div class="checkbox-row"><input type="checkbox" id="adir-labels" checked/><span>Show node labels</span></div>
          <div class="checkbox-row"><input type="checkbox" id="adir-local"/><span>Show only local document nodes</span></div>
          <button class="run-btn" onclick="runNetworkExtra('adj_dir',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- NETWORK: FIND DIRECTED NODES -->
      <div class="params-section" id="params-finddir">
        <div class="params-box">
          <div class="params-title">Highlight Citation Nodes Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Article IDs (comma-sep)</label><input type="text" id="fdir-articles" placeholder="44, 235"/></div>
            <div class="param-item"><label class="fl">Reference IDs (comma-sep)</label><input type="text" id="fdir-refs" placeholder="r_5602, r_6822"/></div>
            <div class="param-item"><label class="fl">Node size</label><input type="number" id="fdir-size" value="20" min="1"/></div>
            <div class="param-item"><label class="fl">Font size</label><input type="number" id="fdir-font" value="10" min="6"/></div>
          </div>
          <button class="run-btn" onclick="runNetworkExtra('find_dir',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- NETWORK: SALSA -->
      <div class="params-section" id="params-salsa">
        <div class="params-box">
          <div class="params-title">SALSA Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Max iterations</label><input type="number" id="salsa-iter" value="150" min="1"/></div>
            <div class="param-item"><label class="fl">Tolerance</label><input type="number" id="salsa-tol" value="0.000001" step="0.000001"/></div>
            <div class="param-item"><label class="fl">Top per decade</label><input type="number" id="salsa-topd" value="5" min="1"/></div>
          </div>
          <button class="run-btn" onclick="runNetworkExtra('salsa',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- REFS: MATRIX -->
      <div class="params-section" id="params-refmatrix">
        <div class="params-box">
          <div class="params-title">Reference Citation Matrix Parameters</div>
          <div class="param-item" style="margin-bottom:14px;"><label class="fl">Reference IDs (comma-sep)</label><input type="text" id="rm-ids" placeholder="r_5602, r_6822, r_6820"/></div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Start year</label><input type="number" id="rm-start" placeholder="optional"/></div>
            <div class="param-item"><label class="fl">End year</label><input type="number" id="rm-end" placeholder="optional"/></div>
          </div>
          <button class="run-btn" onclick="runRefsExtra('ref_matrix',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- REFS: CO-REFERENCES -->
      <div class="params-section" id="params-corefs">
        <div class="params-box">
          <div class="params-title">Co-References Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Group size</label><input type="number" id="cr-group" value="2" min="2"/></div>
            <div class="param-item"><label class="fl">Top N</label><input type="number" id="cr-topn" value="10" min="1"/></div>
          </div>
          <button class="run-btn" onclick="runRefsExtra('co_refs',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- REFS: CO-CITATION NETWORK -->
      <div class="params-section" id="params-cocitnet">
        <div class="params-box">
          <div class="params-title">Co-Citation Network Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Target reference ID</label><input type="text" id="ccn-target" placeholder="r_5602"/></div>
            <div class="param-item"><label class="fl">Top N</label><input type="number" id="ccn-topn" value="50" min="1"/></div>
          </div>
          <button class="run-btn" onclick="runRefsExtra('co_citation_network',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- REFS: SLEEPING BEAUTIES -->
      <div class="params-section" id="params-sleep">
        <div class="params-box">
          <div class="params-title">Sleeping Beauties Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Top N</label><input type="number" id="sb-topn" value="10" min="1"/></div>
            <div class="param-item"><label class="fl">Minimum citation count</label><input type="number" id="sb-minc" value="5" min="1"/></div>
          </div>
          <button class="run-btn" onclick="runRefsExtra('sleeping_beauties',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- REFS: PRINCES -->
      <div class="params-section" id="params-princes">
        <div class="params-box">
          <div class="params-title">Princes Parameters</div>
          <div class="callout info" style="margin-bottom:12px;">Uses the most recent Sleeping Beauties table if available. Otherwise it will compute it with the parameters below.</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Fallback Top N</label><input type="number" id="prin-topn" value="10" min="1"/></div>
            <div class="param-item"><label class="fl">Fallback minimum citation count</label><input type="number" id="prin-minc" value="5" min="1"/></div>
          </div>
          <button class="run-btn" onclick="runRefsExtra('princes',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- REFS: REFERENCE DIVERSITY -->
      <div class="params-section" id="params-refdiv">
        <div class="params-box">
          <div class="params-title">Reference Diversity Parameters</div>
          <div class="callout info" style="margin-bottom:12px;">Leave paper IDs empty to analyze all papers.</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Paper IDs (comma-sep)</label><input type="text" id="rd-paperids" placeholder="optional, e.g. 8, 56, 87"/></div>
          </div>
          <button class="run-btn" onclick="runRefsExtra('reference_diversity',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- REFS: DISRUPTION INDEX -->
      <div class="params-section" id="params-disruption">
        <div class="params-box">
          <div class="params-title">Disruption Index Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Paper IDs (comma-sep)</label><input type="text" id="di-paperids" placeholder="optional, e.g. 8, 56, 87"/></div>
            <div class="param-item"><label class="fl">Min future citers</label><input type="number" id="di-minfuture" value="1" min="0"/></div>
          </div>
          <div class="checkbox-row"><input type="checkbox" id="di-strict" checked/><span>Only count strictly future citers</span></div>
          <button class="run-btn" onclick="runRefsExtra('disruption_index',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- AI: AUTHORS PER TOPIC -->
      <div class="params-section" id="params-ai-authors">
        <div class="params-box">
          <div class="params-title">Authors per Topic Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Top N authors</label><input type="number" id="ai-auth-topn" value="15" min="1"/></div>
          </div>
          <button class="run-btn" onclick="runAITool('topics_authors',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- AI: REPRESENTATIVES -->
      <div class="params-section" id="params-ai-representatives">
        <div class="params-box">
          <div class="params-title">Representative Documents</div>
          <button class="run-btn" onclick="runAITool('topics_representatives',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- AI: SUMMARIZE -->
      <div class="params-section" id="params-ai-summarize">
        <div class="params-box">
          <div class="params-title">Summarize Abstracts</div>
          <div class="callout info" style="margin-bottom:12px;">Choose extractive or abstractive summarization. The model field is auto-filled with a recommended default for the selected method, and you can edit it manually.</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Method</label><select id="ai-sum-mode" onchange="syncSummarizeModel(false)"><option value="ext">Extractive</option><option value="abs">Abstractive</option></select></div>
            <div class="param-item"><label class="fl">Model</label><input type="text" id="ai-sum-model" value="sshleifer/distilbart-cnn-12-6" oninput="markSummarizeModelManual()"/></div>
            <div class="param-item"><label class="fl">Article IDs</label><input type="text" id="ai-sum-ids" placeholder="blank = all valid abstracts"/></div>
          </div>
          <button class="run-btn" onclick="runAITool('summarize',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- AI: WORDS VS TOPICS -->
      <div class="params-section" id="params-ai-doc-topic">
        <div class="params-box">
          <div class="params-title">AI - Topics Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Document ID</label><input type="number" id="ai-doc-id" value="0" min="0"/></div>
          </div>
          <button class="run-btn" onclick="runAITool('topics_words',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- AI: WORD EMBEDDINGS -->
      <div class="params-section" id="params-ai-word-embeddings">
        <div class="params-box">
          <div class="params-title">Word Embeddings Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Stop-word language</label><select id="aiwe-lang"><option value="ar">Arabic</option><option value="bn">Bengali</option><option value="bg">Bulgarian</option><option value="zh">Chinese</option><option value="cs">Czech</option><option value="en">English</option><option value="fi">Finnish</option><option value="fr">French</option><option value="de">German</option><option value="el">Greek</option><option value="he">Hebrew</option><option value="hi">Hindi</option><option value="hu">Hungarian</option><option value="it">Italian</option><option value="ja">Japanese</option><option value="ko">Korean</option><option value="mr">Marathi</option><option value="fa">Persian</option><option value="pl">Polish</option><option value="pt-br">Portuguese (Brazil)</option><option value="ro">Romanian</option><option value="ru">Russian</option><option value="sk">Slovak</option><option value="es">Spanish</option><option value="sv">Swedish</option><option value="th">Thai</option><option value="uk">Ukrainian</option></select></div>
            <div class="param-item"><label class="fl">Remove words (comma-sep)</label><input type="text" id="aiwe-rmv" placeholder="study, paper"></div>
            <div class="param-item"><label class="fl">Vector size</label><input type="number" id="aiwe-vector" value="100" min="10"></div>
            <div class="param-item"><label class="fl">Window</label><input type="number" id="aiwe-window" value="5" min="1"></div>
            <div class="param-item"><label class="fl">Min count</label><input type="number" id="aiwe-mincount" value="1" min="1"></div>
            <div class="param-item"><label class="fl">Epochs</label><input type="number" id="aiwe-epochs" value="10" min="1"></div>
          </div>
          <button class="run-btn" onclick="runAITool('word_embeddings',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Build Model</button>
        </div>
      </div>

      <!-- AI: WORD SIMILARITY -->
      <div class="params-section" id="params-ai-word-sim">
        <div class="params-box">
          <div class="params-title">Word Similarity Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Word 1</label><input type="text" id="aiws-word1" placeholder="innovation"></div>
            <div class="param-item"><label class="fl">Word 2</label><input type="text" id="aiws-word2" placeholder="technology"></div>
          </div>
          <button class="run-btn" onclick="runAITool('word_embeddings_sim',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- AI: WORD OPERATIONS -->
      <div class="params-section" id="params-ai-word-ops">
        <div class="params-box">
          <div class="params-title">Word Operations Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Positive words</label><input type="text" id="aiwo-positive" placeholder="king, woman"></div>
            <div class="param-item"><label class="fl">Negative words</label><input type="text" id="aiwo-negative" placeholder="man"></div>
            <div class="param-item"><label class="fl">Top N</label><input type="number" id="aiwo-topn" value="10" min="1"></div>
          </div>
          <button class="run-btn" onclick="runAITool('word_embeddings_operations',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Run</button>
        </div>
      </div>

      <!-- AI: PLOT WORD EMBEDDINGS -->
      <div class="params-section" id="params-ai-word-plot">
        <div class="params-box">
          <div class="params-title">Plot Word Embeddings Parameters</div>
          <div class="params-grid">
            <div class="param-item"><label class="fl">Positive words</label><input type="text" id="aiwp-positive" placeholder="innovation, technology"></div>
            <div class="param-item"><label class="fl">Negative words</label><input type="text" id="aiwp-negative" placeholder="optional"></div>
            <div class="param-item"><label class="fl">Top N related</label><input type="number" id="aiwp-topn" value="5" min="1"></div>
            <div class="param-item"><label class="fl">Node size</label><input type="number" id="aiwp-nodesize" value="10" min="1"></div>
            <div class="param-item"><label class="fl">Font size</label><input type="number" id="aiwp-fontsize" value="14" min="1"></div>
          </div>
          <button class="run-btn" onclick="runAITool('plot_word_embeddings',this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Generate Plot</button>
        </div>
      </div>

      <!-- AI: GRAPH -->
      <div class="params-section" id="params-ai-graph">
        <div class="params-box">
          <div class="params-title">Topic Graph</div>
          <button class="run-btn" id="ai-graph-run-btn" onclick="runTopicGraph(this)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>Generate Plot</button>
        </div>
      </div>

    </div><!-- /tab-parameters -->

    <!-- ════ TAB: DATASET ════ -->
    <div id="tab-dataset" class="tab-pane">
      <div id="dataset-empty">
        <div class="e-icon">🧾</div>
        <p>Load a dataset to inspect, edit, export, or reload <code>bibfile.data</code> in memory.</p>
      </div>
      <div id="dataset-shell" class="dataset-shell" style="display:none">
        <div class="dataset-toolbar">
          <button type="button" onclick="saveDatasetChanges()">Save changes</button>
          <button type="button" onclick="exportDatasetFile()">Export CSV</button>
          <label class="dataset-upload-label">Load Edited Dataset<input type="file" id="dataset-file-input" accept=".csv,.txt" onchange="loadEditedDatasetFile(this)"></label>
          <button type="button" onclick="resetDatasetChanges()">Reset to original</button>
          <div class="param-item" style="min-width:150px;max-width:190px;">
            <label class="fl">Delimiter</label>
            <select id="dataset-sep">
              <option value="	" selected>Tab</option>
              <option value=",">Comma</option>
              <option value=";">Semicolon</option>
              <option value="|">Pipe</option>
            </select>
          </div>
          <div class="param-item" style="min-width:140px;max-width:180px;">
            <label class="fl">Visible rows</label>
            <input id="dataset-preview-limit" type="number" min="1" step="1" value="20">
          </div>
        </div>
        <div class="callout info">You can paginate through the full dataset, add rows, remove the selected row, and edit cells inline.</div>
        <div id="dataset-meta" class="dataset-meta"></div>
        <div id="dataset-workbench" class="dataset-workbench raw-hidden">
          <div class="dataset-pane dataset-pane-main">
            <div class="dataset-grid-toolbar">
              <div>
                <div class="dataset-section-title" style="margin-bottom:0;">Editable dataframe view</div>
                <div class="dataset-grid-note">Select a row to remove it. Inline edits update the in-memory dataframe immediately. The raw text pane is a derived read-only view.</div>
              </div>
              <div class="dataset-grid-actions">
                <button type="button" class="dataset-inline-btn" onclick="addDatasetRow()">Add row</button>
                <button type="button" class="dataset-inline-btn" onclick="removeDatasetRow()">Remove selected row</button>
                <button type="button" class="dataset-inline-btn" id="dataset-raw-toggle" onclick="toggleDatasetRawPane()">Show raw text</button>
              </div>
            </div>
            <div id="dataset-grid" class="grid-shell"></div>
            <div class="dataset-grid-pager">
              <button type="button" class="dataset-inline-btn" onclick="changeDatasetPage(-1)">◀ Prev</button>
              <button type="button" class="dataset-inline-btn" onclick="changeDatasetPage(1)">Next ▶</button>
              <span class="page-label" id="dataset-page-label">Page 1 of 1</span>
              <label class="fl" style="margin-bottom:0;">Go to page</label>
              <input id="dataset-page-input" type="number" min="1" step="1" value="1">
              <button type="button" class="dataset-inline-btn" onclick="jumpToDatasetPage()">Go</button>
            </div>
          </div>
          <div id="dataset-side-pane" class="dataset-pane dataset-pane-side is-hidden">
            <div class="dataset-section-title">Raw delimited text</div>
            <textarea id="dataset-editor" class="dataset-editor" spellcheck="false" readonly></textarea>
          </div>
        </div>
      </div>
    </div>

    <!-- ════ TAB: OBJECTS ════ -->
    <div id="tab-objects" class="tab-pane">
      <div id="objects-body">
        <div id="empty-objects">
          <div class="e-icon">🗂</div>
          <p>Load a dataset to inspect object tables and IDs.</p>
        </div>
      </div>
    </div>

    <!-- ════ TAB: RESULTS ════ -->
    <div id="tab-results" class="tab-pane">
      <div id="output-body">
        <div id="empty-out">
          <div class="e-icon">📭</div>
          <p>Run an analysis to see results here</p>
        </div>
      </div>
    </div>

    <!-- ════ TAB: TEMPORAL Scholarly GRAPH ════ -->
    <div id="tab-tsg" class="tab-pane">
      <div id="tsg-body">
        <div id="empty-tsg">
          <div class="e-icon">🧭</div>
          <p>Run the Temporal Scholarly Graph to open it here</p>
        </div>
      </div>
    </div>

  </div><!-- /body -->
</div><!-- /main -->

<div id="toast"></div>

<script>
// ══ State ══
let _selectedFile = null;
let _uploadMode = 'load';
let _loaded = false;
let _currentAiKind = 'graph_topics';
let _resultCount = 0;
let _datasetPage = 1;
let _datasetTotalPages = 1;
let _datasetSelectedRow = null;
let _datasetRawVisible = false;
let _datasetColumns = [];
let _datasetAllRows = [];
let _topicModelReady = false;
let _wordEmbeddingsReady = false;
let _currentAIArea = 'topics';

// ══ Tab switching ══
function switchTab(name){
  ['functions','parameters','dataset','objects','results','tsg'].forEach(t=>{
    document.getElementById('tab-'+t).classList.toggle('active', t===name);
    document.getElementById('tbtn-'+t).classList.toggle('active', t===name);
  });
}

// ══ File drop ══
function uploadModeNeedsLoadedDataset(){
  return _uploadMode === 'merge' || _uploadMode === 'edited';
}

function refreshUploadActionState(){
  const btn = document.getElementById('load-btn');
  const requirement = document.getElementById('upload-mode-requirement');
  const ready = Boolean(_selectedFile) && (!uploadModeNeedsLoadedDataset() || _loaded);
  if(btn) btn.disabled = !ready;
  if(requirement){
    if(uploadModeNeedsLoadedDataset() && !_loaded){
      requirement.style.display = 'block';
      requirement.textContent = _uploadMode === 'merge'
        ? 'Merge file requires an existing loaded dataset.'
        : 'Load edited dataset requires an existing loaded dataset.';
    }else if(_uploadMode === 'merge' && _loaded){
      requirement.style.display = 'block';
      requirement.textContent = 'The selected file will be merged into the current in-memory dataset.';
    }else if(_uploadMode === 'edited' && _loaded){
      requirement.style.display = 'block';
      requirement.textContent = 'The selected file will replace the current working dataset.';
    }else{
      requirement.style.display = 'none';
      requirement.textContent = '';
    }
  }
}

function setSelectedUploadFile(file){
  _selectedFile = file || null;
  const label = document.getElementById('file-label');
  if(label) label.textContent = _selectedFile ? _selectedFile.name : 'No file selected';
  updateOpenAlexUploadOptions();
  refreshUploadActionState();
}

function updateOpenAlexUploadOptions(){
  const dbSel = document.getElementById('db-select');
  const row = document.getElementById('openalex-expand-row');
  const show = Boolean(dbSel) && (_uploadMode !== 'edited') && (dbSel.value === 'openalex');
  if(row) row.style.display = show ? 'flex' : 'none';
}

function updateUploadModeUI(){
  const note = document.getElementById('upload-mode-note');
  const dbRow = document.getElementById('upload-db-row');
  const editedRow = document.getElementById('upload-edited-row');
  const input = document.getElementById('file-input');
  const main = document.getElementById('drop-main');
  const sub = document.getElementById('drop-sub');
  const btn = document.getElementById('load-btn');
  document.querySelectorAll('.upload-mode-card').forEach(card=>card.classList.toggle('active', card.dataset.uploadMode === _uploadMode));
  if(_uploadMode === 'load'){
    if(note) note.textContent = 'Load a fresh export as a new working dataset.';
    if(dbRow) dbRow.style.display = 'grid';
    if(editedRow) editedRow.style.display = 'none';
    if(input) input.accept = '.bib,.csv,.txt,.json';
    if(main) main.textContent = 'Click or drag & drop your database export';
    if(sub) sub.textContent = '.bib · .csv · .txt · .json';
    if(btn) btn.textContent = 'Load Dataset';
  }else if(_uploadMode === 'merge'){
    if(note) note.textContent = 'Merge another export into the currently loaded dataset using bibfile.merge_database(...).';
    if(dbRow) dbRow.style.display = 'grid';
    if(editedRow) editedRow.style.display = 'none';
    if(input) input.accept = '.bib,.csv,.txt,.json';
    if(main) main.textContent = 'Click or drag & drop the file to merge';
    if(sub) sub.textContent = '.bib · .csv · .txt · .json';
    if(btn) btn.textContent = 'Merge File';
  }else{
    if(note) note.textContent = 'Load an edited dataset file and replace the current working table.';
    if(dbRow) dbRow.style.display = 'none';
    if(editedRow) editedRow.style.display = 'grid';
    if(input) input.accept = '.csv,.tsv,.txt';
    if(main) main.textContent = 'Click or drag & drop the edited dataset';
    if(sub) sub.textContent = '.csv · .tsv · .txt';
    if(btn) btn.textContent = 'Load Edited Dataset';
  }
  updateOpenAlexUploadOptions();
  refreshUploadActionState();
}

function setUploadMode(mode, el){
  _uploadMode = mode;
  const input = document.getElementById('file-input');
  if(input) input.value = '';
  setSelectedUploadFile(null);
  if(el){
    document.querySelectorAll('.upload-mode-card').forEach(card=>card.classList.remove('active'));
    el.classList.add('active');
  }
  updateUploadModeUI();
}

const dz = document.getElementById('drop-zone');
if(dz){
  dz.addEventListener('dragover', e=>{e.preventDefault();dz.classList.add('drag');});
  dz.addEventListener('dragleave', ()=>dz.classList.remove('drag'));
  dz.addEventListener('drop', e=>{
    e.preventDefault(); dz.classList.remove('drag');
    const f = e.dataTransfer.files[0];
    if(f) setSelectedUploadFile(f);
  });
}
function handleFile(inp){ if(inp && inp.files && inp.files[0]) setSelectedUploadFile(inp.files[0]); }
document.addEventListener('change', (e)=>{
  if(!e.target || !_loaded) return;
  if(e.target.id === 'dataset-sep' || e.target.id === 'dataset-preview-limit'){
    setDatasetPage(1);
    refreshDatasetEditor();
  }
});
document.addEventListener('keydown', (e)=>{
  if(!_loaded || !e.target || e.target.id !== 'dataset-preview-limit' || e.key !== 'Enter') return;
  e.preventDefault();
  setDatasetPage(1);
  refreshDatasetEditor();
});
document.addEventListener('keydown', (e)=>{ if(e.target && e.target.id === 'dataset-page-input' && e.key === 'Enter'){ e.preventDefault(); jumpToDatasetPage(); } });

// ══ Load data ══
function uploadSuccessToastMessage(d){
  if(_uploadMode === 'merge') return `✓ Merged file into current dataset (${d.docs} documents total)`;
  if(_uploadMode === 'edited') return '✓ Edited dataset loaded and working dataset replaced';
  return `✓ Loaded ${d.docs} documents from ${d.db}`;
}

async function finalizeLoadedDataset(d, options = {}){
  const opts = options || {};
  _loaded = true;
  updateStatus(d.docs, d.db);
  toast(opts.toastMessage || uploadSuccessToastMessage(d), 'ok');
  enableAnalysis();
  document.getElementById('stat-docs').textContent = d.docs;
  document.getElementById('stat-db').textContent = d.db;
  document.getElementById('tbtn-dataset').style.display='inline-flex';
  document.getElementById('tbtn-objects').style.display='inline-flex';
  initSankeyBuilder();
  setTopicModelReady(Boolean(d.topic_ready));
  setWordEmbeddingsReady(Boolean(d.word_embeddings_ready));
  _datasetRawVisible = false;
  setDatasetPage(1);
  const hasDatasetPayload = Array.isArray(d.columns) || !!d.grid || typeof d.csv_text === 'string';
  await syncDatasetDerivedViews({datasetPayload: hasDatasetPayload ? d : null});
  showView('dataset', document.querySelector('[data-view="dataset"]'));
  refreshUploadActionState();
}

async function loadData(){
  if(!_selectedFile) return;
  if(uploadModeNeedsLoadedDataset() && !_loaded){
    toast(_uploadMode === 'merge' ? 'Load a dataset before merging another file' : 'Load a dataset before replacing it with an edited file', 'err');
    return;
  }
  const btn = document.getElementById('load-btn');
  const selectedDb = _uploadMode === 'edited' ? '' : (document.getElementById('db-select')?.value || '');
  const expandRefs = selectedDb === 'openalex' && Boolean(document.getElementById('openalex-expand-check')?.checked);
  const idleLabel = _uploadMode === 'merge' ? 'Merge File' : (_uploadMode === 'edited' ? 'Load Edited Dataset' : 'Load Dataset');
  btn.disabled = true;
  btn.textContent = _uploadMode === 'merge' ? 'Merging…' : (_uploadMode === 'edited' ? 'Loading edited dataset…' : 'Loading…');
  if(_uploadMode === 'edited'){
    showSpinner('Loading edited dataset…');
  }else if(selectedDb === 'openalex'){
    showSpinner(expandRefs ? 'Loading OpenAlex data and expanding references…' : 'Loading OpenAlex data…');
  }else{
    showSpinner(_uploadMode === 'merge' ? 'Merging dataset…' : 'Loading dataset…');
  }
  try{
    if(_uploadMode === 'edited'){
      const fd = new FormData();
      fd.append('file', _selectedFile);
      fd.append('sep', document.getElementById('upload-edited-sep').value);
      fd.append('preview_limit', String(getDatasetPreviewLimit()));
      fd.append('page', '1');
      const r = await fetch('/api/dataset/load_csv', {method:'POST', body:fd});
      const d = await r.json();
      if(!d.ok){
        toast('Error: ' + d.error, 'err');
        appendError('Load failed: ' + d.error + '\n\n' + (d.trace || ''));
        switchTab('results');
        return;
      }
      await finalizeLoadedDataset(d);
    }else{
      const fd = new FormData();
      const selectedDb = document.getElementById('db-select').value;
      fd.append('file', _selectedFile);
      fd.append('db', selectedDb);
      fd.append('del_duplicated', document.getElementById('dup-check').checked);
      fd.append('expand_references', selectedDb === 'openalex' && document.getElementById('openalex-expand-check')?.checked);
      const url = _uploadMode === 'merge' ? '/api/upload_merge' : '/api/upload';
      const r = await fetch(url,{method:'POST',body:fd});
      const d = await r.json();
      if(!d.ok){
        toast('Error: ' + d.error, 'err');
        appendError('Load failed: ' + d.error + '\n\n' + (d.trace || ''));
        switchTab('results');
        return;
      }
      await finalizeLoadedDataset(d);
    }
  }catch(e){ toast('Connection error: ' + e.message,'err'); }
  finally{
    hideSpinner();
    btn.disabled = false;
    btn.textContent = idleLabel;
    const input = document.getElementById('file-input');
    if(input) input.value = '';
    setSelectedUploadFile(null);
  }
}

function updateStatus(docs, db){
  document.getElementById('status-dot').className='dot green';
  document.getElementById('status-txt').style.color='var(--green)';
  document.getElementById('status-txt').textContent=docs+' docs · '+db;
  const pill = document.getElementById('db-pill');
  pill.classList.add('show');
  document.getElementById('db-info').textContent = db+' · '+docs+' docs';
}

function enableAnalysis(){
  document.querySelectorAll('.nav-item.disabled').forEach(el=>el.classList.remove('disabled'));
}

async function syncDatasetDerivedViews(options = {}){
  const opts = options || {};
  const datasetPayload = opts.datasetPayload || null;
  const switchToDataset = Boolean(opts.switchToDataset);
  try{
    const statusResp = await fetch('/api/status');
    const status = await statusResp.json();
    if(status && status.loaded){
      const docs = Number(status.info?.docs ?? datasetPayload?.docs ?? datasetPayload?.rows ?? _datasetAllRows.length ?? 0);
      const db = String(status.info?.db || datasetPayload?.db || document.getElementById('stat-db')?.textContent || document.getElementById('db-select')?.value || 'dataset');
      const statDocs = document.getElementById('stat-docs');
      const statDb = document.getElementById('stat-db');
      if(statDocs) statDocs.textContent = docs;
      if(statDb) statDb.textContent = db;
      updateStatus(docs, db);
      setTopicModelReady(Boolean(status.info?.topic_ready));
      setWordEmbeddingsReady(Boolean(status.info?.word_embeddings_ready));
    }
  }catch(e){
    console.error(e);
  }
  if(datasetPayload){
    renderDatasetState(datasetPayload);
  }else if(_loaded){
    await refreshDatasetEditor();
  }
  await fetchObjects();
  if(switchToDataset) switchTab('dataset');
}

async function fetchObjects(){
  try{
    const r = await fetch('/api/objects');
    const d = await r.json();
    if(!d.ok) return;
    renderObjects(d.sections || []);
  }catch(e){ console.error(e); }
}

function escapeHtml(value){
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function getDatasetPreviewLimit(){
  const el = document.getElementById('dataset-preview-limit');
  const raw = el ? parseInt(el.value || '20', 10) : 20;
  return Number.isFinite(raw) && raw > 0 ? raw : 20;
}

function getDatasetSeparatorValue(){
  const sel = document.getElementById('dataset-sep');
  return sel ? sel.value : '\t';
}

function getDatasetSeparatorChar(){
  const sep = getDatasetSeparatorValue();
  return sep === '\t' ? '	' : sep;
}

function getDatasetPage(){
  return Number.isFinite(_datasetPage) && _datasetPage > 0 ? _datasetPage : 1;
}

function setDatasetPage(page){
  const next = parseInt(page || '1', 10);
  _datasetPage = Number.isFinite(next) && next > 0 ? next : 1;
  const inp = document.getElementById('dataset-page-input');
  if(inp) inp.value = _datasetPage;
}

function updateDatasetPager(payload = {}){
  _datasetPage = Number(payload?.page || 1) || 1;
  _datasetTotalPages = Number(payload?.total_pages || 1) || 1;
  const label = document.getElementById('dataset-page-label');
  if(label) label.textContent = `Page ${_datasetPage.toLocaleString()} of ${_datasetTotalPages.toLocaleString()}`;
  const inp = document.getElementById('dataset-page-input');
  if(inp){
    inp.min = '1';
    inp.max = String(_datasetTotalPages);
    inp.value = String(_datasetPage);
  }
}

function toggleDatasetRawPane(forceVisible = null){
  const workbench = document.getElementById('dataset-workbench');
  const sidePane = document.getElementById('dataset-side-pane');
  const btn = document.getElementById('dataset-raw-toggle');
  if(!workbench || !sidePane || !btn) return;
  _datasetRawVisible = forceVisible === null ? !_datasetRawVisible : !!forceVisible;
  workbench.classList.toggle('raw-hidden', !_datasetRawVisible);
  sidePane.classList.toggle('is-hidden', !_datasetRawVisible);
  btn.textContent = _datasetRawVisible ? 'Hide raw text' : 'Show raw text';
}

function selectDatasetRow(absRow){
  _datasetSelectedRow = Number.isFinite(absRow) ? absRow : null;
  document.querySelectorAll('#dataset-grid tbody tr').forEach(tr => {
    tr.classList.toggle('is-selected', Number(tr.dataset.absRow) === _datasetSelectedRow);
  });
}

function normalizeDatasetStateRows(rows, width){
  const colCount = Math.max(0, Number(width || 0));
  return (Array.isArray(rows) ? rows : []).map(row => {
    const seq = Array.isArray(row) ? row.slice(0, colCount) : [row];
    while(seq.length < colCount) seq.push('');
    return seq.map(value => value == null ? '' : String(value));
  });
}

function setDatasetStateFromPayload(d){
  const fullGrid = d && d.full_grid ? d.full_grid : {};
  _datasetColumns = Array.isArray(fullGrid.columns) ? fullGrid.columns.map(col => String(col ?? '')) : [];
  _datasetAllRows = normalizeDatasetStateRows(fullGrid.rows, _datasetColumns.length);
}

function getDatasetTotalRows(){
  return Array.isArray(_datasetAllRows) ? _datasetAllRows.length : 0;
}

function getDatasetTotalPagesFromState(){
  const pageSize = getDatasetPreviewLimit();
  const totalRows = getDatasetTotalRows();
  return Math.max(1, Math.ceil(totalRows / Math.max(1, pageSize)));
}

function getDatasetGridPayloadFromState(){
  const pageSize = getDatasetPreviewLimit();
  const totalRows = getDatasetTotalRows();
  const totalPages = getDatasetTotalPagesFromState();
  const page = Math.min(Math.max(1, getDatasetPage()), totalPages);
  _datasetPage = page;
  _datasetTotalPages = totalPages;
  const startRowIndex = (page - 1) * pageSize;
  const rows = _datasetAllRows.slice(startRowIndex, startRowIndex + pageSize).map(row => {
    const copy = Array.isArray(row) ? row.slice(0, _datasetColumns.length) : [];
    while(copy.length < _datasetColumns.length) copy.push('');
    return copy;
  });
  return {
    columns: _datasetColumns.slice(),
    rows,
    visible_rows: rows.length,
    total_rows: totalRows,
    total_cols: _datasetColumns.length,
    page,
    total_pages: totalPages,
    page_size: pageSize,
    start_row_index: startRowIndex,
    end_row_index: rows.length ? startRowIndex + rows.length - 1 : startRowIndex,
  };
}

function syncRawEditorFromDatasetState(){
  const editor = document.getElementById('dataset-editor');
  if(!editor) return;
  const sep = getDatasetSeparatorChar();
  if(!_datasetColumns.length){
    editor.value = '';
    return;
  }
  const lines = [];
  lines.push(_datasetColumns.map(cell => serializeDelimitedCell(cell, sep)).join(sep));
  _datasetAllRows.forEach(row => {
    const normalized = Array.isArray(row) ? row.slice(0, _datasetColumns.length) : [];
    while(normalized.length < _datasetColumns.length) normalized.push('');
    lines.push(normalized.map(cell => serializeDelimitedCell(cell, sep)).join(sep));
  });
  editor.value = lines.join('\n');
}

function updateDatasetMeta(){
  const meta = document.getElementById('dataset-meta');
  if(!meta) return;
  const sepValue = getDatasetSeparatorValue();
  const sepLabel = (sepValue === '	' || sepValue === 'tab') ? 'Tab' : (sepValue || ',');
  const payload = getDatasetGridPayloadFromState();
  meta.innerHTML = `<span>${Number(getDatasetTotalRows() || 0).toLocaleString()} rows</span><span>${Number(_datasetColumns.length || 0).toLocaleString()} columns</span><span>Delimiter: ${sepLabel}</span><span>Visible in grid: ${Number(payload.visible_rows || 0).toLocaleString()}</span><span>Rows per page: ${Number(getDatasetPreviewLimit() || 0).toLocaleString()}</span>`;
}

function renderDatasetGridFromState(){
  const grid = document.getElementById('dataset-grid');
  if(!grid) return;
  const payload = getDatasetGridPayloadFromState();
  mountGrid(grid, payload, {editable:true, caption:'Current bibfile.data'});
  updateDatasetPager(payload);
  updateDatasetMeta();
  syncRawEditorFromDatasetState();
}

function addDatasetRow(){
  if(!_datasetColumns.length){
    toast('No dataframe is loaded', 'err');
    return;
  }
  const emptyRow = new Array(_datasetColumns.length).fill('');
  const insertAt = Number.isInteger(_datasetSelectedRow) ? Math.max(0, _datasetSelectedRow + 1) : _datasetAllRows.length;
  _datasetAllRows.splice(insertAt, 0, emptyRow);
  _datasetSelectedRow = insertAt;
  const pageSize = getDatasetPreviewLimit();
  setDatasetPage(Math.floor(insertAt / Math.max(1, pageSize)) + 1);
  renderDatasetGridFromState();
  toast('✓ Empty row added', 'ok');
}

function removeDatasetRow(){
  if(!Number.isInteger(_datasetSelectedRow)){
    toast('Select a row first', 'err');
    return;
  }
  if(_datasetSelectedRow < 0 || _datasetSelectedRow >= _datasetAllRows.length){
    toast('Selected row is out of range', 'err');
    return;
  }
  _datasetAllRows.splice(_datasetSelectedRow, 1);
  if(!_datasetAllRows.length){
    _datasetSelectedRow = null;
    setDatasetPage(1);
  }else{
    _datasetSelectedRow = Math.min(_datasetSelectedRow, _datasetAllRows.length - 1);
    const pageSize = getDatasetPreviewLimit();
    setDatasetPage(Math.floor(_datasetSelectedRow / Math.max(1, pageSize)) + 1);
  }
  renderDatasetGridFromState();
  toast('✓ Row removed', 'ok');
}

function changeDatasetPage(delta){
  const next = Math.min(Math.max(1, getDatasetPage() + Number(delta || 0)), Math.max(1, getDatasetTotalPagesFromState()));
  setDatasetPage(next);
  renderDatasetGridFromState();
}

function jumpToDatasetPage(){
  const inp = document.getElementById('dataset-page-input');
  const target = inp ? parseInt(inp.value || '1', 10) : 1;
  setDatasetPage(Math.min(Math.max(1, target || 1), Math.max(1, getDatasetTotalPagesFromState())));
  renderDatasetGridFromState();
}

function parseDelimitedLine(line, sep){
  const out = [];
  let cur = '';
  let inQuotes = false;
  for(let i = 0; i < line.length; i += 1){
    const ch = line[i];
    if(ch === '"'){
      if(inQuotes && line[i + 1] === '"'){
        cur += '"';
        i += 1;
      }else{
        inQuotes = !inQuotes;
      }
    }else if(ch === sep && !inQuotes){
      out.push(cur);
      cur = '';
    }else{
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

function serializeDelimitedCell(value, sep){
  const s = String(value ?? '');
  if(s.includes('"') || s.includes('\n') || s.includes('\r') || s.includes(sep)){
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

function setGridColumnWidth(table, colIdx, width){
  const w = Math.max(90, Math.round(width));
  table.querySelectorAll('tr').forEach(tr => {
    const cell = tr.children[colIdx + 1];
    if(cell){
      cell.style.width = w + 'px';
      cell.style.minWidth = w + 'px';
      cell.style.maxWidth = w + 'px';
    }
  });
}

function initGridColumnResize(e){
  e.preventDefault();
  e.stopPropagation();
  const handle = e.currentTarget;
  const th = handle.closest('th');
  const table = handle.closest('table');
  if(!th || !table) return;
  const colIdx = Number(th.dataset.col || 0);
  const startX = e.clientX;
  const startW = th.getBoundingClientRect().width;
  function onMove(ev){
    setGridColumnWidth(table, colIdx, startW + (ev.clientX - startX));
  }
  function onUp(){
    window.removeEventListener('mousemove', onMove);
    window.removeEventListener('mouseup', onUp);
  }
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onUp);
}

function mountGrid(host, payload, opts = {}){
  if(!host) return;
  const columns = Array.isArray(payload?.columns) ? payload.columns : [];
  const rows = Array.isArray(payload?.rows) ? payload.rows : [];
  const editable = !!opts.editable;
  const tableClass = editable ? 'grid-table is-editable' : 'grid-table is-readonly';
  if(!columns.length){
    host.innerHTML = '<div class="grid-empty">No data available.</div>';
    return;
  }
  const caption = opts.caption || '';
  const captionHtml = escapeHtml(caption);
  const totals = [];
  if(Number.isFinite(payload?.visible_rows)) totals.push(`<strong>${Number(payload.visible_rows).toLocaleString()}</strong> visible rows`);
  if(Number.isFinite(payload?.total_rows)) totals.push(`<strong>${Number(payload.total_rows).toLocaleString()}</strong> total rows`);
  if(Number.isFinite(payload?.total_cols)) totals.push(`<strong>${Number(payload.total_cols).toLocaleString()}</strong> columns`);
  const startRowIndex = Number(payload?.start_row_index || 0);
  host.innerHTML = `
    ${caption || totals.length ? `<div class="grid-caption"><span>${captionHtml}</span><span>${totals.join(' · ')}</span></div>` : ''}
    <table class="${tableClass}">
      <thead>
        <tr>
          <th class="row-index" title="Row number">#</th>
          ${columns.map((col, idx) => `<th data-col="${idx}" title="${escapeHtml(col)}"><span class="cell-content">${escapeHtml(col)}</span><span class="col-resizer" title="Drag to resize column"></span></th>`).join('')}
        </tr>
      </thead>
      <tbody>
        ${rows.length ? rows.map((row, rowIdx) => {
          const absRow = startRowIndex + rowIdx;
          return `
          <tr data-abs-row="${absRow}">
            <td class="row-index row-selector" data-abs-row="${absRow}" title="Select row ${absRow + 1}">${(absRow + 1).toLocaleString()}</td>
            ${columns.map((_, colIdx) => {
              const value = Array.isArray(row) && row[colIdx] != null ? row[colIdx] : '';
              return `<td ${editable ? 'contenteditable="true"' : ''} data-row="${rowIdx}" data-abs-row="${absRow}" data-col="${colIdx}" spellcheck="false" title="${escapeHtml(value)}"><span class="cell-content">${escapeHtml(value)}</span></td>`;
            }).join('')}
          </tr>`;
        }).join('') : `<tr><td colspan="${columns.length + 1}"><div class="grid-empty">No rows to display.</div></td></tr>`}
      </tbody>
    </table>`;
  const table = host.querySelector('table');
  if(!table) return;
  table.querySelectorAll('.col-resizer').forEach(handle => handle.addEventListener('mousedown', initGridColumnResize));
  table.querySelectorAll('tbody .row-selector').forEach(cell => {
    cell.addEventListener('click', () => selectDatasetRow(Number(cell.dataset.absRow)));
  });
  table.querySelectorAll('tbody tr').forEach(tr => {
    tr.addEventListener('click', (ev) => {
      if(ev.target && ev.target.classList && ev.target.classList.contains('col-resizer')) return;
      if(editable) selectDatasetRow(Number(tr.dataset.absRow));
    });
  });
  if(editable){
    table.querySelectorAll('tbody td[contenteditable="true"]').forEach(td => {
      td.addEventListener('focus', () => {
        const current = td.innerText;
        td.textContent = current;
        selectDatasetRow(Number(td.dataset.absRow));
      });
      td.addEventListener('blur', syncGridCellToDatasetState);
      td.addEventListener('keydown', (ev) => {
        if(ev.key === 'Enter'){
          ev.preventDefault();
          td.blur();
        }
      });
    });
  }
  if(editable && Number.isInteger(_datasetSelectedRow)) selectDatasetRow(_datasetSelectedRow);
}

function syncGridCellToDatasetState(ev){
  const td = ev.currentTarget;
  if(!td) return;
  const absRow = Number(td.dataset.absRow || 0);
  const colIdx = Number(td.dataset.col || 0);
  if(!Number.isInteger(absRow) || absRow < 0 || absRow >= _datasetAllRows.length) return;
  if(!Array.isArray(_datasetAllRows[absRow])) _datasetAllRows[absRow] = new Array(_datasetColumns.length).fill('');
  while(_datasetAllRows[absRow].length < _datasetColumns.length) _datasetAllRows[absRow].push('');
  const value = td.innerText.replace(/\n/g, '').replace(/\r/g, ' ');
  _datasetAllRows[absRow][colIdx] = value;
  td.innerHTML = `<span class="cell-content">${escapeHtml(value)}</span>`;
  syncRawEditorFromDatasetState();
  selectDatasetRow(absRow);
}


function renderObjects(sections){
  const body = document.getElementById('objects-body');
  if(!body) return;
  if(!sections.length){
    body.innerHTML = '<div id="empty-objects"><div class="e-icon">🗂</div><p>No object tables available.</p></div>';
    return;
  }
  sections = [...sections].sort((a,b)=>String(a.title).localeCompare(String(b.title)));
  body.innerHTML = '<div class="obj-help">Each object family is shown as a structured dataframe. Open a section, drag the table corner to resize it, and drag header edges to resize columns.</div>' + sections.map((s, idx)=>`<div class="obj-item"><div class="obj-head" onclick="toggleObjectSection(this.parentElement)"><span class="obj-arrow">▶</span><span class="obj-title">${escapeHtml(s.title)}</span><span class="obj-count">${Number(s.count||0).toLocaleString()} rows</span></div><div class="obj-body"><div class="object-grid-wrap" id="object-grid-${idx}"></div></div></div>`).join('');
  sections.forEach((s, idx) => {
    const host = document.getElementById(`object-grid-${idx}`);
    mountGrid(host, s.grid || {}, {editable:false, caption: s.title || 'Object table'});
  });
}

function toggleObjectSection(el){ el.classList.toggle('open'); }

async function refreshDatasetEditor(useCurrentSep=true){
  try{
    const sep = useCurrentSep ? getDatasetSeparatorValue() : '\t';
    const previewLimit = getDatasetPreviewLimit();
    const r = await fetch('/api/dataset/current?sep=' + encodeURIComponent(sep) + '&preview_limit=' + encodeURIComponent(previewLimit) + '&page=' + encodeURIComponent(getDatasetPage()));
    const d = await r.json();
    if(!d.ok) return;
    renderDatasetState(d);
  }catch(e){ console.error(e); }
}

function renderDatasetState(d){
  const empty = document.getElementById('dataset-empty');
  const shell = document.getElementById('dataset-shell');
  if(!empty || !shell) return;
  empty.style.display = 'none';
  shell.style.display = 'flex';
  const sepSel = document.getElementById('dataset-sep');
  if(sepSel && d.sep) sepSel.value = d.sep;
  const previewLimitInput = document.getElementById('dataset-preview-limit');
  if(previewLimitInput) previewLimitInput.value = Number(d.preview_limit || 20);
  setDatasetStateFromPayload(d || {});
  const requestedPage = Number(d?.grid?.page || getDatasetPage() || 1);
  setDatasetPage(requestedPage);
  if(_datasetSelectedRow != null && _datasetSelectedRow >= _datasetAllRows.length){
    _datasetSelectedRow = _datasetAllRows.length ? _datasetAllRows.length - 1 : null;
  }
  renderDatasetGridFromState();
}


async function saveDatasetChanges(){
  if(!_datasetColumns.length){ return; }
  showSpinner('Applying dataset changes and recomputing everything…');
  try{
    syncRawEditorFromDatasetState();
    const editor = document.getElementById('dataset-editor');
    const payload = {
      csv_text: editor ? editor.value : '',
      sep: getDatasetSeparatorValue(),
      preview_limit: getDatasetPreviewLimit(),
      page: getDatasetPage()
    };
    const d = await post('/api/dataset/apply', payload);
    hideSpinner();
    if(!d.ok){ appendError((d.error || 'Failed to save dataset') + '\n\n' + (d.trace || '')); toast('Error: ' + d.error, 'err'); return; }
    _loaded = true;
    document.getElementById('tbtn-dataset').style.display='inline-flex';
    document.getElementById('tbtn-objects').style.display='inline-flex';
    enableAnalysis();
    clearOutput();
    await syncDatasetDerivedViews({datasetPayload: d, switchToDataset: true});
    const activeCard = document.querySelector('#view-dataset .fn-card.active');
    if(activeCard) activeCard.classList.remove('active');
    switchTab('dataset');
    toast('✓ Dataset reapplied, bibfile.data replaced, results cleared, and objects recomputed', 'ok');
  }catch(e){ hideSpinner(); appendError(e.message); toast('Error: ' + e.message, 'err'); }
}

async function exportDatasetFile(){
  try{
    syncRawEditorFromDatasetState();
    const editor = document.getElementById('dataset-editor');
    triggerDownload('data.csv', editor ? editor.value : '', 'text/csv;charset=utf-8');
    toast('✓ Dataset exported', 'ok');
  }catch(e){ toast('Error: ' + e.message, 'err'); }
}


async function loadEditedDatasetFile(inp){
  const f = inp && inp.files && inp.files[0];
  if(!f) return;
  const sepSel = document.getElementById('dataset-sep');
  const fd = new FormData();
  fd.append('file', f);
  fd.append('sep', sepSel ? sepSel.value : '	');
  fd.append('preview_limit', String(getDatasetPreviewLimit()));
  fd.append('page', String(getDatasetPage()));
  showSpinner('Reapplying edited dataset and recomputing everything…');
  try{
    const r = await fetch('/api/dataset/load_csv', {method:'POST', body: fd});
    const d = await r.json();
    hideSpinner();
    if(!d.ok){ appendError((d.error || 'Failed to load edited dataset') + '\\n\\n' + (d.trace || '')); toast('Error: ' + d.error, 'err'); return; }
    _loaded = true;
    document.getElementById('tbtn-dataset').style.display='inline-flex';
    document.getElementById('tbtn-objects').style.display='inline-flex';
    enableAnalysis();
    clearOutput();
    await syncDatasetDerivedViews({datasetPayload: d, switchToDataset: true});
    const activeCard = document.querySelector('#view-dataset .fn-card.active');
    if(activeCard) activeCard.classList.remove('active');
    switchTab('dataset');
    toast('✓ Edited dataset reapplied, bibfile.data replaced, results cleared, and objects recomputed', 'ok');
  }catch(e){ hideSpinner(); appendError(e.message); toast('Error: ' + e.message, 'err'); }
  finally{ if(inp) inp.value = ''; }
}

async function resetDatasetChanges(){
  const sepSel = document.getElementById('dataset-sep');
  showSpinner('Restoring original dataset…');
  try{
    const d = await post('/api/dataset/reset', {sep: sepSel ? sepSel.value : '\\t'});
    hideSpinner();
    if(!d.ok){ appendError((d.error || 'Failed to reset dataset') + '\n\n' + (d.trace || '')); toast('Error: ' + d.error, 'err'); return; }
    await syncDatasetDerivedViews({datasetPayload: d});
    toast('✓ Dataset restored and all objects were refreshed', 'ok');
  }catch(e){ hideSpinner(); appendError(e.message); toast('Error: ' + e.message, 'err'); }
}

// ══ View switching ══
function showView(id, el, aiArea=null){
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
  document.getElementById('view-'+id).classList.add('active');
  if(el) el.classList.add('active');
  const titles = {
    upload:'Upload Data', dataset:'Dataset & Reports', visuals:'Visualizations',
    network:'Network Analysis', temporal:'Temporal Scholarly Graph', references:'Reference Analysis',
    profiling:'Entity Profiling', ai:'AI & Tools'
  };
  const subs = {
    upload:'Select your database export file to begin',
    dataset:'EDA, health report, and editable dataset synchronization',
    visuals:'Interactive charts and document projections',
    network:'Co-authorship, keywords, and citation networks',
    temporal:'Dedicated paper-centred temporal explorer with its own workspace tab',
    references:'Citation patterns, RPYS, and trajectories',
    profiling:'Profile entities or compute H, G, E, J, and M indices',
    ai:'Topic modelling, word-level analysis, and concise topic summaries'
  };
  if(id === 'ai') setAIArea(aiArea || (el && el.dataset && el.dataset.aiArea) || _currentAIArea || 'topics');
  document.getElementById('view-title').textContent = titles[id] || id;
  document.getElementById('view-sub').textContent = subs[id] || '';
  switchTab('functions');
}

function setTopicModelReady(ready){
  _topicModelReady = Boolean(ready);
  syncSummarizeModel(true);
  updateAITopicDependencyUI();
}

function setWordEmbeddingsReady(ready){
  _wordEmbeddingsReady = Boolean(ready);
  updateAIWordEmbeddingDependencyUI();
}

function topicDependentKinds(){
  return new Set(['topics_authors','topics_representatives','topics_words']);
}

function wordEmbeddingDependentKinds(){
  return new Set(['word_embeddings_sim','word_embeddings_operations','plot_word_embeddings']);
}

function ensureTopicModelReady(){
  if(_topicModelReady) return true;
  toast('Run Create Topics first to unlock this action', 'err');
  return false;
}

function ensureWordEmbeddingsReady(){
  if(_wordEmbeddingsReady) return true;
  toast('Run Word Embeddings first to unlock this action', 'err');
  return false;
}

function updateAITopicDependencyUI(){
  document.querySelectorAll('#view-ai .fn-card[data-topic-required="1"]').forEach(card=>{
    card.classList.toggle('disabled', !_topicModelReady);
    card.setAttribute('aria-disabled', _topicModelReady ? 'false' : 'true');
  });
  ['ai-graph-run-btn'].forEach(id=>{
    const btn = document.getElementById(id);
    if(btn) btn.disabled = !_topicModelReady;
  });
  document.querySelectorAll('#params-ai-authors .run-btn, #params-ai-representatives .run-btn, #params-ai-doc-topic .run-btn').forEach(btn=>{
    btn.disabled = !_topicModelReady;
  });
}

function updateAIWordEmbeddingDependencyUI(){
  document.querySelectorAll('#view-ai .fn-card[data-word-emb-required="1"]').forEach(card=>{
    card.classList.toggle('disabled', !_wordEmbeddingsReady);
    card.setAttribute('aria-disabled', _wordEmbeddingsReady ? 'false' : 'true');
  });
  document.querySelectorAll('#params-ai-word-sim .run-btn, #params-ai-word-ops .run-btn, #params-ai-word-plot .run-btn').forEach(btn=>{
    btn.disabled = !_wordEmbeddingsReady;
  });
}

function setAIArea(area){
  _currentAIArea = area || 'topics';
  document.querySelectorAll('#view-ai .ai-group').forEach(group=>{
    const active = group.dataset.aiGroup === _currentAIArea;
    group.classList.toggle('active', active);
    group.style.display = active ? 'block' : 'none';
  });
  document.querySelectorAll('.nav-item[data-view="ai"]').forEach(item=>{
    item.classList.toggle('active', item.dataset.aiArea === _currentAIArea);
  });
}

function summarizeDefaultModel(mode){
  return mode === 'abs' ? 'google/pegasus-xsum' : 'sshleifer/distilbart-cnn-12-6';
}

function markSummarizeModelManual(){
  _summarizeModelTouched = true;
}

function syncSummarizeModel(force=false){
  const modeEl = document.getElementById('ai-sum-mode');
  const modelEl = document.getElementById('ai-sum-model');
  if(!modeEl || !modelEl) return;
  const nextModel = summarizeDefaultModel(modeEl.value);
  if(force || !_summarizeModelTouched || !modelEl.value.trim() || modelEl.value.trim() === summarizeDefaultModel(modeEl.dataset.lastMode || 'ext')){
    modelEl.value = nextModel;
    _summarizeModelTouched = false;
  }
  modeEl.dataset.lastMode = modeEl.value;
}

// ══ Select function → show params tab ══
function selectParams(paramsId, section, card){
  if(section === 'ai' && card && card.classList.contains('disabled')){
    if(card.dataset && card.dataset.wordEmbRequired === '1'){ ensureWordEmbeddingsReady(); return; }
    ensureTopicModelReady();
    return;
  }
  if(paramsId === 'ai-summarize') syncSummarizeModel(true);
  // Deactivate all cards in section
  const view = document.getElementById('view-'+section);
  if(view) view.querySelectorAll('.fn-card').forEach(c=>c.classList.remove('active'));
  if(card) card.classList.add('active');
  // Hide all params sections
  document.querySelectorAll('.params-section').forEach(p=>p.style.display='none');
  document.getElementById('params-empty').style.display='none';
  const el = document.getElementById('params-'+paramsId);
  if(el){ el.style.display='block'; } else { document.getElementById('params-empty').style.display='flex'; }
  switchTab('parameters');
}

// ══ Run helpers ══
async function post(url, body={}){
  const r = await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  return r.json();
}

function setLoading(btn, on){
  btn.classList.toggle('loading', on);
  const svg = btn.querySelector('svg');
  if(svg) svg.style.display = on ? 'block' : 'none';
}

async function runAndDisplay(url, body, btn, label){
  setLoading(btn, true);
  showSpinner(label+' running…');
  try{
    const d = await post(url, body);
    hideSpinner();
    if(!d.ok){ appendError(d.error+'\n\n'+(d.trace||'')); toast('Error: '+d.error,'err'); switchTab('results'); return; }
    renderResult(d, label);
    if(d.docs) document.getElementById('stat-docs').textContent = d.docs;
    if(url==='/api/filter'){ await fetchObjects(); await refreshDatasetEditor(false); }
    toast('✓ Done','ok');
    switchTab('results');
  }catch(e){ hideSpinner(); toast('Error: '+e.message,'err'); appendError(e.message); switchTab('results'); }
  finally{ setLoading(btn, false); }
}

// ── EDA / Health (no params — run directly and switch to results) ──
function runEda(card){
  card.classList.add('active');
  _runSimple('/api/eda', 'EDA Report', card);
}
function runHealth(card){
  card.classList.add('active');
  _runSimple('/api/health', 'Health Check', card);
}
async function _runSimple(url, label, btn){
  setLoading(btn, true);
  showSpinner(label+'…');
  try{
    const d = await post(url);
    hideSpinner();
    if(!d.ok){
      appendError(d.error + '\n\n' + (d.trace || ''));
      toast('Error: ' + d.error, 'err');
      switchTab('results');
      return;
    }
    renderResult(d, label);
    toast('✓ Done','ok');
    switchTab('results');
  }catch(e){ hideSpinner(); appendError(e.message); switchTab('results'); }
  finally{ setLoading(btn, false); }
}

// ── Filter ──
function runFilter(btn){ runAndDisplay('/api/filter',{
  documents: document.getElementById('f-docs').value,
  doc_type: document.getElementById('f-dtype').value,
  year_str: document.getElementById('f-ystr').value,
  year_end: document.getElementById('f-yend').value,
  sources: document.getElementById('f-sources').value,
  core: document.getElementById('f-core').value,
  country: document.getElementById('f-country').value,
  language: document.getElementById('f-lang').value,
  abstract: document.getElementById('f-abs').checked
}, btn, 'Filter'); }

// ── Visualizations ──
function runWordcloud(btn){ runAndDisplay('/api/wordcloud',{entry:document.getElementById('wc-entry').value,wordsn:document.getElementById('wc-words').value,rmv_words:document.getElementById('wc-rmv').value},btn,'Word Cloud'); }
function runNgrams(btn){ runAndDisplay('/api/ngrams',{entry:document.getElementById('ng-entry').value,ngrams:document.getElementById('ng-n').value,wordsn:document.getElementById('ng-topn').value,lang:document.getElementById('ng-lang').value,rmv_words:document.getElementById('ng-rmv').value},btn,'N-Grams'); }
function runTreemap(btn){ runAndDisplay('/api/treemap',{entry:document.getElementById('tm-entry').value,topn:document.getElementById('tm-topn').value},btn,'TreeMap'); }
function runBars(btn){ runAndDisplay('/api/bars',{statistic:document.getElementById('bars-stat').value,topn:document.getElementById('bars-topn').value},btn,'Bar Chart'); }
// function runEvolution(btn){ runAndDisplay('/api/evolution',{key:document.getElementById('ev-key').value,topn:document.getElementById('ev-topn').value,start:document.getElementById('ev-start').value,end:document.getElementById('ev-end').value,lang:document.getElementById('ev-lang').value},btn,'Evolution'); }
function runTermGrowth(btn){ runAndDisplay('/api/term_growth',{source:document.getElementById('tg-source').value,topn:document.getElementById('tg-topn').value,cumulative:document.getElementById('tg-cumulative').checked,lang:document.getElementById('tg-lang').value,rmv_words:document.getElementById('tg-rmv').value,start:document.getElementById('tg-start').value,end:document.getElementById('tg-end').value,line:document.getElementById('tg-line').checked,bubble:document.getElementById('tg-bubble').checked},btn,'Term Growth'); }

function sankeyLayerRow(idx, withRemove=true){
  const removeBtn = withRemove
    ? `<button type="button" class="sankey-remove" onclick="removeSankeyLayer(${idx})">✕</button>`
    : '<div></div>';

  const topNBlock = idx===0
    ? `<div class="param-item">
         <label class="fl">.</label>
         <input type="text" class="sk-topn" placeholder="Not required. No previous layer." disabled>
       </div>`
    : `<div class="param-item">
         <label class="fl">Top N from previous layer</label>
         <input type="number" class="sk-topn" placeholder="Leave blank for all">
       </div>`;

  return `<div class="sankey-layer" data-idx="${idx}">
    <div class="param-item">
      <label class="fl">Layer ${idx+1}</label>
      <select class="sk-entry">
        <option value="aut">Authors</option>
        <option value="kwa">Authors Keywords</option>
        <option value="cout">Countries</option>
        <option value="inst">Institutions</option>
        <option value="kwp">Keywords Plus</option>
        <option value="lan">Language</option>
        <option value="jou">Sources</option>
      </select>
    </div>
    ${topNBlock}
    ${removeBtn}
  </div>`;
}
function initSankeyBuilder(){
  const host = document.getElementById('sankey-builder');
  if(!host || host.dataset.ready==='1') return;
  host.innerHTML = sankeyLayerRow(0,false) + sankeyLayerRow(1,false);
  host.querySelectorAll('.sk-entry')[1].value = 'lan';
  host.dataset.ready='1';
}
function addSankeyLayer(){
  const host = document.getElementById('sankey-builder');
  if(!host) return;
  const idx = host.querySelectorAll('.sankey-layer').length;
  host.insertAdjacentHTML('beforeend', sankeyLayerRow(idx,true));
}
function removeSankeyLayer(idx){
  const host = document.getElementById('sankey-builder');
  const rows = [...host.querySelectorAll('.sankey-layer')];
  if(rows.length <= 2) return;
  const target = rows.find(r=>Number(r.dataset.idx)===idx);
  if(target) target.remove();
  [...host.querySelectorAll('.sankey-layer')].forEach((row,i)=>{
    row.dataset.idx = i;
    const label = row.querySelector('label.fl');
    if(label) label.textContent = `Layer ${i+1}`;
    const topn = row.querySelector('.sk-topn');
    if(topn) topn.disabled = i===0;
    const btn = row.querySelector('.sankey-remove');
    if(btn) btn.setAttribute('onclick', `removeSankeyLayer(${i})`);
  });
}
function runSankey(btn){
  initSankeyBuilder();
  const layers = [...document.querySelectorAll('#sankey-builder .sankey-layer')].map((row,i)=>({
    entry: row.querySelector('.sk-entry').value,
    topn: i===0 ? '' : row.querySelector('.sk-topn').value
  }));
  runAndDisplay('/api/sankey',{layers:layers,rmv_unknowns:document.getElementById('sk-rmv').checked},btn,'Sankey');
}

function runProductivity(btn){ runAndDisplay('/api/productivity',{kind:document.getElementById('pr-kind').value,topn:document.getElementById('pr-topn').value},btn,'Productivity'); }
function runProjection(btn){ runAndDisplay('/api/projection',{corpus_type:document.getElementById('pj-corpus').value,lang:document.getElementById('pj-lang').value,rmv_words:document.getElementById('pj-rmv').value,n_components:document.getElementById('pj-components').value,n_clusters:document.getElementById('pj-clusters').value,node_labels:document.getElementById('pj-labels').checked,node_size:document.getElementById('pj-size').value,node_font_size:document.getElementById('pj-font').value,tf_idf:document.getElementById('pj-tfidf').checked,embeddings:document.getElementById('pj-emb').checked,model:document.getElementById('pj-model').value,method:document.getElementById('pj-method').value,showlegend:document.getElementById('pj-legend').checked,cluster_method:document.getElementById('pj-cm').value,min_size:document.getElementById('pj-minsize').value,max_size:document.getElementById('pj-maxsize').value},btn,'Projection'); }
function runCross(kind, btn){
  const bodies = {
    count_y_x: {kind:'count_y_x',x:document.getElementById('cxy-x').value,y:document.getElementById('cxy-y').value,topn_x:document.getElementById('cxy-topx').value,topn_y:document.getElementById('cxy-topy').value,text_font_size:document.getElementById('cxy-font').value,x_angle:document.getElementById('cxy-angle').value,rmv_unknowns:document.getElementById('cxy-rmv').checked},
    heatmap_y_x: {kind:'heatmap_y_x',x:document.getElementById('hxy-x').value,y:document.getElementById('hxy-y').value,topn_x:document.getElementById('hxy-topx').value,topn_y:document.getElementById('hxy-topy').value,element_x:document.getElementById('hxy-ex').value,element_y:document.getElementById('hxy-ey').value,rmv_unknowns:document.getElementById('hxy-rmv').checked}
  };
  runAndDisplay('/api/cross', bodies[kind]||{kind}, btn, kind==='count_y_x' ? 'Count Y per X' : 'Heatmap Y per X');
}

async function runTemporalSG(btn){
  setLoading(btn, true);
  showSpinner('Temporal Scholarly Graph running…');
  try{
    const d = await post('/api/temporal_sg', {
      view: document.getElementById('tsg-view').value,
      center: document.getElementById('tsg-center').value,
      max_papers: document.getElementById('tsg-maxp').value,
      max_references: document.getElementById('tsg-maxr').value,
      start_year: document.getElementById('tsg-start').value,
      end_year: document.getElementById('tsg-end').value
    });
    hideSpinner();
    if(!d.ok){ appendError(d.error+'\n\n'+(d.trace||'')); toast('Error: '+d.error,'err'); switchTab('results'); return; }
    renderTemporalSG(d);
    if(d.docs) document.getElementById('stat-docs').textContent = d.docs;
    toast('✓ Done','ok');
    switchTab('tsg');
  }catch(e){ hideSpinner(); toast('Error: '+e.message,'err'); appendError(e.message); switchTab('results'); }
  finally{ setLoading(btn, false); }
}

// ── Networks ──
function runNetwork(kind, btn){
  const bodies = {
    adj:  {kind:'adj',adj_type:document.getElementById('adj-type').value,min_count:document.getElementById('adj-min').value,node_labels:document.getElementById('adj-labels').checked,node_size:document.getElementById('adj-size').value,label_type:document.getElementById('adj-labeltype').value,centrality:document.getElementById('adj-centrality').value},
    map:  {kind:'map',connections:document.getElementById('map-conn').checked,country_lst:document.getElementById('map-countries').value},
    sim:  {kind:'sim',sim_type:document.getElementById('sim-type').value,node_size:document.getElementById('sim-size').value,node_labels:document.getElementById('sim-labels').checked,cut_coup:document.getElementById('sim-coup').value,cut_cocit:document.getElementById('sim-cocit').value},
    hist: {kind:'hist',min_links:document.getElementById('hist-min').value,chain:document.getElementById('hist-chain').value,path:document.getElementById('hist-path').checked,node_size:document.getElementById('hist-size').value,font_size:document.getElementById('hist-font').value,node_labels:document.getElementById('hist-labels').checked,dist:document.getElementById('hist-dist').value,dist_pad:document.getElementById('hist-distpad').value},
    main_path: {kind:'main_path',method:document.getElementById('mp-method').value,min_path_size:document.getElementById('mp-minsize').value,strict_year:document.getElementById('mp-strict').checked}
  };
  runAndDisplay('/api/network', bodies[kind]||{kind}, btn, 'Network '+kind);
}

function runNetworkExtra(kind, btn){
  const bodies = {
    adj_dir: {kind:'adj_dir',min_count:document.getElementById('adir-min').value,node_labels:document.getElementById('adir-labels').checked,local_nodes:document.getElementById('adir-local').checked,node_size:document.getElementById('adir-size').value,font_size:document.getElementById('adir-font').value},
    find_dir: {kind:'find_dir',article_ids:document.getElementById('fdir-articles').value,ref_ids:document.getElementById('fdir-refs').value,node_size:document.getElementById('fdir-size').value,font_size:document.getElementById('fdir-font').value},
    salsa: {kind:'salsa',max_iter:document.getElementById('salsa-iter').value,tol:document.getElementById('salsa-tol').value,topn_decade:document.getElementById('salsa-topd').value}
  };
  runAndDisplay('/api/network', bodies[kind]||{kind}, btn, 'Network');
}

// ── References ──
function runRefs(kind, btn){
  const bodies = {
    top_refs:   {kind:'top_refs',topn:document.getElementById('tr-topn').value,font_size:document.getElementById('tr-font').value,use_ref_id:document.getElementById('tr-useid').checked,date_start:document.getElementById('tr-start').value,date_end:document.getElementById('tr-end').value},
    rpys:       {kind:'rpys',peaks_only:document.getElementById('rpys-peaks').checked},
    trajectory: {kind:'trajectory',ref_ids:document.getElementById('traj-ids').value,ref_names:document.getElementById('traj-names').value}
  };
  if(kind==='trajectory' && !document.getElementById('traj-ids').value.trim() && !document.getElementById('traj-names').value.trim()){
    toast('Please enter at least one reference ID or name','err'); return;
  }
  runAndDisplay('/api/refs', bodies[kind]||{kind}, btn, 'References');
}

function runRefsExtra(kind, btn){
  const bodies = {
    ref_matrix: {kind:'ref_matrix',ref_ids:document.getElementById('rm-ids').value,date_start:document.getElementById('rm-start').value,date_end:document.getElementById('rm-end').value},
    co_refs: {kind:'co_refs',group:document.getElementById('cr-group').value,topn:document.getElementById('cr-topn').value},
    co_citation_network: {kind:'co_citation_network',target_ref_id:document.getElementById('ccn-target').value,topn:document.getElementById('ccn-topn').value},
    sleeping_beauties: {kind:'sleeping_beauties',topn:document.getElementById('sb-topn').value,min_count:document.getElementById('sb-minc').value},
    princes: {kind:'princes',topn:document.getElementById('prin-topn').value,min_count:document.getElementById('prin-minc').value},
    reference_diversity: {kind:'reference_diversity',paper_ids:document.getElementById('rd-paperids').value},
    disruption_index: {kind:'disruption_index',paper_ids:document.getElementById('di-paperids').value,strict_future:document.getElementById('di-strict').checked,min_future_citers:document.getElementById('di-minfuture').value}
  };
  runAndDisplay('/api/refs', bodies[kind]||{kind}, btn, 'References');
}

// ── Profiling ──
function runProfiling(btn){
  const name = document.getElementById('prof-name').value.trim();
  const pid  = document.getElementById('prof-id').value.trim();
  if(!name && !pid){ toast('Please enter a Name or Entity ID','err'); return; }
  runAndDisplay('/api/profiling',{kind:document.getElementById('prof-kind').value,name:name||null,id:pid||null,topn:document.getElementById('prof-topn').value},btn,'Profiling');
}

// ── Metrics ──
function runMetrics(btn){ runAndDisplay('/api/hindex',{year:document.getElementById('mi-year').value},btn,'Metrics'); }

// ── AI / Topics ──
let _currentAiKindGlobal = 'graph_topics';
let _summarizeModelTouched = false;
function selectAIGraph(kind, card){
  if(card && card.classList.contains('disabled')){ ensureTopicModelReady(); return; }
  _currentAiKindGlobal = {graph_topics:'topics',distribution:'distribution',projection:'projection',heatmap:'heatmap',time:'time'}[kind] || kind;
  const view = document.getElementById('view-ai');
  if(view) view.querySelectorAll('.fn-card').forEach(c=>c.classList.remove('active'));
  if(card) card.classList.add('active');
  document.querySelectorAll('.params-section').forEach(p=>p.style.display='none');
  document.getElementById('params-empty').style.display='none';
  document.getElementById('params-ai-graph').style.display='block';
  switchTab('parameters');
}
async function runTopics(btn){
  setLoading(btn, true);
  showSpinner('Topic Modelling running…');
  try{
    const d = await post('/api/topics', {lang:document.getElementById('ai-lang').value, embeddings:document.getElementById('ai-emb').checked});
    hideSpinner();
    if(!d.ok){ appendError(d.error+'\n\n'+(d.trace||'')); toast('Error: '+d.error,'err'); switchTab('results'); return; }
    renderResult(d, 'Topic Modelling');
    setTopicModelReady(true);
    toast('✓ Topic model created. Dependent AI cards are now enabled', 'ok');
    switchTab('results');
  }catch(e){ hideSpinner(); toast('Error: '+e.message,'err'); appendError(e.message); switchTab('results'); }
  finally{ setLoading(btn, false); }
}
function runTopicGraph(btn){
  if(!ensureTopicModelReady()) return;
  runAndDisplay('/api/topics_graph',{kind:_currentAiKindGlobal},btn,'Topic Graph');
}
async function runAITool(kind, btn){
  if(topicDependentKinds().has(kind) && !ensureTopicModelReady()) return;
  if(wordEmbeddingDependentKinds().has(kind) && !ensureWordEmbeddingsReady()) return;
  const bodies = {
    topics_authors: {kind:'topics_authors',topn:document.getElementById('ai-auth-topn').value},
    topics_representatives: {kind:'topics_representatives'},
    topics_words: {kind:'topics_words',doc_id:document.getElementById('ai-doc-id').value},
    summarize: {kind:'summarize',mode:document.getElementById('ai-sum-mode').value,model:document.getElementById('ai-sum-model').value,article_ids:document.getElementById('ai-sum-ids').value},
    word_embeddings: {kind:'word_embeddings',lang:document.getElementById('aiwe-lang').value,rmv_words:document.getElementById('aiwe-rmv').value,vector_size:document.getElementById('aiwe-vector').value,window:document.getElementById('aiwe-window').value,min_count:document.getElementById('aiwe-mincount').value,epochs:document.getElementById('aiwe-epochs').value},
    word_embeddings_sim: {kind:'word_embeddings_sim',word_1:document.getElementById('aiws-word1').value,word_2:document.getElementById('aiws-word2').value},
    word_embeddings_operations: {kind:'word_embeddings_operations',positive:document.getElementById('aiwo-positive').value,negative:document.getElementById('aiwo-negative').value,topn:document.getElementById('aiwo-topn').value},
    plot_word_embeddings: {kind:'plot_word_embeddings',positive:document.getElementById('aiwp-positive').value,negative:document.getElementById('aiwp-negative').value,topn:document.getElementById('aiwp-topn').value,node_size:document.getElementById('aiwp-nodesize').value,font_size:document.getElementById('aiwp-fontsize').value}
  };
  const labels = {
    topics_authors: 'AI Tools',
    topics_representatives: 'Representative Documents',
    topics_words: 'AI - Topics',
    summarize: document.getElementById('ai-sum-mode')?.value === 'abs' ? 'Abstractive Summary' : 'Extractive Summary',
    word_embeddings: 'Word Embeddings',
    word_embeddings_sim: 'Word Similarity',
    word_embeddings_operations: 'Word Operations',
    plot_word_embeddings: 'Plot Word Embeddings'
  };
  if(kind === 'word_embeddings'){
    setLoading(btn, true);
    showSpinner('Word Embeddings running…');
    try{
      const d = await post('/api/ai_tools', bodies[kind]);
      hideSpinner();
      if(!d.ok){ appendError(d.error+'\n\n'+(d.trace||'')); toast('Error: '+d.error,'err'); switchTab('results'); return; }
      renderResult(d, labels[kind]);
      setWordEmbeddingsReady(true);
      toast('✓ Word embeddings created. Dependent AI cards are now enabled', 'ok');
      switchTab('results');
    }catch(e){ hideSpinner(); toast('Error: '+e.message,'err'); appendError(e.message); switchTab('results'); }
    finally{ setLoading(btn, false); }
    return;
  }
  runAndDisplay('/api/ai_tools', bodies[kind]||{kind}, btn, labels[kind] || 'AI Tools');
}

// ══ Render output ══
function prettyTitle(raw){
  return String(raw || 'Output')
    .replace(/^ask[_-]?gpt[_-]?/i,'')
    .replace(/^ask[_-]?/i,'')
    .replace(/[_-]+/g,' ')
    .replace(/\b\w/g, m => m.toUpperCase())
    .trim() || 'Output';
}
function decodeB64Text(value){
  if(!value) return '';
  try{
    const binary = atob(value);
    const bytes = Uint8Array.from(binary, ch => ch.charCodeAt(0));
    return new TextDecoder('utf-8').decode(bytes);
  }catch(_err){
    try{ return atob(value); }catch(_err2){ return ''; }
  }
}
function getTablePayload(node){
  const root = node?.closest?.('.table-wrap') || node;
  const script = root?.querySelector?.('.df-payload');
  if(!script) return null;
  try{ return JSON.parse(script.textContent || '{}'); }catch(_err){ return null; }
}
function enhanceTableWrap(wrap, title='Table'){
  const table = wrap.querySelector('table');
  if(!table || table.dataset.enhanced === '1') return;
  table.dataset.enhanced = '1';
  const tbody = table.tBodies && table.tBodies[0];
  const thead = table.tHead;
  if(!tbody || !thead || !thead.rows.length) return;

  const payload = getTablePayload(wrap);
  const headerRow = thead.rows[thead.rows.length - 1];
  const allRows = [...tbody.rows];
  const tools = document.createElement('div');
  tools.className = 'table-tools';
  const stat = document.createElement('div');
  stat.className = 'table-stat';
  stat.textContent = `${allRows.length} rows`;
  const resetBtn = document.createElement('button');
  resetBtn.type = 'button';
  resetBtn.textContent = 'Reset filters';
  tools.appendChild(stat);
  if(payload?.truncated && payload?.full_html_b64){
    const expandBtn = document.createElement('button');
    expandBtn.type = 'button';
    expandBtn.textContent = 'Show all rows';
    expandBtn.addEventListener('click', ()=>{
      const fullHtml = decodeB64Text(payload.full_html_b64);
      if(!fullHtml){ toast('Could not expand full table', 'err'); return; }
      wrap.innerHTML = fullHtml;
      enhanceTableWrap(wrap, title);
      toast('✓ Full table loaded', 'ok');
    });
    tools.appendChild(expandBtn);
  }
  tools.appendChild(resetBtn);
  wrap.prepend(tools);

  table.querySelectorAll('th, td').forEach(cell => {
    const text = (cell.textContent || '').trim();
    if(text) cell.title = text;
  });

  const filterRow = thead.insertRow(-1);
  filterRow.className = 'filter-row';
  [...headerRow.cells].forEach((cell, idx)=>{
    const th = document.createElement('th');
    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = `Filter ${cell.textContent.trim() || ('col ' + (idx + 1))}`;
    th.appendChild(input);
    filterRow.appendChild(th);
  });

  let sortCol = -1;
  let sortDir = 1;
  [...headerRow.cells].forEach((cell, idx)=>{
    const marker = document.createElement('span');
    marker.className = 'sort-ind';
    marker.textContent = '↕';
    cell.appendChild(marker);
    cell.addEventListener('click', ()=>{
      if(sortCol === idx){
        sortDir = sortDir * -1;
      }else{
        sortCol = idx;
        sortDir = 1;
      }
      [...headerRow.cells].forEach((other, otherIdx)=>{
        const ind = other.querySelector('.sort-ind');
        if(!ind) return;
        ind.textContent = otherIdx === sortCol ? (sortDir === 1 ? '↑' : '↓') : '↕';
      });
      applyTableState();
    });
  });

  function cellValue(row, idx){
    return (row.cells[idx]?.textContent || '').trim();
  }

  function compareRows(a, b, idx){
    const av = cellValue(a, idx);
    const bv = cellValue(b, idx);
    const an = Number(av.replace(/,/g,''));
    const bn = Number(bv.replace(/,/g,''));
    const aIsNum = av !== '' && Number.isFinite(an);
    const bIsNum = bv !== '' && Number.isFinite(bn);
    if(aIsNum && bIsNum) return an - bn;
    return av.localeCompare(bv, undefined, {numeric:true, sensitivity:'base'});
  }

  function applyTableState(){
    const filters = [...filterRow.cells].map(th => (th.querySelector('input')?.value || '').trim().toLowerCase());
    let visibleRows = allRows.filter(row => filters.every((flt, idx) => !flt || cellValue(row, idx).toLowerCase().includes(flt)));
    if(sortCol >= 0){
      visibleRows = visibleRows.slice().sort((a,b)=>compareRows(a,b,sortCol) * sortDir);
    }
    allRows.forEach(row => { row.style.display = 'none'; });
    visibleRows.forEach(row => {
      row.style.display = '';
      tbody.appendChild(row);
    });
    stat.textContent = `${visibleRows.length} / ${allRows.length} rows · ${title}`;
  }

  [...filterRow.querySelectorAll('input')].forEach(input => input.addEventListener('input', applyTableState));
  resetBtn.addEventListener('click', ()=>{
    [...filterRow.querySelectorAll('input')].forEach(input => input.value = '');
    sortCol = -1;
    sortDir = 1;
    [...headerRow.cells].forEach(other => {
      const ind = other.querySelector('.sort-ind');
      if(ind) ind.textContent = '↕';
    });
    applyTableState();
  });
  applyTableState();
}


function makeOutItem(label){
  const card = document.createElement('section');
  card.className = 'out-item';

  const hdr = document.createElement('div');
  hdr.className = 'out-item-hdr';

  const tag = document.createElement('span');
  tag.className = 'tag';
  tag.textContent = `Result ${_resultCount + 1}`;

  const ttl = document.createElement('strong');
  ttl.textContent = label || 'Output';

  const time = document.createElement('span');
  time.style.marginLeft = 'auto';
  time.textContent = new Date().toLocaleString();

  const tabs = document.createElement('div');
  tabs.className = 'out-item-tabs';

  hdr.appendChild(tag);
  hdr.appendChild(ttl);
  hdr.appendChild(tabs);
  hdr.appendChild(time);

  const body = document.createElement('div');
  body.className = 'out-item-body';

  card.appendChild(hdr);
  card.appendChild(body);
  return card;
}

function sanitizeDownloadName(name, fallback='output'){
  return String(name || fallback)
    .trim()
    .replace(/[^a-z0-9._-]+/gi, '_')
    .replace(/^_+|_+$/g, '') || fallback;
}
function triggerDownload(filename, content, mime='text/plain;charset=utf-8'){
  const blob = new Blob([content], {type: mime});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(()=>URL.revokeObjectURL(url), 500);
}
function escapeCSVCell(value){
  const s = String(value == null ? '' : value).replace(/\r?\n|\r/g, ' ');
  return /[",;]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}
function tableElementToCSV(tableEl){
  if(!tableEl) return '';
  const payload = getTablePayload(tableEl);
  if(payload?.csv_b64){
    const csv = decodeB64Text(payload.csv_b64);
    if(csv) return csv;
  }
  const rows = [...tableEl.querySelectorAll('tr')];
  return rows.map((tr)=>{
    const cells = [...tr.querySelectorAll('th,td')];
    return cells.map((cell)=>escapeCSVCell(cell.innerText.trim())).join(',');
  }).join('\n');
}
function downloadTextFile(filename, text){
  triggerDownload(sanitizeDownloadName(filename, 'output') + '.txt', text || '', 'text/plain;charset=utf-8');
}
function downloadTableFile(filename, tableEl){
  triggerDownload(sanitizeDownloadName(filename, 'table') + '.csv', tableElementToCSV(tableEl), 'text/csv;charset=utf-8');
}

function renderTemporalSG(d){
  const body = document.getElementById('tsg-body');
  const oldFrame = body.querySelector('iframe');
  if(oldFrame) oldFrame.srcdoc = '';
  body.innerHTML = '';

  const shell = document.createElement('div');
  shell.className = 'tsg-shell';

  if(d.stats){
    const stats = document.createElement('div');
    stats.className = 'tsg-stats';
    const parts = [];
    if(d.stats.papers != null) parts.push(`${Number(d.stats.papers).toLocaleString()} papers`);
    if(d.stats.nodes != null) parts.push(`${Number(d.stats.nodes).toLocaleString()} nodes`);
    if(d.stats.edges != null) parts.push(`${Number(d.stats.edges).toLocaleString()} edges`);
    stats.textContent = parts.length ? parts.join(' · ') : 'Interactive temporal graph';
    shell.appendChild(stats);
  }

  const iframe = document.createElement('iframe');
  iframe.className = 'html-out-frame';
  iframe.setAttribute('loading', 'lazy');
  iframe.srcdoc = d.html || '';
  shell.appendChild(iframe);
  body.appendChild(shell);
}

function renderResult(d, label){
  const body = document.getElementById('output-body');
  document.getElementById('empty-out').style.display='none';

  const card = makeOutItem(label);
  const tabs = card.querySelector('.out-item-tabs');
  const content = card.querySelector('.out-item-body');
  const panels = [];

  function addPanel(kind, title, node, activate=false, downloadInfo=null){
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'out-tab' + (activate ? ' active' : '');
    btn.textContent = title;
    const panel = document.createElement('div');
    panel.className = 'out-panel' + (activate ? ' active' : '');
    panel.dataset.kind = kind;
    if(downloadInfo){
      const tools = document.createElement('div');
      tools.className = 'out-panel-tools';
      const dl = document.createElement('button');
      dl.type = 'button';
      dl.className = 'out-download-btn';
      dl.textContent = downloadInfo.kind === 'table' ? 'Download CSV' : 'Download TXT';
      dl.onclick = ()=>{
        if(downloadInfo.kind === 'table') downloadTableFile(downloadInfo.filename || title, downloadInfo.tableEl || node.querySelector('table'));
        else downloadTextFile(downloadInfo.filename || title, downloadInfo.text || node.textContent || '');
      };
      tools.appendChild(dl);
      panel.appendChild(tools);
    }
    panel.appendChild(node);
    btn.onclick = ()=>{
      tabs.querySelectorAll('.out-tab').forEach(b=>b.classList.remove('active'));
      content.querySelectorAll('.out-panel').forEach(p=>p.classList.remove('active'));
      btn.classList.add('active');
      panel.classList.add('active');
      panel.querySelectorAll('.plotly-out').forEach((div)=>{
        if(div.dataset.rendered === '1') return;
        const spec = JSON.parse(div.dataset.plotlySpec || '{}');
        if(!spec.layout) spec.layout = {};
        const isSankey = Array.isArray(spec.data) && spec.data.some(trace => trace && trace.type === 'sankey');
        if(isSankey){
          spec.layout.paper_bgcolor = '#000000';
          spec.layout.plot_bgcolor = '#000000';
          spec.layout.font = Object.assign({color:'#f9fafb',family:'IBM Plex Mono'}, spec.layout.font || {});
        }else{
          if(!spec.layout.paper_bgcolor) spec.layout.paper_bgcolor = '#eceff3';
          if(!spec.layout.plot_bgcolor) spec.layout.plot_bgcolor = '#eceff3';
          spec.layout.font = Object.assign({color:'#111827',family:'IBM Plex Mono'}, spec.layout.font || {});
          if(spec.layout.xaxis && !spec.layout.xaxis.gridcolor) spec.layout.xaxis.gridcolor = 'rgba(0,0,0,.12)';
          if(spec.layout.yaxis && !spec.layout.yaxis.gridcolor) spec.layout.yaxis.gridcolor = 'rgba(0,0,0,.12)';
        }
        spec.layout.margin = spec.layout.margin || {t:40,b:60,l:60,r:20};
        requestAnimationFrame(()=>{
          Plotly.newPlot(div, spec.data || [], spec.layout || {}, {responsive:true, displayModeBar:true, displaylogo:false});
          div.dataset.rendered = '1';
        });
      });
    };
    tabs.appendChild(btn);
    content.appendChild(panel);
    panels.push({kind, btn, panel});
  }

  let chartIndex = 0;
  (d.plotly || []).forEach((fig)=>{
    chartIndex += 1;
    const plotDiv = document.createElement('div');
    plotDiv.className = 'plotly-out';
    plotDiv.dataset.plotlySpec = fig;
    addPanel('chart', chartIndex === 1 ? 'Chart' : `Chart ${chartIndex}`, plotDiv, panels.length === 0);
  });

  (d.images || []).forEach((b64, i)=>{
    const img = document.createElement('img');
    img.src = 'data:image/png;base64,' + b64;
    img.className = 'img-out';
    addPanel('chart', chartIndex === 0 && i === 0 ? 'Chart' : `Image ${i+1}`, img, panels.length === 0);
  });

  if(d.html){
    const wrap = document.createElement('div');
    if(d.stats){
      const stats = document.createElement('div');
      stats.className = 'html-stats';
      const parts = [];
      if(d.stats.papers != null) parts.push(`${Number(d.stats.papers).toLocaleString()} papers`);
      if(d.stats.nodes != null) parts.push(`${Number(d.stats.nodes).toLocaleString()} nodes`);
      if(d.stats.edges != null) parts.push(`${Number(d.stats.edges).toLocaleString()} edges`);
      stats.textContent = parts.length ? parts.join(' · ') : 'Interactive HTML result';
      wrap.appendChild(stats);
    }
    const iframe = document.createElement('iframe');
    iframe.className = 'html-out-frame';
    iframe.setAttribute('loading', 'lazy');
    iframe.srcdoc = d.html;
    wrap.appendChild(iframe);
    addPanel('html', 'Explorer', wrap, panels.length === 0);
  }

  if(d.result && d.result.type === 'dataframe'){
    const wrap = document.createElement('div');
    wrap.className = 'table-wrap';
    wrap.innerHTML = d.result.html;
    enhanceTableWrap(wrap, 'Table');
    addPanel('table', 'Table', wrap, panels.length === 0, {kind:'table', filename:'Table', tableEl: wrap.querySelector('table')});
  }

  if(d.tables){
    const entries = Object.entries(d.tables);
    entries.forEach(([key, html], i)=>{
      const wrap = document.createElement('div');
      wrap.className = 'table-wrap';
      wrap.innerHTML = html;
      const title = prettyTitle(entries.length === 1 ? key : key || `table_${i+1}`);
      enhanceTableWrap(wrap, title);
      addPanel('table', title, wrap, panels.length === 0, {kind:'table', filename:title, tableEl: wrap.querySelector('table')});
    });
  }

  if(Array.isArray(d.artifacts)){
    d.artifacts.forEach((artifact, i)=>{
      const title = prettyTitle(artifact.title || `artifact_${i+1}`);
      if(artifact.type === 'table' && artifact.html){
        const wrap = document.createElement('div');
        wrap.className = 'table-wrap';
        wrap.innerHTML = artifact.html;
        enhanceTableWrap(wrap, title);
        addPanel('table', title, wrap, panels.length === 0, {kind:'table', filename:title, tableEl: wrap.querySelector('table')});
      }else if(artifact.value){
        const pre = document.createElement('pre');
        pre.className = 'text-out';
        pre.textContent = artifact.value;
        addPanel('text', title, pre, panels.length === 0, {kind:'text', filename:title, text: artifact.value});
      }
    });
  }

  const textBlocks = [];
  if(d.stdout && d.stdout.trim()) textBlocks.push(d.stdout.trim());
  if(d.result && (d.result.type === 'text' || d.result.type === 'dict') && d.result.value) textBlocks.push(d.result.value);
  if(textBlocks.length){
    const pre = document.createElement('pre');
    pre.className = 'text-out';
    const joinedText = textBlocks.join('\n\n');
    pre.textContent = joinedText;
    addPanel('text', 'Text', pre, panels.length === 0, {kind:'text', filename:'Text', text: joinedText});
  }

  if(!panels.length){
    const pre = document.createElement('pre');
    pre.className = 'text-out';
    pre.textContent = 'No visible output returned.';
    addPanel('text', 'Text', pre, true);
  }

  body.prepend(card);
  const badge = document.getElementById('results-badge');
  _resultCount++;
  badge.textContent = _resultCount;
  badge.style.display = 'inline-block';

  const firstBtn = tabs.querySelector('.out-tab.active') || tabs.querySelector('.out-tab');
  if(firstBtn) firstBtn.click();
}
// ══ Spinner ══
function showSpinner(msg){
  let sp = document.getElementById('spinner-overlay');
  if(!sp){
    sp = document.createElement('div');
    sp.id = 'spinner-overlay';
    sp.innerHTML = '<div class="spinner-card"><div class="spinner-orbit"></div><div class="spinner-copy"><div class="spinner-title">Working...</div><div class="spinner-sub">Please wait</div></div></div>';
    document.body.appendChild(sp);
  }
  const title = sp.querySelector('.spinner-title');
  if(title) title.textContent = msg || 'Working...';
  sp.classList.add('show');
}
function hideSpinner(){ const sp=document.getElementById('spinner-overlay'); if(sp) sp.classList.remove('show'); }

// ══ Clear ══
function clearOutput(){
  document.getElementById('output-body').innerHTML='<div id="empty-out"><div class="e-icon">📭</div><p>Run an analysis to see results here</p></div>';
  const tsgBody = document.getElementById('tsg-body');
  if(tsgBody) tsgBody.innerHTML='<div id="empty-tsg"><div class="e-icon">🧭</div><p>Run the Temporal Scholarly Graph to open it here</p></div>';
  _resultCount=0;
  document.getElementById('results-badge').style.display='none';
  window._resultFilter = 'all';
}

function appendError(message){
  renderResult({result:{type:'text', value:String(message || 'Unknown error')}}, 'Error');
}

// ══ Toast ══
function toast(msg, type=''){
  const t = document.getElementById('toast');
  t.textContent=msg; t.className='show'+(type?' '+type:'');
  setTimeout(()=>t.className='',3000);
}


const PARAM_HELP = {
  'pj-lang':'Optional stopword language for the projection corpus. Leave blank for none.',
  'pj-components':'Projection dimensionality. Usually 2 for plotting.',
  'pj-clusters':'Used when cluster method is K-Means.',
  'pj-method':'tsvd is fast; umap and tsne are nonlinear reductions.',
  'pj-cm':'kmeans assigns every document; HDBSCAN can mark outliers.',
  'pj-model':'Transformer model used only when embeddings are enabled.',
  'adj-labeltype':'When labels are shown, choose whether nodes display IDs or names.',
  'hist-chain':'Optional document IDs to highlight a citation chain.',
  'tr-useid':'Switches the plot labels from full references to reference IDs.',
  'f-docs':'List of document positions to retain. Example: 0, 5, 12.',
  'f-dtype':'Exact document types to retain, as shown in the EDA report.',
  'f-ystr':'Start year. Leave blank for all years.',
  'f-yend':'End year. Leave blank for all years.',
  'f-sources':'Exact source names to retain, separated by commas.',
  'f-core':'Bradford core selection: 1, 2, 3, 12, 23, or blank for all.',
  'f-country':'Exact country names to retain, separated by commas.',
  'f-lang':'Exact language names to retain, separated by commas.',
  'sk-rmv':'When enabled, removes flows that contain UNKNOWN values.',
  'prof-id':'Use IDs exactly as shown in the Objects & IDs tab, such as a_1 or r_5.',
  'prof-name':'Optional alternative to the ID. Exact entity names work best.',
  'mi-year':'Used only for M-index. It normalizes H-index by career length.',
  'ng-lang':'Optional stopword language. Leave blank to apply no predefined stopword list.',
  'ev-lang':'Optional stopword language. Leave blank to apply no predefined stopword list.',
  'ai-lang':'Optional stopword language. Leave blank to apply no predefined stopword list.',
  'aiemb-lang':'Optional stopword language. Leave blank to apply no predefined stopword list.',
  'tsg-maxp':'Maximum number of dataset papers sent to the explorer. Higher values are richer but heavier.',
  'tsg-maxr':'Maximum number of cited references represented as reference nodes.',
  'tsg-start':'Optional lower bound for publication year. Leave blank for automatic.',
  'tsg-end':'Optional upper bound for publication year. Leave blank for automatic.'
};
function addHelpTips(){
  Object.entries(PARAM_HELP).forEach(([id,tip])=>{
    const input = document.getElementById(id);
    if(!input) return;
    const label = input.closest('.param-item')?.querySelector('label.fl') || input.parentElement?.querySelector('label.fl');
    if(!label || label.querySelector('.help-tip')) return;
    const span = document.createElement('span');
    span.className = 'help-tip';
    span.tabIndex = 0;
    span.textContent = '?';
    span.setAttribute('data-tip', tip);
    label.appendChild(span);
  });
}
function normalizeStopwordSelects(){
  ['ng-lang','ev-lang','ai-lang','aiemb-lang','aiwe-lang','pj-lang'].forEach(id=>{
    const sel = document.getElementById(id);
    if(!sel) return;
    if(!sel.querySelector('option[value=""]')){
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'None';
      sel.prepend(opt);
    }
    sel.value = '';
  });
}
function sortSelectByText(id){
  const sel = document.getElementById(id);
  if(!sel) return;
  const options = [...sel.options];
  const empty = options.filter(o => o.value === '');
  const rest = options.filter(o => o.value !== '').sort((a,b)=>a.text.localeCompare(b.text, undefined, {numeric:true, sensitivity:'base'}));
  sel.innerHTML = '';
  [...empty, ...rest].forEach(o => sel.add(o));
}
function sortDropdowns(){
  [
    'dbsel','wc-entry','ng-entry','ng-lang','tm-entry','ev-key','ev-lang','pr-kind','pj-corpus','pj-lang','pj-method','pj-cm',
    'adj-type','adj-centrality','adj-labeltype','sim-type','tsg-view','prof-kind','ai-lang','aiwe-lang','cxy-x','cxy-y','hxy-x','hxy-y','aiemb-corpus','aiemb-lang'
  ].forEach(sortSelectByText);
}

// ══ Check server status on load ══
window._resultFilter = 'all';
initSankeyBuilder();
normalizeStopwordSelects();
sortDropdowns();
addHelpTips();
fetch('/api/status').then(r=>r.json()).then(d=>{
  if(d.loaded){
    _loaded = true;
    updateStatus(d.info.docs, d.info.db);
    enableAnalysis();
    document.getElementById('tbtn-dataset').style.display='inline-flex';
    document.getElementById('tbtn-objects').style.display='inline-flex';
    fetchObjects();
    initSankeyBuilder();
  }
  setTopicModelReady(Boolean(d.info?.topic_ready));
  setWordEmbeddingsReady(Boolean(d.info?.word_embeddings_ready));
  setAIArea('topics');
  updateUploadModeUI();
  const dbSel = document.getElementById('db-select');
  if(dbSel) dbSel.addEventListener('change', updateOpenAlexUploadOptions);
  updateOpenAlexUploadOptions();
});
</script>
</body>
</html>"""



@app.route('/api/temporal_sg', methods=['POST'])
def temporal_sg_route():
    if not STATE['pbx']:
        return jsonify({'ok': False, 'error': 'No dataset loaded'})
    d   = request.json or {}
    pbx = STATE['pbx']
    try:
        result = pbx.temporal_sg(
            view                    = d.get('view', 'timeline'),
            center                  = d.get('center', 'paper'),
            max_papers              = int(d.get('max_papers', 500)),
            max_references          = int(d.get('max_references', 300)),
            color_by                = d.get('color_by', 'type'),
            size_by                 = d.get('size_by', 'citations'),
            start_year              = int(d['start_year']) if d.get('start_year') else None,
            end_year                = int(d['end_year'])   if d.get('end_year')   else None,
            notebook                = False,
            open_browser            = False,
            save_html               = None,
            preview                 = True,
        )
        html = result['html']
        html = re.sub(r'<button\s+class="ego-btn"[^>]*>.*?<\/button>', '', html, flags=re.IGNORECASE | re.DOTALL)
        return jsonify({
            'ok':   True,
            'html': html,
            'stats': {
                'nodes':  len(result['nodes']),
                'edges':  len(result['edges']),
                'papers': len(result['papers']),
            }
        })
    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()})

# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def web_app(port=5173, open_browser=True):
    """Start the PyBibX web app in the background."""
    global _SERVER, _SERVER_THREAD, _SERVER_URL
    import socket

    if _SERVER is not None:
        print(f"\n  pybibx Web App already running  →  {_SERVER_URL}\n")
        return _SERVER_URL

    def free_port(p):
        for candidate in range(p, p + 20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("localhost", candidate)) != 0:
                    return candidate
        return p

    port = free_port(port)
    url = f"http://localhost:{port}"

    _SERVER = make_server("0.0.0.0", port, app)
    _SERVER_URL = url
    _SERVER_THREAD = threading.Thread(target=_SERVER.serve_forever, daemon=True)
    _SERVER_THREAD.start()

    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"      pybibx Web App       ")
    print(f"      {url:<40}")
    print(f"  ╚══════════════════════════════════════════╝\n")
    print("")
    print("Terminate the web service using: pybibx.web_stop() \n")


    return url
    
def web_stop():
    """Stop the PyBibX web app if it is running."""
    global _SERVER, _SERVER_THREAD, _SERVER_URL

    if _SERVER is None:
        print("\n  pybibx Web App is not running.\n")
        return False

    _SERVER.shutdown()
    _SERVER.server_close()

    if _SERVER_THREAD is not None and _SERVER_THREAD.is_alive():
        _SERVER_THREAD.join(timeout=2)

    _SERVER = None
    _SERVER_THREAD = None
    _SERVER_URL = None

    print("\n  pybibx Web App stopped.\n")
    return True

def launch(port=5173, open_browser=True):
    """Backward-compatible launcher."""
    return web_app(port=port, open_browser=open_browser)
    
    
if __name__ == "__main__":
    web_app()
    try:
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        web_stop()
