############################################################################

# pyBibX - Internal batch primitives.
#
# This module contains *only* generic batch primitives used by the
# pbx_probe class to switch between an in-memory ("small") path and a
# chunked ("batch") path. It must NOT contain any bibliometric
# semantics. Anything that knows about authors, references, journals,
# countries, etc. lives in pbx.py.

############################################################################

from dataclasses import dataclass

import numpy as np
import pandas as pd

############################################################################

@dataclass
class BatchConfig:
    """Configuration for internal batch processing.

    Attributes
    ----------
    mode : str
        One of "auto", "off", "on".
        - "auto": switch to batch path when the dataset exceeds
          ``auto_batch_rows`` rows or ``auto_batch_memory_mb`` MB.
        - "off":  always use the in-memory ("small") path.
        - "on":   always use the chunked ("batch") path.
    auto_batch_rows : int
        Threshold (rows) above which the auto-mode triggers batching.
    auto_batch_memory_mb : int
        Threshold (megabytes) above which the auto-mode triggers
        batching, based on a rough DataFrame memory estimate.
    text_chunk_size : int
        Chunk size (number of strings) for ``clear_text``.
    count_chunk_size : int
        Chunk size (rows) for ``__make_bib`` and counting passes.
    dedup_chunk_size : int
        Chunk size (rows) for ``merge_database``.
    embedding_text_chunk_size : int
        Chunk size (texts) for ``create_embeddings``.
    embedding_batch_size : int
        Forwarded as ``batch_size`` to the encoder.
    tfidf_dense_limit_rows : int
        Above this corpus size, ``dtm_tf_idf`` returns sparse output
        when called with ``return_type='auto'``.
    verbose : bool
        Toggle progress messages emitted by batch helpers.
    """

    mode: str = "auto"
    auto_batch_rows: int = 20000
    auto_batch_memory_mb: int = 256
    text_chunk_size: int = 5000
    count_chunk_size: int = 10000
    dedup_chunk_size: int = 10000
    embedding_text_chunk_size: int = 2000
    embedding_batch_size: int = 128
    tfidf_dense_limit_rows: int = 5000
    verbose: bool = False

############################################################################

def estimate_dataframe_memory_mb(df):
    """Rough estimate of DataFrame memory usage in megabytes.

    Uses ``df.memory_usage(deep=True)`` so object dtype columns
    (where most of the bibliographic text lives) are counted
    realistically.
    """
    if df is None:
        return 0.0
    try:
        if not isinstance(df, pd.DataFrame):
            return 0.0
        if df.shape[0] == 0:
            return 0.0
        bytes_total = int(df.memory_usage(index=True, deep=True).sum())
        return bytes_total / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def should_batch_df(df, config):
    """Decide whether an operation on ``df`` should run in batch mode.

    Returns False for None / empty / very small frames.
    Returns True when either the row threshold or the memory
    threshold from ``config`` is exceeded.
    """
    if df is None:
        return False
    try:
        nrows = len(df)
    except Exception:
        return False
    if nrows == 0:
        return False
    if nrows >= int(getattr(config, 'auto_batch_rows', 20000)):
        return True
    mem_mb = estimate_dataframe_memory_mb(df)
    if mem_mb >= float(getattr(config, 'auto_batch_memory_mb', 256)):
        return True
    return False

############################################################################

def chunk_dataframe(df, chunk_size):
    """Yield ``df`` in contiguous slices of ``chunk_size`` rows.

    Each slice is returned as a view-like ``DataFrame`` whose original
    integer index is preserved. The caller is expected to know the
    global row offset (typically ``chunk.index[0]``) when needed.
    """
    if df is None:
        return
    try:
        n = len(df)
    except Exception:
        return
    if n == 0:
        return
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        chunk_size = n
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield df.iloc[start:end]


def chunk_list(items, chunk_size):
    """Yield a list / sequence in contiguous slices of ``chunk_size``."""
    if items is None:
        return
    try:
        n = len(items)
    except Exception:
        # Generic iterable fallback.
        buf = []
        for it in items:
            buf.append(it)
            if len(buf) >= chunk_size:
                yield buf
                buf = []
        if buf:
            yield buf
        return
    if n == 0:
        return
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        chunk_size = n
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield items[start:end]

############################################################################

def batch_apply(df, func, chunk_size, concat=True):
    """Apply ``func`` to each chunk of ``df``.

    ``func`` should take a DataFrame chunk and return a DataFrame.
    When ``concat=True`` (default) the chunks are concatenated and
    returned as a single DataFrame. When False, the list of partial
    results is returned.
    """
    results = []
    for chunk in chunk_dataframe(df, chunk_size):
        results.append(func(chunk))
    if not concat:
        return results
    if not results:
        return pd.DataFrame()
    return pd.concat(results, axis=0, ignore_index=False)


def batch_reduce(df, map_func, reduce_func, chunk_size):
    """Streaming map-reduce over ``df`` chunks.

    ``map_func(chunk) -> partial`` is called per chunk and
    ``reduce_func(accumulator, partial) -> accumulator`` folds the
    partial result into an accumulator. The final accumulator is
    returned. For the first chunk the accumulator argument is None.
    """
    accumulator = None
    for chunk in chunk_dataframe(df, chunk_size):
        partial = map_func(chunk)
        accumulator = reduce_func(accumulator, partial)
    return accumulator


def merge_counter_dicts(results):
    """Sum a list of ``Counter``-like dicts into a single dict.

    Accepts dictionaries of ``key -> int`` (or float). Missing keys
    are treated as 0. Returns a regular ``dict``.
    """
    out = {}
    if not results:
        return out
    for d in results:
        if not d:
            continue
        for key, value in d.items():
            out[key] = out.get(key, 0) + value
    return out


def concat_numpy_chunks(chunks):
    """Concatenate a list of numpy arrays along axis 0.

    Returns an empty 0-dim float array when ``chunks`` is empty or
    contains only empty arrays. Tolerates mixed scalar/2-D shapes
    by coercing to ``np.asarray``.
    """
    if not chunks:
        return np.empty((0, 0))
    arrays = []
    for c in chunks:
        if c is None:
            continue
        a = np.asarray(c)
        if a.size == 0:
            continue
        arrays.append(a)
    if not arrays:
        return np.empty((0, 0))
    if len(arrays) == 1:
        return arrays[0]
    return np.vstack(arrays)


def sparse_vstack_safe(chunks):
    """Vertically stack a list of sparse / dense matrices safely.

    Used for incremental construction of TF-IDF-like matrices.
    Returns ``None`` for an empty input.
    """
    if not chunks:
        return None
    try:
        from scipy.sparse import issparse, vstack as _sp_vstack
    except Exception:
        # Fall back to dense numpy stacking.
        return concat_numpy_chunks(chunks)
    cleaned = [c for c in chunks if c is not None]
    if not cleaned:
        return None
    if any(issparse(c) for c in cleaned):
        return _sp_vstack(cleaned)
    return concat_numpy_chunks(cleaned)
