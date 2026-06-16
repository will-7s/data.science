"""
parsers.py  —  Fast, numpy-first file parsers.

Architecture
------------
The pipeline is split into two phases:

  Phase 1 — Raw bytes → Python objects (language-specific C libraries)
    JSON/JSONL : orjson   (3-5× faster than stdlib json, Rust C-extension)
    CSV        : csv      (stdlib C parser)
    Excel/ODS  : pandas read_excel (openpyxl/xlrd engine)
    Parquet    : pyarrow  (Arrow C++ columnar engine → zero-copy numpy)

  Phase 2 — Python objects → numpy arrays (_to_numpy_arrays)
    All paths converge here.  No pandas after this point.

    For JSON/JSONL, the input is list[dict] (row-major).
    pd.DataFrame() is used as a thin C bridge to transpose row-major
    dicts into column-major arrays in one C-level pass.  The DataFrame
    is discarded immediately after — it is never returned or stored.

    For CSV, the input is already column-major (list[list]).
    pd.DataFrame is NOT used.  Each column is cast with np.array()
    and a vectorised float attempt (no Python loop per cell).

    For Parquet, pyarrow .to_pydict() returns column-major lists of
    Python scalars; same vectorised numpy path as CSV.

Type inference  (_infer_numpy_col)
    1. Check dtype of first non-null value (O(1) — no full scan).
       int/float → try arr.astype(float64) on the full array.
    2. On failure or string input → try float64 cast on string array.
    3. On failure → str array with None → ''.
    All three steps are vectorised numpy operations; no Python per-cell loop.

Deduplication  (in loader.py)
    Numpy void-view + np.unique — O(n log n), no pandas.

Encoding detection  (_to_utf8)
    BOM-based detection: UTF-32 → UTF-16 → UTF-8 BOM → plain UTF-8.
    Transcoding uses Python's C codec layer (decode+encode in one C call).
"""
from __future__ import annotations
import csv
import io
import numpy as np

try:
    import orjson as _json
    def _json_loads(b: bytes):
        return _json.loads(b)
except ImportError:
    import json as _json                    # type: ignore[no-redef]
    def _json_loads(b: bytes):              # type: ignore[misc]
        return _json.loads(b.decode('utf-8', errors='replace'))


# ── Encoding detection ────────────────────────────────────────────────────────

def _to_utf8(raw: bytes) -> bytes:
    """
    Detect BOM, transcode to UTF-8.  Returns raw bytes if already UTF-8.
    Handles UTF-32 BE/LE, UTF-16 BE/LE, UTF-8 BOM, plain UTF-8.
    Uses Python's C codec layer — no Python byte loop.
    """
    if raw[:4] in (b'\x00\x00\xfe\xff', b'\xff\xfe\x00\x00'):
        return raw.decode('utf-32').encode('utf-8')
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return raw.decode('utf-16').encode('utf-8')
    if raw[:3] == b'\xef\xbb\xbf':
        return raw[3:]
    return raw


# ── Vectorised type inference ─────────────────────────────────────────────────

def _infer_numpy_col(obj_arr: np.ndarray) -> np.ndarray:
    """
    Convert a 1-D numpy object array to float64 or str array.

    Steps (all vectorised, no Python per-cell loop):
    1. Build None mask with np.equal (handles NaN-like None from JSON).
    2. Check dtype of first non-null value for a fast-path decision.
    3. Attempt arr.astype(float64) on non-null values.
    4. Fallback: str array, None → ''.
    """
    n         = len(obj_arr)
    none_mask = np.equal(obj_arr, None)     # vectorised None check
    non_null  = obj_arr[~none_mask]

    # All-null column → float64 full of NaN
    if non_null.size == 0:
        return np.full(n, np.nan, dtype=np.float64)

    # Fast-path: first non-null is already numeric (int/float from JSON/CSV)
    first = non_null.flat[0]
    is_numeric_first = isinstance(first, (int, float)) and not isinstance(first, bool)

    if is_numeric_first:
        out = np.full(n, np.nan, dtype=np.float64)
        try:
            out[~none_mask] = non_null.astype(np.float64)
            return out
        except (ValueError, TypeError):
            pass

    # Slow path: string column — try numeric cast on the string values
    try:
        out = np.full(n, np.nan, dtype=np.float64)
        out[~none_mask] = non_null.astype(np.float64)
        return out
    except (ValueError, TypeError):
        pass

    # String fallback — replace None with ''
    result = obj_arr.astype(str)
    result[none_mask] = ''
    return result


def _list_to_numpy(values: list) -> np.ndarray:
    """
    Convert a plain Python list to a typed numpy array.
    Used for CSV columns (already column-major, no None from JSON).
    """
    obj = np.array(values, dtype=object)
    return _infer_numpy_col(obj)


# ── JSON dict → numpy arrays (pandas C bridge, then discard) ─────────────────

def _is_numeric_string(s: str) -> bool:
    """True if s represents a number (int or float, including French format)."""
    try:
        float(s.replace(',', '.').replace(' ', '').replace(' ', ''))
        return True
    except (ValueError, AttributeError):
        return False


def _rows_to_col_data(rows: list[dict]) -> dict[str, np.ndarray]:
    """
    Transpose list[dict] to {col: np.ndarray} via a pandas DataFrame.

    Why pandas here: pd.DataFrame(list_of_dicts) is implemented in C
    (pandas/_libs/lib.pyx) and traverses all dicts in a single C-level loop.
    No pure-Python alternative matches its speed for this input shape.
    The DataFrame is created, converted to numpy, then immediately discarded.
    pandas is NOT imported anywhere else in this file.

    Embedded header row detection
    ------------------------------
    Some exports (datasud.fr, Opendatasoft) embed a row where every value is
    the column's display name (e.g. AGRICULTURE→'CONSOA', ANNEE→'ANNEE').
    This makes _infer_numpy_col see a string as the first element and
    classifies numeric columns as string.  We detect and strip this row:
    a row is considered a header row if every non-null value is a string
    AND at least one value matches its key (i.e. is the column name itself).
    """
    if not rows:
        return {}

    # ── Strip embedded header row ─────────────────────────────────────────────
    # Some exports embed a row where every string value equals its column name.
    # Detection: check only the string values in the first row.
    first = rows[0]
    if isinstance(first, dict) and len(rows) > 1:
        str_vals = {k: v for k, v in first.items()
                    if isinstance(v, str) and v is not None}
        if str_vals:
            # Hallmark: at least one string value matches its own column key
            key_match = any(v == k or v == k.upper() or v == k.lower()
                            for k, v in str_vals.items())
            # Secondary: ALL string values are non-numeric (pure label strings)
            all_label = all(not _is_numeric_string(v) for v in str_vals.values())
            if key_match and all_label:
                rows = rows[1:]

    if not rows:
        return {}

    import pandas as pd
    df       = pd.DataFrame(rows)           # C-level dict→column transpose
    col_data = {}
    for col in df.columns:
        arr = df[col].to_numpy()            # zero-copy (or minimal copy)
        if arr.dtype.kind == 'O':
            col_data[str(col)] = _infer_numpy_col(arr)
        elif np.issubdtype(arr.dtype, np.integer):
            col_data[str(col)] = arr.astype(np.float64)
        elif np.issubdtype(arr.dtype, np.floating):
            col_data[str(col)] = arr.astype(np.float64)
        else:
            col_data[str(col)] = arr.astype(str)
    del df                                  # discard DataFrame immediately
    return col_data


# ── JSON / JSONL ──────────────────────────────────────────────────────────────

def _extract_rows(data) -> list[dict] | None:
    """Recognise all common JSON structures and return a list of row-dicts."""
    if isinstance(data, list):
        if not data:
            return []
        if isinstance(data[0], dict):
            return data
        return [{"value": v} for v in data]

    if isinstance(data, dict):
        # Opendatasoft / datasud.fr export formats
        for key in ("values", "results"):
            if key in data and isinstance(data[key], list) and data[key]:
                if isinstance(data[key][0], dict):
                    return data[key]
        # Classic Opendatasoft v1: {"records": [{"fields": {...}}, ...]}
        if "records" in data and isinstance(data["records"], list):
            first = data["records"][0] if data["records"] else {}
            if isinstance(first, dict) and "fields" in first:
                return [r.get("fields", {}) for r in data["records"]]
    return None


def _parse_json(raw: bytes) -> dict[str, np.ndarray]:
    utf8 = _to_utf8(raw)
    data = _json_loads(utf8)
    rows = _extract_rows(data)

    # Row-major formats → use pandas C bridge
    if rows is not None:
        if not rows:
            raise ValueError("JSON data array is empty.")
        return _rows_to_col_data(rows)

    # Column-major dict {"col": [v1, v2, ...]}
    if isinstance(data, dict):
        list_cols   = {k: v for k, v in data.items() if isinstance(v, list)}
        scalar_cols = {k: [v] for k, v in data.items()
                       if not isinstance(v, (list, dict))}
        if list_cols:
            return {str(k): _list_to_numpy(v) for k, v in list_cols.items()}
        if scalar_cols:
            return {str(k): _list_to_numpy(v) for k, v in scalar_cols.items()}

    raise ValueError(
        "Unrecognised JSON structure. "
        "Expected list of objects [{...}], column dict {\"col\": [...]}, "
        "or Opendatasoft export {\"values\": [{...}]}."
    )


def _parse_jsonl(raw: bytes) -> dict[str, np.ndarray]:
    text  = _to_utf8(raw).decode('utf-8', errors='replace')
    lines = [_json_loads(ln.encode()) for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("JSONL file is empty.")
    rows = lines if isinstance(lines[0], dict) else [{"value": v} for v in lines]
    return _rows_to_col_data(rows)


# ── CSV / TSV ─────────────────────────────────────────────────────────────────

def _parse_csv(raw: bytes, delim: str | None = None) -> dict[str, np.ndarray]:
    """
    CSV parser — column-major path, no pandas.
    csv.reader (C) → per-column list → vectorised numpy cast.
    """
    raw  = raw[3:] if raw.startswith(b'\xef\xbb\xbf') else raw
    text = raw.decode('utf-8', errors='replace').replace('\r\n', '\n')
    if delim is None:
        delim = '\t' if '\t' in text[:4096] else ','
    reader  = csv.reader(io.StringIO(text), delimiter=delim)
    headers = [h.strip().strip('"') for h in next(reader)]
    cols: list[list] = [[] for _ in headers]
    for row in reader:
        for j, val in enumerate(row[:len(headers)]):
            cols[j].append(val.strip())
    # Vectorised numpy cast — no pandas
    return {h: _list_to_numpy(c) for h, c in zip(headers, cols)}


# ── Excel / ODS (unavoidable pandas dependency) ───────────────────────────────

def _parse_excel(raw: bytes, engine: str) -> dict[str, np.ndarray]:
    """
    pandas read_excel is the only reliable cross-format Excel parser.
    DataFrame is discarded immediately after numpy extraction.
    """
    import pandas as pd
    df = pd.read_excel(io.BytesIO(raw), engine=engine)
    col_data = {}
    for col in df.columns:
        arr = df[col].to_numpy()
        if arr.dtype.kind == 'O':
            col_data[str(col)] = _infer_numpy_col(arr)
        elif np.issubdtype(arr.dtype, np.integer):
            col_data[str(col)] = arr.astype(np.float64)
        elif np.issubdtype(arr.dtype, np.floating):
            col_data[str(col)] = arr.astype(np.float64)
        else:
            col_data[str(col)] = arr.astype(str)
    del df
    return col_data

def _parse_xlsx(raw): return _parse_excel(raw, 'openpyxl')
def _parse_xls(raw):  return _parse_excel(raw, 'xlrd')
def _parse_ods(raw):  return _parse_excel(raw, 'odf')


# ── Parquet (pyarrow → numpy, zero pandas) ────────────────────────────────────

def _parse_parquet(raw: bytes) -> dict[str, np.ndarray]:
    """
    pyarrow columnar read → numpy arrays directly.
    No pandas involved.
    """
    import pyarrow.parquet as pq
    table    = pq.read_table(io.BytesIO(raw))
    col_data = {}
    for col in table.column_names:
        arr = table.column(col).to_pylist()
        col_data[col] = _list_to_numpy(arr)
    return col_data


# ── Registry ──────────────────────────────────────────────────────────────────

PARSERS: dict[str, callable] = {
    'csv':     _parse_csv,
    'txt':     _parse_csv,
    'tsv':     lambda r: _parse_csv(r, '\t'),
    'json':    _parse_json,
    'jsonl':   _parse_jsonl,
    'xlsx':    _parse_xlsx,
    'xlsm':    _parse_xlsx,
    'xls':     _parse_xls,
    'ods':     _parse_ods,
    'parquet': _parse_parquet,
}
SUPPORTED_EXTENSIONS = set(PARSERS.keys())
