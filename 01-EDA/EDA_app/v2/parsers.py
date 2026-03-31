"""
parsers.py
──────────
One pure function per file format.

Contract
--------
Every parser receives raw bytes and returns:
    dict[str, np.ndarray]   (column name → 1-D array, dtype float64 or str)

Raising any exception is fine — the caller (loader.py) wraps everything
in a try/except and surfaces a human-readable error message.

Adding a new format
-------------------
1. Write `_parse_<ext>(raw: bytes) -> dict`.
2. Register it in PARSERS at the bottom of this file.
That's it — no other file needs to change.
"""

import csv
import io
import json
from datetime import date, datetime

import numpy as np


# ── shared helper ─────────────────────────────────────────────────────────────

def _to_numpy_col(values: list) -> np.ndarray:
    """
    Convert a Python list to a typed NumPy column.

    Priority
    --------
    1. date / datetime  →  ISO-8601 string  (kept as categorical)
    2. None / NaN       →  np.nan           (so float columns stay float)
    3. float64          →  numeric
    4. str              →  categorical fallback
    """
    normalised = []
    for v in values:
        if isinstance(v, (date, datetime)):
            normalised.append(v.isoformat())     # '2012-09-04' — stays str
        elif v is None:
            normalised.append(np.nan)            # will enable float cast
        else:
            normalised.append(v)

    arr = np.asarray(normalised)
    try:
        return arr.astype(np.float64)
    except (ValueError, TypeError):
        return arr.astype(str)


# ── text formats ──────────────────────────────────────────────────────────────

def _parse_csv(raw: bytes, delimiter: str | None = None) -> dict[str, np.ndarray]:
    """
    Parse CSV (or any delimiter-separated text).

    Robustness features
    -------------------
    - Strips UTF-8 BOM (\\xef\\xbb\\xbf) that Excel adds when saving as CSV.
    - Normalises CRLF → LF line endings.
    - Auto-detects delimiter (,  ;  TAB  |) when not provided.
    - Reads genfromtxt's structured array by field name (not column index)
      to avoid the IndexError that occurs on mixed-type files.
    """
    # 1. Strip BOM and normalise line endings
    if raw.startswith(b'\xef\xbb\xbf'):
        raw = raw[3:]
    text = raw.decode('utf-8', errors='replace').replace('\r\n', '\n').replace('\r', '\n')

    # 2. Auto-detect delimiter if not specified
    if delimiter is None:
        try:
            dialect   = csv.Sniffer().sniff(text[:4096], delimiters=',;\t|')
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ','

    # 3. Parse header
    reader  = io.StringIO(text)
    headers = [h.strip().strip('"').strip()
               for h in reader.readline().strip().split(delimiter)]

    # 4. Parse body — genfromtxt returns a structured 1-D array for mixed types
    body = np.genfromtxt(
        io.StringIO(reader.read()),
        delimiter=delimiter,
        dtype=None,        # auto-infer per-column types → structured array
        encoding='utf-8',
        names=None,        # we supply our own header names
    )

    if body.size == 0:
        raise ValueError("File has a header but no data rows.")

    # 5. Map our headers to structured array fields (named f0, f1, …)
    field_names = body.dtype.names   # ('f0', 'f1', ...) or None if homogeneous

    result: dict[str, np.ndarray] = {}

    if field_names:
        # Typical case: mixed types → structured array
        for i, col_name in enumerate(headers):
            col = body[field_names[i]]
            result[col_name] = _to_numpy_col(col.tolist())
    else:
        # Homogeneous file (all numeric or all strings) → 2-D array
        if body.ndim == 1:
            body = body.reshape(1, -1)
        for i, col_name in enumerate(headers):
            result[col_name] = _to_numpy_col(body[:, i].tolist())

    return result


def _parse_tsv(raw: bytes) -> dict[str, np.ndarray]:
    return _parse_csv(raw, delimiter='\t')


def _parse_json(raw: bytes) -> dict[str, np.ndarray]:
    data = json.loads(raw.decode('utf-8', errors='replace'))

    if isinstance(data, list) and data and isinstance(data[0], dict):
        # [{col: val, ...}, ...]  — most common export shape
        keys = list(data[0].keys())
        return {k: _to_numpy_col([row.get(k) for row in data]) for k in keys}

    if isinstance(data, dict):
        # {col: [val, ...], ...}  — columnar JSON
        return {k: _to_numpy_col(v) for k, v in data.items()}

    raise ValueError(
        "JSON must be a list of objects [{...}, ...] or a column dict {col: [...], ...}."
    )


def _parse_jsonl(raw: bytes) -> dict[str, np.ndarray]:
    rows = [
        json.loads(line)
        for line in raw.decode('utf-8', errors='replace').splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("JSONL file is empty.")
    return _parse_json(json.dumps(rows).encode())


# ── binary formats ────────────────────────────────────────────────────────────
#
# Each parser tries its preferred library first, then falls back gracefully.
# A missing optional dependency raises a clear ImportError with install hints
# rather than a cryptic module-not-found traceback.

def _require(package: str, install_name: str | None = None) -> object:
    """
    Import *package* and return the module.
    On failure, raise ImportError with a pip install hint.
    """
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        pip_name = install_name or package
        raise ImportError(
            f"Missing optional dependency '{package}'. "
            f"Install it with:  pip install {pip_name}"
        )


def _parse_xlsx_or_xlsm(raw: bytes) -> dict[str, np.ndarray]:
    """
    Read .xlsx / .xlsm — tries openpyxl first, falls back to pandas.

    openpyxl returns native Python types per cell (int, float, str, date,
    datetime, None) which _to_numpy_col() handles cleanly, including dates.
    The pandas fallback covers environments where openpyxl is absent.
    """
    # ── primary: openpyxl (no pandas needed) ─────────────────────────────
    try:
        openpyxl = _require('openpyxl')
    except ImportError:
        openpyxl = None

    if openpyxl is not None:
        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
        ws = wb.active
        all_rows = list(ws.iter_rows(values_only=True))
        if not all_rows:
            raise ValueError("Workbook active sheet appears empty.")
        headers   = [str(h) for h in all_rows[0]]
        col_lists: dict[str, list] = {h: [] for h in headers}
        for row in all_rows[1:]:
            for h, v in zip(headers, row):
                col_lists[h].append(v)
        return {col: _to_numpy_col(vals) for col, vals in col_lists.items()}

    # ── fallback: pandas + openpyxl engine (pandas will surface the error) ─
    pd = _require('pandas')
    df = pd.read_excel(io.BytesIO(raw), engine='openpyxl')
    return {str(col): _to_numpy_col(df[col].tolist()) for col in df.columns}


def _parse_xls(raw: bytes) -> dict[str, np.ndarray]:
    """
    Read legacy .xls — requires xlrd (pandas is used as the I/O bridge).
    xlrd only supports the old binary format; openpyxl cannot read .xls.
    """
    pd   = _require('pandas')
    _    = _require('xlrd')           # explicit check so the error is clear
    df   = pd.read_excel(io.BytesIO(raw), engine='xlrd')
    return {str(col): _to_numpy_col(df[col].tolist()) for col in df.columns}


def _parse_ods(raw: bytes) -> dict[str, np.ndarray]:
    """
    Read OpenDocument Spreadsheet (.ods) — requires odfpy.
    """
    pd   = _require('pandas')
    _    = _require('odf', install_name='odfpy')   # explicit check
    df   = pd.read_excel(io.BytesIO(raw), engine='odf')
    return {str(col): _to_numpy_col(df[col].tolist()) for col in df.columns}


def _parse_parquet(raw: bytes) -> dict[str, np.ndarray]:
    """
    Read Parquet — requires pyarrow.
    """
    pq    = _require('pyarrow.parquet', install_name='pyarrow')
    table = pq.read_table(io.BytesIO(raw))
    return {col: _to_numpy_col(table.column(col).to_pylist())
            for col in table.column_names}


# ── registry ──────────────────────────────────────────────────────────────────
#
# Map file extension → parser function.
# To support a new format: add one entry here + write its _parse_* above.

PARSERS: dict[str, callable] = {
    'csv':     _parse_csv,
    'txt':     _parse_csv,     # assume delimiter-separated plain text
    'tsv':     _parse_tsv,
    'json':    _parse_json,
    'jsonl':   _parse_jsonl,
    'xlsx':    _parse_xlsx_or_xlsm,
    'xlsm':    _parse_xlsx_or_xlsm,
    'xls':     _parse_xls,
    'ods':     _parse_ods,
    'parquet': _parse_parquet,
}

SUPPORTED_EXTENSIONS = set(PARSERS.keys())
