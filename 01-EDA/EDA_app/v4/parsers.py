import csv
import io
import json
import numpy as np
from datetime import date, datetime

_MISSING = frozenset({'', 'na', 'n/a', 'null', 'none', 'nan'})

def _to_numpy_col(values):
    numeric = []
    is_num = True
    for v in values:
        if v is None or (isinstance(v, str) and v.strip().lower() in _MISSING):
            numeric.append(np.nan)
        elif isinstance(v, (int, float, bool)):
            numeric.append(float(v))
        elif isinstance(v, (date, datetime)):
            is_num = False
            break
        elif isinstance(v, str):
            try:
                numeric.append(float(v.strip()))
            except ValueError:
                is_num = False
                break
        else:
            is_num = False
            break
    if is_num and numeric:
        return np.array(numeric, dtype=np.float64)
    return np.array([str(v) if v is not None else '' for v in values], dtype=str)

def _parse_csv(raw, delim=None):
    raw = raw[3:] if raw.startswith(b'\xef\xbb\xbf') else raw
    text = raw.decode('utf-8', errors='replace').replace('\r\n', '\n')
    if delim is None:
        try:
            dialect = csv.Sniffer().sniff(text[:4096])
            delim = dialect.delimiter
        except:
            delim = ','
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    headers = [h.strip().strip('"') for h in next(reader)]
    cols = [[] for _ in headers]
    for row in reader:
        for j, val in enumerate(row[:len(headers)]):
            cols[j].append(val.strip())
    return {h: _to_numpy_col(c) for h, c in zip(headers, cols)}

def _parse_json(raw):
    data = json.loads(raw.decode())
    if isinstance(data, list) and data:
        keys = list(data[0].keys())
        return {k: _to_numpy_col([row.get(k) for row in data]) for k in keys}
    if isinstance(data, dict):
        return {k: _to_numpy_col(v) for k, v in data.items()}
    raise ValueError("JSON must be list of objects or column dict")

def _parse_jsonl(raw):
    lines = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    return _parse_json(json.dumps(lines).encode())

def _parse_excel(raw, engine):
    import pandas as pd
    df = pd.read_excel(io.BytesIO(raw), engine=engine)
    return {str(col): _to_numpy_col(df[col].tolist()) for col in df.columns}

def _parse_xlsx(raw): return _parse_excel(raw, 'openpyxl')
def _parse_xls(raw): return _parse_excel(raw, 'xlrd')
def _parse_ods(raw): return _parse_excel(raw, 'odf')
def _parse_parquet(raw):
    import pyarrow.parquet as pq
    table = pq.read_table(io.BytesIO(raw))
    return {col: _to_numpy_col(table.column(col).to_pylist()) for col in table.column_names}

PARSERS = {
    'csv': _parse_csv, 'txt': _parse_csv, 'tsv': lambda r: _parse_csv(r, '\t'),
    'json': _parse_json, 'jsonl': _parse_jsonl,
    'xlsx': _parse_xlsx, 'xlsm': _parse_xlsx, 'xls': _parse_xls, 'ods': _parse_ods,
    'parquet': _parse_parquet,
}
SUPPORTED_EXTENSIONS = set(PARSERS.keys())