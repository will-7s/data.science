from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

_SUPPORTED = {
    ".csv": "csv",
    ".xlsx": "excel",
    ".json": "json",
    ".parquet": "parquet",
}
MAX_ROWS_BEFORE_SAMPLING = 100_000
SAMPLE_SIZE = 100_000
_ENCODING_CANDIDATES = ["utf-8", "latin-1", "cp1252", "utf-16"]


def _detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    fmt = _SUPPORTED.get(ext)
    if fmt is None:
        exts = ", ".join(sorted(_SUPPORTED))
        raise ValueError(f"Unsupported file extension: {ext}. Use: {exts}")
    return fmt


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in _ENCODING_CANDIDATES:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(
        f"Could not determine encoding for {path}. Tried: {', '.join(_ENCODING_CANDIDATES)}"
    )


def _read_file(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        if pq is None:
            raise ImportError("pyarrow is required for Parquet files. Install: pip install pyarrow")
        return pq.read_table(path).to_pandas()
    if fmt == "csv":
        return _read_csv(path)
    if fmt == "excel":
        return pd.read_excel(path, engine="openpyxl")
    if fmt == "json":
        try:
            return pd.read_json(path)
        except ValueError as e:
            if "Trailing data" in str(e):
                raise ValueError("JSON appears to be ndjson format. Use .jsonl instead.") from e
            if "If using all scalar values" in str(e):
                raise ValueError("JSON is a single object, not an array of records.") from e
            raise
    return pd.DataFrame()


def _maybe_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if len(df) > MAX_ROWS_BEFORE_SAMPLING:
        return df.sample(n=SAMPLE_SIZE, random_state=42), True
    return df, False


def _is_id_column(col_name: str) -> bool:
    name_lower = col_name.lower()
    id_keywords = ["id", "user", "timestamp", "date", "time", "key", "index", "uuid"]
    return any(kw in name_lower for kw in id_keywords)


def _best_target_column(df: pd.DataFrame) -> str | None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    candidates = [c for c in numeric_cols if not _is_id_column(c)] or list(numeric_cols)
    if not candidates:
        return None
    for c in candidates:
        if set(df[c].dropna().unique()).issubset({0, 1}):
            return c
    return candidates[0]


def _compute_overview(df: pd.DataFrame, target_col: str | None = None) -> dict:
    n_rows = len(df)
    n_cols = len(df.columns)
    column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_pct = {col: round(float(v) * 100, 2) for col, v in df.isna().mean().items()}

    focus = target_col if target_col is not None and target_col in df.columns else None
    if focus is None and n_rows > 0:
        focus = _best_target_column(df)
    target_distribution = None
    if focus is not None and pd.api.types.is_numeric_dtype(df[focus].dtype):
        values = df[focus].dropna()
        if len(values) > 0:
            target_distribution = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()) if len(values) > 1 else 0.0,
            }

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "column_types": column_types,
        "missing_pct": missing_pct,
        "target_distribution": target_distribution,
        "target_column": focus,
    }


def load_data(path: str | Path, target_col: str | None = None) -> dict:
    path = Path(path)
    if not path.exists():
        return {"result": None, "error": f"File not found: {path}"}
    if not path.is_file():
        return {"result": None, "error": f"Not a regular file: {path}"}
    try:
        fmt = _detect_format(path)
    except ValueError as e:
        return {"result": None, "error": str(e)}
    try:
        df = _read_file(path, fmt)
        if len(df) == 0:
            return {"result": None, "error": "File is empty"}
        sampled, is_sampled = _maybe_sample(df)
        overview = _compute_overview(sampled, target_col=target_col)
        return {
            "result": {"df": sampled, "overview": overview, "sampled": is_sampled},
            "error": None,
        }
    except ImportError as e:
        return {"result": None, "error": str(e)}
    except Exception as e:
        return {"result": None, "error": f"Failed to load file: {e}"}


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    control_value: str,
    time_col: str | None = None,
    covariate_cols: list[str] | None = None,
    cleaning_report: dict | None = None,
) -> dict:
    target = df[target_col].to_numpy(dtype=float)
    group = df[group_col].to_numpy()

    control_mask = group.astype(str) == str(control_value)
    treatment_mask = ~control_mask

    # Typage des colonnes — évite de tout convertir en numpy d'un coup.
    # Seules les colonnes catégorielles (utilisées par l'analyse segmentée)
    # sont converties en tableaux numpy, ce qui économise la RAM pour les
    # gros jeux de données avec nombreuses colonnes numériques.
    cols = df.columns
    dtypes = df.dtypes
    is_num = {c: pd.api.types.is_numeric_dtype(dtypes[c]) for c in cols}
    meta = {c: "numeric" if is_num[c] else "categorical" for c in cols}
    all_cols = {}
    for c in cols:
        if meta[c] == "categorical":
            all_cols[c] = df[c].to_numpy()

    time_arr = df[time_col].to_numpy() if time_col and time_col in df.columns else None

    covariates = None
    if covariate_cols:
        covariates = {}
        for col in covariate_cols:
            if col in is_num and is_num[col]:
                covariates[col] = df[col].to_numpy(dtype=float)

    is_binary = bool(np.all(np.isin(target[~np.isnan(target)], [0.0, 1.0])))

    return {
        "target": target,
        "group": group,
        "control_mask": control_mask,
        "treatment_mask": treatment_mask,
        "all": all_cols,
        "meta": meta,
        "time": time_arr,
        "covariates": covariates,
        "cleaning_report": cleaning_report,
        "is_binary": is_binary,
    }


def get_group_stats(prepared: dict) -> dict:
    target = prepared["target"]
    c_mask = prepared["control_mask"]
    t_mask = prepared["treatment_mask"]

    n_control = int(c_mask.sum())
    n_treatment = int(t_mask.sum())

    c_vals = target[c_mask]
    t_vals = target[t_mask]

    c_mean = float(np.nanmean(c_vals)) if n_control > 0 else 0.0
    t_mean = float(np.nanmean(t_vals)) if n_treatment > 0 else 0.0

    return {
        "n_control": n_control,
        "n_treatment": n_treatment,
        "conv_control": int(np.nansum(c_vals)) if n_control > 0 else 0,
        "conv_treatment": int(np.nansum(t_vals)) if n_treatment > 0 else 0,
        "absolute_diff": t_mean - c_mean,
        "control_rate": c_mean,
        "treatment_rate": t_mean,
    }
