from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from src.store import dataset, col_meta, schema


def load_data(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(path, low_memory=False, encoding_errors="replace")
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif ext == ".json":
            df = pd.read_json(path)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, low_memory=False, encoding_errors="replace")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")
    if df.empty:
        raise ValueError(f"File is empty: {path}")
    return df


def resolve_column_names(
    target_col: str = "",
    group_col: str = "",
    covariate_cols: list[str] = None,
    time_col: str = "",
    id_col: str = "",
) -> dict:
    from src.store import schema, all_cols, binary_cols, target_candidates, group_candidates, datetime_cols, id_candidates

    if not target_col:
        if target_candidates:
            target_col = target_candidates[0]
        elif binary_cols:
            target_col = binary_cols[0]
    if not group_col and group_candidates:
        group_col = group_candidates[0]
    if not id_col and id_candidates:
        id_col = id_candidates[0]
    if not time_col and datetime_cols:
        time_col = datetime_cols[0]

    # Validate: ensure selected columns exist
    for name, val in [("target", target_col), ("group", group_col),
                       ("time", time_col), ("id", id_col)]:
        if val and val not in all_cols:
            raise ValueError(f"{name} column '{val}' not found in dataset columns: {all_cols}")

    schema.target_col = target_col
    schema.group_col = group_col
    schema.covariate_cols.clear()
    if covariate_cols:
        schema.covariate_cols.extend(covariate_cols)
    schema.time_col = time_col
    schema.id_col = id_col

    return {
        "target_col": target_col,
        "group_col": group_col,
        "covariate_cols": list(covariate_cols or []),
        "time_col": time_col,
        "id_col": id_col,
    }


def prepare_data() -> dict:
    from src.store import dataset, col_meta, schema

    if not schema.is_ready():
        raise ValueError("Schema not configured. Set target_col and group_col first.")

    raw_target = dataset[schema.target_col]
    if raw_target.dtype.kind not in ('f', 'i', 'u'):
        raw_target = pd.to_numeric(pd.Series(raw_target), errors='coerce').values.astype(float)
    elif raw_target.dtype.kind in ('i', 'u'):
        raw_target = raw_target.astype(float)
    raw_target = np.nan_to_num(raw_target, nan=0.0)

    out = {
        "target": raw_target,
        "group": dataset[schema.group_col],
        "group_categories": np.unique(dataset[schema.group_col]),
    }

    if schema.time_col and schema.time_col in dataset:
        out["time"] = dataset[schema.time_col]
    if schema.id_col and schema.id_col in dataset:
        out["id"] = dataset[schema.id_col]

    out["covariates"] = {}
    for c in schema.covariate_cols:
        if c in dataset:
            out["covariates"][c] = dataset[c]

    ctrl_val = get_control_value()
    group_col_data = dataset[schema.group_col]
    if group_col_data.dtype.kind in ('f', 'i', 'u'):
        try:
            ctrl_val = float(ctrl_val)
            if np.isnan(ctrl_val):
                ctrl_val = str(group_col_data[0])
        except ValueError:
            pass
    control_mask = group_col_data == ctrl_val
    treatment_mask = ~control_mask

    out["control_mask"] = control_mask
    out["treatment_mask"] = treatment_mask
    out["all"] = dataset
    out["meta"] = col_meta

    return out


def get_group_stats(prepared: dict) -> dict:
    target = prepared["target"]
    c_mask = prepared["control_mask"]
    t_mask = prepared["treatment_mask"]

    n_c = int(c_mask.sum())
    n_t = int(t_mask.sum())
    conv_c = int(np.nansum(target[c_mask])) if n_c > 0 else 0
    conv_t = int(np.nansum(target[t_mask])) if n_t > 0 else 0
    rate_c = conv_c / n_c if n_c > 0 else 0.0
    rate_t = conv_t / n_t if n_t > 0 else 0.0

    return {
        "n_control": n_c,
        "n_treatment": n_t,
        "conv_control": conv_c,
        "conv_treatment": conv_t,
        "rate_control": rate_c,
        "rate_treatment": rate_t,
        "absolute_diff": rate_t - rate_c,
        "relative_diff": (rate_t - rate_c) / rate_c if rate_c > 0 else 0.0,
        "ratio": rate_t / rate_c if rate_c > 0 else 0.0,
    }


def get_control_value() -> str:
    from src.store import dataset, schema
    arr = dataset[schema.group_col]
    uniq = np.unique(arr)
    if len(uniq) == 0:
        return "control"
    val = uniq[0]
    if isinstance(val, (int, float, np.integer, np.floating)):
        return val
    return str(val)
