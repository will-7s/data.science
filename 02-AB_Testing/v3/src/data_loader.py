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
    control_value: str = "",
) -> dict:
    """Resolve and validate column mapping; persist into global schema.

    FIX: accepts explicit control_value so callers are not forced to rely on
    the fragile alphabetical-first heuristic in get_control_value().
    """
    from src.store import (
        schema, all_cols, binary_cols,
        target_candidates, group_candidates, datetime_cols, id_candidates,
        dataset,
    )

    if not target_col:
        target_col = target_candidates[0] if target_candidates else (binary_cols[0] if binary_cols else "")
    if not group_col and group_candidates:
        group_col = group_candidates[0]
    if not id_col and id_candidates:
        id_col = id_candidates[0]
    if not time_col and datetime_cols:
        time_col = datetime_cols[0]

    for role, val in [("target", target_col), ("group", group_col),
                      ("time", time_col), ("id", id_col)]:
        if val and val not in all_cols:
            raise ValueError(f"{role} column '{val}' not found in dataset columns: {all_cols}")

    # FIX: resolve control_value — use explicit arg, else alphabetical fallback
    if not control_value and group_col and group_col in dataset:
        control_value = str(np.unique(dataset[group_col])[0])

    schema.target_col = target_col
    schema.group_col = group_col
    schema.covariate_cols.clear()
    if covariate_cols:
        schema.covariate_cols.extend(covariate_cols)
    schema.time_col = time_col
    schema.id_col = id_col
    schema.control_value = control_value

    return {
        "target_col": target_col,
        "group_col": group_col,
        "covariate_cols": list(covariate_cols or []),
        "time_col": time_col,
        "id_col": id_col,
        "control_value": control_value,
    }


def clean_data(df: pd.DataFrame, group_col: str, page_col: str | None,
               id_col: str | None) -> tuple[pd.DataFrame, dict]:
    """FIX: remove mismatched group/page rows and duplicate IDs.

    Returns the cleaned DataFrame and a dict of quality metrics for the report.
    page_col and id_col are optional — if absent the corresponding cleaning
    step is skipped.
    """
    n_raw = len(df)
    n_mismatch = 0
    n_dupes = 0

    # Step 1 — remove group/landing_page mismatches if both columns exist
    if page_col and page_col in df.columns:
        groups = df[group_col].unique()
        pages  = df[page_col].unique()
        # Build expected mapping: sort both arrays and zip them (control→old, treatment→new)
        # Heuristic: group label containing "control" maps to page label NOT containing "new"
        def _is_control_group(v: str) -> bool:
            return "control" in str(v).lower()

        def _is_old_page(v: str) -> bool:
            return "old" in str(v).lower() or "control" in str(v).lower()

        # Only attempt mismatch removal when we can identify pairs unambiguously
        ctrl_groups  = [g for g in groups if _is_control_group(g)]
        treat_groups = [g for g in groups if not _is_control_group(g)]
        old_pages    = [p for p in pages if _is_old_page(p)]
        new_pages    = [p for p in pages if not _is_old_page(p)]

        if ctrl_groups and treat_groups and old_pages and new_pages:
            cg, tg = ctrl_groups[0], treat_groups[0]
            op, np_ = old_pages[0], new_pages[0]
            mismatch_mask = (
                ((df[group_col] == cg) & (df[page_col] == np_)) |
                ((df[group_col] == tg) & (df[page_col] == op))
            )
            n_mismatch = int(mismatch_mask.sum())
            df = df[~mismatch_mask].copy()

    # Step 2 — deduplicate on ID column
    if id_col and id_col in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset=id_col, keep="first").copy()
        n_dupes = n_before - len(df)

    metrics = {
        "n_raw": n_raw,
        "n_mismatch_removed": n_mismatch,
        "n_dupes_removed": n_dupes,
        "n_clean": len(df),
        "pct_removed": round((n_raw - len(df)) / max(n_raw, 1) * 100, 2),
    }
    return df, metrics


def prepare_data(page_col: str | None = None) -> dict:
    """Build the analysis-ready dict from the global store.

    FIX: now calls clean_data() to strip mismatches and duplicates before
    any statistical computation.  The cleaning report is attached under
    prepared['cleaning_report'].
    """
    from src.store import dataset, col_meta, schema

    if not schema.is_ready():
        raise ValueError("Schema not configured. Call resolve_column_names() first.")

    # Reconstruct a lightweight DataFrame for cleaning (only needed cols)
    needed = [c for c in [schema.target_col, schema.group_col,
                           schema.time_col, schema.id_col,
                           page_col] + list(schema.covariate_cols)
              if c and c in dataset]
    df_clean = pd.DataFrame({c: dataset[c] for c in needed})

    df_clean, cleaning = clean_data(
        df_clean,
        group_col=schema.group_col,
        page_col=page_col,
        id_col=schema.id_col if schema.id_col else None,
    )

    # Re-extract arrays from cleaned DataFrame
    def _to_float(series: pd.Series) -> np.ndarray:
        if series.dtype.kind not in ("f", "i", "u"):
            series = pd.to_numeric(series, errors="coerce")
        return np.nan_to_num(series.to_numpy(dtype=float), nan=0.0)

    raw_target = _to_float(df_clean[schema.target_col])
    group_arr  = df_clean[schema.group_col].to_numpy()

    ctrl_val = schema.control_value or str(np.unique(group_arr)[0])
    control_mask   = group_arr == ctrl_val
    treatment_mask = ~control_mask

    out: dict = {
        "target":          raw_target,
        "group":           group_arr,
        "group_categories": np.unique(group_arr),
        "control_mask":    control_mask,
        "treatment_mask":  treatment_mask,
        "cleaning_report": cleaning,
        "all":             {c: df_clean[c].to_numpy() for c in df_clean.columns},
        "meta":            col_meta,
    }

    if schema.time_col and schema.time_col in df_clean.columns:
        out["time"] = df_clean[schema.time_col].to_numpy()
    if schema.id_col and schema.id_col in df_clean.columns:
        out["id"] = df_clean[schema.id_col].to_numpy()

    out["covariates"] = {}
    for c in schema.covariate_cols:
        if c in df_clean.columns:
            out["covariates"][c] = df_clean[c].to_numpy()

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
        "n_control":    n_c,
        "n_treatment":  n_t,
        "conv_control": conv_c,
        "conv_treatment": conv_t,
        "rate_control": rate_c,
        "rate_treatment": rate_t,
        "absolute_diff": rate_t - rate_c,
        "relative_diff": (rate_t - rate_c) / rate_c if rate_c > 0 else 0.0,
        "ratio": rate_t / rate_c if rate_c > 0 else 0.0,
    }


def get_control_value() -> str:
    """Return the designated control group label from schema (never guesses)."""
    from src.store import schema
    return schema.control_value or ""
