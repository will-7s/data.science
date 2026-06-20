from __future__ import annotations
import numpy as np
from src.store import dataset, col_meta, schema, all_cols


def report_data_quality(cleaning_report: dict | None = None) -> dict:
    """Build a data quality report from the current store state.

    FIX: accepts an optional cleaning_report dict (produced by prepare_data)
    so callers can surface mismatch/duplicate counts in the UI and text report.
    """
    if not dataset or not all_cols:
        return {
            "total_rows": 0, "total_columns": 0, "schema": "",
            "missing_values": {}, "column_types": {},
            "column_names": [], "unique_counts": {},
        }

    n_rows = len(next(iter(dataset.values())))

    missing: dict[str, int] = {}
    for col in all_cols:
        if col not in dataset:
            continue
        arr = dataset[col]
        if arr.dtype.kind == "f":
            n_miss = int(np.isnan(arr).sum())
        elif arr.dtype.kind in ("U", "O", "S"):
            n_miss = int((arr == "").sum())
        else:
            n_miss = 0
        if n_miss > 0:
            missing[col] = n_miss

    info: dict = {
        "total_rows":    n_rows,
        "total_columns": len(all_cols),
        "missing_values": missing,
        "column_types":  dict(col_meta),
        "column_names":  list(all_cols),
        "unique_counts": {
            col: int(len(np.unique(dataset[col])))
            for col in all_cols if col in dataset
        },
    }

    if cleaning_report:
        info["cleaning"] = cleaning_report

    if schema.is_ready() and schema.target_col in dataset and schema.group_col in dataset:
        info["schema"] = schema.description()
        target = dataset[schema.target_col]
        group  = dataset[schema.group_col]
        info["group_distribution"] = {
            str(k): int(v)
            for k, v in zip(*np.unique(group, return_counts=True))
        }
        if target.dtype.kind == "f":
            info["target_mean"] = float(np.nanmean(target))
            info["target_sum"]  = int(np.nansum(target))

    return info
