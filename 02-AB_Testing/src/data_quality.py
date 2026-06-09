from __future__ import annotations
import numpy as np
from src.store import dataset, col_meta, schema, all_cols


def report_data_quality() -> dict:
    if not dataset or not all_cols:
        return {
            "total_rows": 0, "total_columns": 0, "schema": "",
            "missing_values": {}, "column_types": {},
            "column_names": [], "unique_counts": {},
        }

    n_rows = len(next(iter(dataset.values())))
    missing = {}
    for col in all_cols:
        if col not in dataset:
            continue
        arr = dataset[col]
        if arr.dtype.kind == "f":
            n_miss = int(np.isnan(arr).sum())
        else:
            n_miss = int((arr == "").sum()) if arr.dtype.kind in ("U", "O", "S") else 0
        if n_miss > 0:
            missing[col] = n_miss

    def _n_unique(arr):
        return int(len(np.unique(arr)))

    info = {
        "total_rows": n_rows,
        "total_columns": len(all_cols),
        "missing_values": missing,
        "column_types": dict(col_meta),
        "column_names": list(all_cols),
        "unique_counts": {col: _n_unique(dataset[col]) for col in all_cols if col in dataset},
    }

    if schema.is_ready() and schema.target_col in dataset and schema.group_col in dataset:
        info["schema"] = schema.description()
        target = dataset[schema.target_col]
        group = dataset[schema.group_col]
        group_counts = {str(k): int(v) for k, v in zip(*np.unique(group, return_counts=True))}
        info["group_distribution"] = group_counts
        if target.dtype.kind == "f":
            info["target_mean"] = float(np.nanmean(target))
            info["target_sum"] = int(np.nansum(target))

    return info
