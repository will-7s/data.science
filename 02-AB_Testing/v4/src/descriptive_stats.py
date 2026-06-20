import numpy as np

PERCENTILES = [1, 5, 25, 50, 75, 95, 99]


def _result(value, error=None):
    return {"result": value if error is None else None, "error": error}


def compute_descriptive_stats(data, target_col, group_col):
    try:
        target = np.asarray(data[target_col], dtype=float)
        group = np.asarray(data[group_col])
    except Exception as e:
        return _result(None, f"column not found: {e}")

    if target.ndim != 1 or group.ndim != 1 or target.size == 0:
        return _result(None, "target and group must be non-empty 1D arrays")

    if target.size != group.size:
        return _result(None, "target and group must have same length")

    valid = ~np.isnan(target) if not np.issubdtype(group.dtype, np.number) else ~(np.isnan(target) | np.isnan(group.astype(float)))
    target_clean = target[valid]
    group_clean = group[valid]

    unique_groups = np.unique(group_clean)
    n_per_group = {}
    mean_per_group = {}
    std_per_group = {}
    conversion_rate_per_group = {}
    missing_by_group = {}

    for g in unique_groups:
        g_mask = group_clean == g
        vals = target_clean[g_mask]
        n = len(vals)
        n_per_group[str(g)] = int(n)
        mean_per_group[str(g)] = float(vals.mean()) if n > 0 else None
        std_per_group[str(g)] = float(vals.std(ddof=1)) if n > 1 else 0.0
        is_binary = np.all(np.isin(vals, [0, 1]))
        conversion_rate_per_group[str(g)] = float(vals.mean()) if n > 0 and is_binary else None

        g_mask_all = group == g
        missing_by_group[str(g)] = int(np.isnan(target[g_mask_all]).sum())

    if len(target_clean) == 0:
        return _result(None, "no valid data after removing NaN")

    distribution_summary = {
        "min": float(target_clean.min()),
        "max": float(target_clean.max()),
        "percentiles": {str(p): float(np.percentile(target_clean, p)) for p in PERCENTILES},
    }

    return _result(
        {
            "n_per_group": n_per_group,
            "mean_per_group": mean_per_group,
            "std_per_group": std_per_group,
            "conversion_rate_per_group": conversion_rate_per_group,
            "missing_by_group": missing_by_group,
            "distribution_summary": distribution_summary,
            "confidence_badge": None,
        }
    )
