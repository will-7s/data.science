import numpy as np


def _infer_numpy_col(obj_arr: np.ndarray) -> np.ndarray:
    n = len(obj_arr)
    none_mask = np.equal(obj_arr, None)
    non_null = obj_arr[~none_mask]

    if non_null.size == 0:
        return np.full(n, np.nan, dtype=np.float64)

    first = non_null.flat[0]
    is_numeric_first = isinstance(first, (int, float)) and not isinstance(first, bool)

    if is_numeric_first:
        out = np.full(n, np.nan, dtype=np.float64)
        try:
            out[~none_mask] = non_null.astype(np.float64)
            return out
        except (ValueError, TypeError):
            pass

    try:
        out = np.full(n, np.nan, dtype=np.float64)
        out[~none_mask] = non_null.astype(np.float64)
        return out
    except (ValueError, TypeError):
        pass

    result = obj_arr.astype(str)
    result[none_mask] = ""
    return result
