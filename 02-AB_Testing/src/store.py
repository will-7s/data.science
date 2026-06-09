from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from src.parsers import _infer_numpy_col
from src.schema import ABTestSchema

_INT_KINDS = frozenset("iu")
_FLOAT_KIND = "f"
_STR_KINDS = frozenset("USOb")


def _classify_column(arr: np.ndarray) -> str:
    kind = arr.dtype.kind
    if kind in _STR_KINDS:
        return "categorical"
    if kind in _INT_KINDS:
        n_uniq = len(np.unique(arr))
        if n_uniq == 2 and set(np.unique(arr)) <= {0, 1}:
            return "numeric"
        if n_uniq <= 10 and n_uniq < max(0.5 * len(arr), 3):
            return "categorical"
        return "numeric"
    if kind == _FLOAT_KIND:
        clean = arr[~np.isnan(arr)]
        n_uniq = len(np.unique(clean))
        if n_uniq <= 10:
            return "categorical"
        return "numeric"
    return "categorical"


def _is_binary(arr: np.ndarray) -> bool:
    if arr.dtype.kind == "f":
        clean = arr[~np.isnan(arr)]
    else:
        clean = arr
    uniq = np.unique(clean)
    if len(uniq) == 2:
        return True
    return False


def _is_datetime_col(arr: np.ndarray) -> bool:
    if arr.dtype.kind in ("M",):
        return True
    if arr.dtype.kind in ("O", "U"):
        sample = [str(x) for x in arr[:100] if x is not None and str(x).strip()]
        if not sample:
            return False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pd.to_datetime(sample[:20])
            return True
        except Exception:
            return False
    return False


def _is_id_column(arr: np.ndarray, col_name: str) -> bool:
    keywords = ["id", "user", "uuid", "key", "code", "num"]
    name_lower = col_name.lower()
    if any(k in name_lower for k in keywords):
        return True
    n_uniq = len(np.unique(arr))
    return n_uniq > 0.85 * len(arr)


dataset: dict[str, np.ndarray] = {}
col_meta: dict[str, str] = {}
all_cols: list[str] = []
num_cols: list[str] = []
cat_cols: list[str] = []
binary_cols: list[str] = []
datetime_cols: list[str] = []
id_candidates: list[str] = []
target_candidates: list[str] = []
group_candidates: list[str] = []

schema = ABTestSchema()
_raw_df: pd.DataFrame | None = None


def reset(df: pd.DataFrame) -> None:
    global col_meta, all_cols, num_cols, cat_cols
    global binary_cols, datetime_cols, id_candidates
    global target_candidates, group_candidates, _raw_df

    _raw_df = df.copy()
    dataset.clear()
    col_meta.clear()
    all_cols.clear()
    num_cols.clear()
    cat_cols.clear()
    binary_cols.clear()
    datetime_cols.clear()
    id_candidates.clear()
    target_candidates.clear()
    group_candidates.clear()

    for col in df.columns:
        arr = df[col].to_numpy()
        if arr.dtype.kind == "O":
            arr = _infer_numpy_col(arr)
        dataset[col] = arr

        meta = _classify_column(arr)
        col_meta[col] = meta
        all_cols.append(col)

        if meta == "numeric":
            num_cols.append(col)
        else:
            cat_cols.append(col)

        if _is_datetime_col(arr):
            datetime_cols.append(col)

        if _is_binary(arr):
            binary_cols.append(col)

    for col in all_cols:
        arr = dataset[col]
        meta = col_meta[col]
        if meta == "numeric" and _is_binary(arr):
            target_candidates.append(col)
        elif meta == "categorical":
            uniq = np.unique(arr)
            if 2 <= len(uniq) <= 20:
                group_candidates.append(col)
        if _is_id_column(arr, col):
            id_candidates.append(col)

    schema.target_col = ""
    schema.group_col = ""
    schema.covariate_cols.clear()
    schema.time_col = ""
    schema.id_col = ""


def get_df() -> pd.DataFrame:
    return pd.DataFrame({col: dataset[col] for col in all_cols})


def get_preview(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({col: dataset[col][:n] for col in all_cols})
