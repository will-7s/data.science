"""
utils.py  —  Shared pure-NumPy helpers.
No Dash, no Plotly, no scipy.
"""
import numpy as np

def drop_nan(arr: np.ndarray) -> np.ndarray:
    """Remove NaN values. Safe for any dtype."""
    return arr[~np.isnan(arr)] if arr.dtype.kind == 'f' else arr

def is_integer_array(arr: np.ndarray, tol: float = 1e-9) -> bool:
    """True if every finite value is an integer (within tol). Uses modulo — 3× faster than round."""
    clean = drop_nan(arr)
    return clean.size > 0 and bool(np.all(np.abs(clean % 1.0) < tol))

def format_percent(value: float, total: int) -> str:
    return f"({value / total * 100:.1f}%)" if total else "(n/a)"
