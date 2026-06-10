"""
loader.py  —  Decode Dash upload → parse → validate → dedup → store.

All operations after parsing are pure numpy — no pandas.

Deduplication strategy
----------------------
Single numpy void-view + np.unique path for ALL column types:
  - float64 columns: stored as (value, nan_flag) pairs — NaN rows
    hash consistently (NaN != NaN in IEEE 754, but two uint8 flags compare equal).
  - str/object columns: stored as fixed-width numpy string fields.

np.unique on the void view is O(n log n) with a fully vectorised
radix/merge sort in C.  No Python per-row iteration.

This is strictly numpy — pandas is not imported here.
"""
from __future__ import annotations
import base64
import numpy as np
import store
from parsers import PARSERS, SUPPORTED_EXTENSIONS


def load(contents: str, filename: str) -> tuple[bool, str]:
    try:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return False, (
                f"Unsupported format. "
                f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        _, payload  = contents.split(",", 1)
        raw         = base64.b64decode(payload)
        col_data    = PARSERS[ext](raw)

        if not col_data:
            return False, "File appears empty."

        col_data, warning = _align_columns(col_data)
        if col_data is None:
            return False, warning

        col_data = _dedup_rows(col_data)
        store.reset(col_data)

        n_rows = len(next(iter(col_data.values())))
        msg    = f"✓ {filename}  —  {len(col_data)} cols · {n_rows:,} rows"
        if warning:
            msg += f"  ⚠ {warning}"
        return True, msg

    except ImportError as exc:
        pkg = str(exc).split("'")[1] if "'" in str(exc) else str(exc)
        return False, (
            f"Missing dependency — `Import {pkg}` failed. "
            f"Use pip or conda to install the {pkg} package."
        )
    except Exception as exc:
        return False, f"Error loading file: {exc}"


# ── Column alignment ──────────────────────────────────────────────────────────

def _align_columns(col_data: dict) -> tuple[dict | None, str]:
    """
    Ensure all columns have the same length.

    Uses numpy to find the majority length efficiently:
    - np.array of lengths → np.unique with counts → argmax in C.
    - Falls back to error message if no clear majority.
    """
    if not col_data:
        return None, "File appears empty."

    cols    = list(col_data.keys())
    lengths = np.array([len(col_data[c]) for c in cols], dtype=np.intp)

    unique_lens, inverse, counts = np.unique(
        lengths, return_inverse=True, return_counts=True
    )

    if len(unique_lens) == 1:
        return col_data, ""

    best_idx     = int(np.argmax(counts))
    majority_len = int(unique_lens[best_idx])
    majority_cnt = int(counts[best_idx])

    if majority_cnt < 2:
        info = ", ".join(f"{cols[i]!r}: {lengths[i]}" for i in range(len(cols)))
        return None, (
            f"Columns have incompatible lengths ({info}). "
            "All columns must have the same number of rows."
        )

    keep_mask = inverse == best_idx          # boolean mask, vectorised
    kept      = {cols[i]: col_data[cols[i]] for i in range(len(cols)) if keep_mask[i]}
    dropped   = [cols[i] for i in range(len(cols)) if not keep_mask[i]]

    warning = (
        f"Columns ignored (length mismatch): {', '.join(dropped)}. "
        f"Kept {len(kept)} columns × {majority_len:,} rows."
    )
    return kept, warning


# ── Deduplication — pure numpy, all dtypes ────────────────────────────────────

def _dedup_rows(col_data: dict) -> dict:
    """
    Remove duplicate rows via numpy void-view + np.unique.

    Works for any combination of float64 and string columns:
    - float64: (value_as_float64, nan_flag_uint8) pair per column
               so that NaN == NaN comparisons work correctly.
    - str/U*:  fixed-width numpy string field per column.

    The void view treats each row as an opaque byte sequence.
    np.unique sorts the byte sequences (O(n log n) in C) and returns
    the index of the first occurrence of each unique row.

    Original row order is preserved by np.sort(idx) after np.unique.
    """
    cols = list(col_data.keys())
    n    = len(col_data[cols[0]])

    if n <= 1:
        return col_data

    fl = [c for c in cols if col_data[c].dtype.kind == 'f']
    st = [c for c in cols if col_data[c].dtype.kind != 'f']

    # Build structured dtype
    fields: list[tuple] = []
    for c in fl:
        fields += [(f"_v_{c}", np.float64), (f"_n_{c}", np.uint8)]
    for c in st:
        fields.append((f"_s_{c}", col_data[c].dtype))

    struct = np.empty(n, dtype=fields)

    # Fill float columns: zero-out NaN values, store flag separately
    for c in fl:
        arr = col_data[c]
        nan = np.isnan(arr)
        struct[f"_v_{c}"] = np.where(nan, 0.0, arr)   # vectorised
        struct[f"_n_{c}"] = nan.view(np.uint8)          # bitcast, no copy

    # Fill string columns
    for c in st:
        struct[f"_s_{c}"] = col_data[c]

    # Void view: each row becomes an opaque byte string
    void_dt = np.dtype((np.void, struct.dtype.itemsize))
    _, idx  = np.unique(struct.view(void_dt).ravel(), return_index=True)
    idx     = np.sort(idx)          # restore original order (writable copy)

    if len(idx) == n:
        return col_data             # no duplicates found — return as-is

    return {c: col_data[c][idx] for c in cols}
