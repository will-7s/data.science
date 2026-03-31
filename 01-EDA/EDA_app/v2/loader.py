"""
loader.py
─────────
Bridges the Dash Upload component and the rest of the app.

Responsibilities
----------------
1. Decode the base64 payload coming from dcc.Upload.
2. Route to the right parser based on file extension.
3. Deduplicate rows (pure NumPy, no pandas).
4. Commit the clean dataset to `store`.

Returns a (success: bool, message: str) tuple — the UI just displays it.
"""

import base64
import numpy as np

import store
from parsers import PARSERS, SUPPORTED_EXTENSIONS


def load(contents: str, filename: str) -> tuple[bool, str]:
    """
    Entry point called by the Dash upload callback.

    Parameters
    ----------
    contents : base64-encoded data URI from dcc.Upload
    filename : original filename (used to detect format)

    Returns
    -------
    (True,  success message)
    (False, error message)
    """
    try:
        ext = filename.rsplit('.', 1)[-1].lower()

        if ext not in SUPPORTED_EXTENSIONS:
            accepted = ', '.join(sorted(SUPPORTED_EXTENSIONS))
            return False, f"Unsupported format '.{ext}'. Accepted: {accepted}"

        # ── decode ────────────────────────────────────────────────────────
        _header, payload = contents.split(',', 1)
        raw: bytes = base64.b64decode(payload)

        # ── parse ─────────────────────────────────────────────────────────
        col_data = PARSERS[ext](raw)

        if not col_data:
            return False, "File appears empty or could not be parsed."

        # ── deduplicate rows ──────────────────────────────────────────────
        col_data = _drop_duplicates(col_data)

        # ── commit to shared store ────────────────────────────────────────
        store.reset(col_data)

        n_rows = len(next(iter(col_data.values())))
        n_cols = len(col_data)
        return True, f"✓ {filename}  —  {n_cols} columns · {n_rows} rows"

    except ImportError as exc:
        return False, f"Missing dependency — {exc}"
    except Exception as exc:
        return False, f"Error loading file: {exc}"


# ── private helpers ───────────────────────────────────────────────────────────

def _drop_duplicates(col_data: dict) -> dict:
    """
    Remove duplicate rows without pandas.

    Strategy: convert each row to a tuple, track seen tuples with a set.
    O(n) time, O(n) memory — fast enough for typical EDA dataset sizes.
    """
    cols = list(col_data.keys())
    n    = len(col_data[cols[0]])

    seen: set  = set()
    keep: list = []

    for i in range(n):
        row = tuple(col_data[c][i] for c in cols)
        if row not in seen:
            seen.add(row)
            keep.append(i)

    if len(keep) == n:
        return col_data   # nothing removed — return as-is

    idx = np.array(keep)
    return {c: col_data[c][idx] for c in cols}
