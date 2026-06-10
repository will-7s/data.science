
from __future__ import annotations
import numpy as np

LAYOUT = dict(
    template="plotly_white",
    height=460,
    margin=dict(l=50, r=50, t=40, b=50),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif", size=12),
    hoverlabel=dict(font_family="Inter, -apple-system, sans-serif"),
)

BLUE = "#3498db"
RED = "#e74c3c"
GREEN = "#27ae60"
GREY = "#95a5a6"
ORANGE = "#e67e22"

CLUSTER_COLORS = [
    "#3498db","#e74c3c","#27ae60","#e67e22","#8e44ad",
    "#16a085","#d35400","#2980b9","#c0392b","#1abc9c",
    "#f39c12","#7f8c8d","#6c3483","#117864","#1a5276",
]

VAR_COLORS = [
    "#3498db","#e74c3c","#27ae60","#e67e22","#8e44ad",
    "#16a085","#d35400","#2980b9","#c0392b","#1abc9c",
    "#f39c12","#7f8c8d","#6c3483","#117864","#1a5276",
]


THETA = np.linspace(0, 2 * np.pi, 300)


def layout(**overrides):
    base = {k: v for k, v in LAYOUT.items() if k not in overrides}
    return {**base, **overrides}
