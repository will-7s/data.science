from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ABTestSchema:
    """Column-role mapping for a single A/B test dataset.

    Populated by resolve_column_names() and consumed by every analytical
    module.  The control_value field is new: it lets the user explicitly
    designate which group label is the baseline, avoiding the fragile
    alphabetical-first heuristic.
    """

    target_col: str = ""
    group_col: str = ""
    covariate_cols: list[str] = field(default_factory=list)
    time_col: str = ""
    id_col: str = ""
    control_value: str = ""   # FIX: explicit control label, not inferred

    def is_ready(self) -> bool:
        return bool(self.target_col) and bool(self.group_col)

    def to_dict(self) -> dict:
        return {
            "target_col": self.target_col,
            "group_col": self.group_col,
            "covariate_cols": list(self.covariate_cols),
            "time_col": self.time_col,
            "id_col": self.id_col,
            "control_value": self.control_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ABTestSchema":
        return cls(
            target_col=d.get("target_col", ""),
            group_col=d.get("group_col", ""),
            covariate_cols=list(d.get("covariate_cols", [])),
            time_col=d.get("time_col", ""),
            id_col=d.get("id_col", ""),
            control_value=d.get("control_value", ""),
        )

    def description(self) -> str:
        parts = [
            f"Target: {self.target_col}",
            f"Group: {self.group_col}",
            f"Control: '{self.control_value}'",
        ]
        if self.covariate_cols:
            parts.append(f"Covariates: {', '.join(self.covariate_cols)}")
        if self.time_col:
            parts.append(f"Time: {self.time_col}")
        if self.id_col:
            parts.append(f"ID: {self.id_col}")
        return " | ".join(parts)
