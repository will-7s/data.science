from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ABTestSchema:
    target_col: str = ""
    group_col: str = ""
    covariate_cols: list[str] = field(default_factory=list)
    time_col: str = ""
    id_col: str = ""

    def is_ready(self) -> bool:
        return bool(self.target_col) and bool(self.group_col)

    def to_dict(self) -> dict:
        return {
            "target_col": self.target_col,
            "group_col": self.group_col,
            "covariate_cols": list(self.covariate_cols),
            "time_col": self.time_col,
            "id_col": self.id_col,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ABTestSchema":
        return cls(
            target_col=d.get("target_col", ""),
            group_col=d.get("group_col", ""),
            covariate_cols=list(d.get("covariate_cols", [])),
            time_col=d.get("time_col", ""),
            id_col=d.get("id_col", ""),
        )

    def description(self) -> str:
        parts = [
            f"Target: {self.target_col}",
            f"Group: {self.group_col}",
        ]
        if self.covariate_cols:
            parts.append(f"Covariates: {', '.join(self.covariate_cols)}")
        if self.time_col:
            parts.append(f"Time: {self.time_col}")
        if self.id_col:
            parts.append(f"ID: {self.id_col}")
        return " | ".join(parts)
