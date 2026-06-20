import math
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


@dataclass
class SprtConfig:
    alpha: float = 0.05
    beta: float = 0.20
    effect_size: float = 0.05
    monitor_default: bool = False

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if not 0 < self.beta < 1:
            raise ValueError(f"beta must be in (0, 1), got {self.beta}")
        if not 0 < self.effect_size < 1:
            raise ValueError(
                f"effect_size must be in (0, 1), got {self.effect_size}"
            )


@dataclass
class ExportConfig:
    schema_version: str = "1.0"
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    default_filename: str = "ab-testing-report"


@dataclass
class ThemeConfig:
    default_theme: str = "light"
    respect_prefers_color_scheme: bool = True

    def __post_init__(self) -> None:
        if self.default_theme not in ("light", "dark"):
            raise ValueError(
                f"default_theme must be 'light' or 'dark', got {self.default_theme!r}"
            )


@dataclass
class AnalysisConfig:
    random_seed: int = 42
    confidence_level: float = 0.95
    bootstrap_iterations: int = 1_000
    bootstrap_max_sample: int = 10_000
    permutation_iterations: int = 1_000
    bayesian_simulations: int = 10_000
    bayesian_prior_alpha: float = 1.0
    bayesian_prior_beta: float = 1.0
    rope_lower: float = -0.005
    rope_upper: float = 0.005

    def __post_init__(self) -> None:
        if math.isnan(self.rope_lower) or math.isnan(self.rope_upper):
            raise ValueError("rope_lower and rope_upper must not be NaN")
        if self.rope_lower >= self.rope_upper:
            raise ValueError(
                f"rope_lower ({self.rope_lower}) must be < rope_upper ({self.rope_upper})"
            )
        if not 0 < self.confidence_level < 1:
            raise ValueError(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )


@dataclass
class AppConfig:
    sprt: SprtConfig = field(default_factory=SprtConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


config = AppConfig()

# ── Module-level aliases for backward compatibility ──────────────────────
ALPHA = 1.0 - config.analysis.confidence_level
BAYESIAN_PRIOR_ALPHA = config.analysis.bayesian_prior_alpha
BAYESIAN_PRIOR_BETA = config.analysis.bayesian_prior_beta
BAYESIAN_SIMULATIONS = config.analysis.bayesian_simulations
ROPE_LOWER = config.analysis.rope_lower
ROPE_UPPER = config.analysis.rope_upper
RANDOM_SEED = config.analysis.random_seed
BOOTSTRAP_ITERATIONS = config.analysis.bootstrap_iterations
BOOTSTRAP_MAX_SAMPLE = config.analysis.bootstrap_max_sample
PERMUTATION_ITERATIONS = config.analysis.permutation_iterations
REPORT_DIR = config.export.output_dir
