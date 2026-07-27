"""Typed configuration loading and validation.

Every user input for the pipeline lives in ``config.yaml``. Unknown keys are
rejected rather than silently ignored, so a typo in the config surfaces as an
error instead of as a silently wrong run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

DEFAULT_CONFIG_PATH = "config.yaml"

PRIMARY_CHOICES = ("ridge", "huber", "ransac", "robust")
SECONDARY_CHOICES = ("isolation_forest", "mahalanobis", "lof")
IDLE_CHOICES = ("min", "percentile")
MODE_CHOICES = ("fit", "load")
SCALE_CHOICES = ("trimmed_mad", "mad", "iqr", "std")


@dataclass
class PathsConfig:
    input_dir: str = "raw"
    power_file: str = "power.csv"
    vibration_file: str = "vibration.csv"
    production_log_file: str = "production_log.csv"
    output_dir: str = "output"
    job_summary_file: str = "job_summary.csv"
    reading_detail_file: str = "reading_detail.csv.gz"
    diagnostics_file: str = "diagnostics.json"
    model_dir: str = "models"


@dataclass
class CleaningConfig:
    power_min_kw: float = 0.0
    power_abs_max_kw: float = 500.0
    robust_outlier_iqr_multiple: float = 50.0
    drop_duplicate_readings: bool = True
    swap_reversed_job_times: bool = True
    impute_missing_quantity: bool = True
    min_job_duration_s: float = 1.0
    align_clocks: bool = True
    alignment_coarse_step_s: int = 3600
    alignment_coarse_range_h: int = 14
    alignment_fine_step_s: int = 60
    alignment_fine_window_s: int = 3600
    pair_tolerance_s: float = 1.0
    job_edge_trim_s: float = 0.0


@dataclass
class IdleConfig:
    method: str = "min"
    percentile: float = 5.0
    fallback_to_global: bool = True


@dataclass
class PrimaryConfig:
    model: str = "huber"
    target: str = "log_vibration_rms_g"
    numeric_features: List[str] = field(
        default_factory=lambda: ["log_net_power_kw", "quantity"])
    categorical_features: List[str] = field(
        default_factory=lambda: ["machine_id", "part_type"])
    interactions: List[List[str]] = field(default_factory=lambda: [
        ["log_net_power_kw", "machine_id"], ["log_net_power_kw", "part_type"]])
    sigma_multiple: float = 6.0
    one_sided: bool = True
    scale_estimator: str = "trimmed_mad"
    scale_trim_sigma: float = 3.0
    ridge_alpha: float = 1.0
    huber_epsilon: float = 1.35
    ransac_trials: int = 120
    cutting_power_floor_kw: float = 0.5


@dataclass
class SecondaryConfig:
    model: str = "mahalanobis"
    features: List[str] = field(default_factory=lambda: [
        "log_net_power_kw", "log_vib_per_kw", "peak_to_rms"])
    contamination: float = 0.15
    if_n_estimators: int = 100
    if_max_samples: int = 256
    lof_n_neighbors: int = 20
    lof_reference_size: int = 4000


@dataclass
class JobFlagConfig:
    min_anomalous_readings: int = 5
    min_anomalous_fraction: float = 0.05
    min_readings_for_verdict: int = 20
    score_quantile: float = 0.95


@dataclass
class RunConfig:
    random_state: int = 7
    mode: str = "fit"
    save_model: bool = True
    verbose: bool = True


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    idle: IdleConfig = field(default_factory=IdleConfig)
    primary: PrimaryConfig = field(default_factory=PrimaryConfig)
    secondary: SecondaryConfig = field(default_factory=SecondaryConfig)
    job_flag: JobFlagConfig = field(default_factory=JobFlagConfig)
    run: RunConfig = field(default_factory=RunConfig)
    base_dir: Path = field(default_factory=lambda: Path(".").resolve())

    # -- path helpers ------------------------------------------------------- #
    def resolve(self, value: str) -> Path:
        """Resolve a config path relative to the config file's directory."""
        path = Path(value)
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    @property
    def input_dir(self) -> Path:
        return self.resolve(self.paths.input_dir)

    @property
    def power_path(self) -> Path:
        return self.input_dir / self.paths.power_file

    @property
    def vibration_path(self) -> Path:
        return self.input_dir / self.paths.vibration_file

    @property
    def production_log_path(self) -> Path:
        return self.input_dir / self.paths.production_log_file

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.paths.output_dir)

    @property
    def model_dir(self) -> Path:
        return self.resolve(self.paths.model_dir)

    # -- serialisation / validation ---------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        out = {f.name: asdict(getattr(self, f.name))
               for f in fields(self) if f.name != "base_dir"}
        out["base_dir"] = str(self.base_dir)
        return out

    def validate(self) -> "Config":
        if self.primary.model not in PRIMARY_CHOICES:
            raise ValueError(f"primary.model must be one of {PRIMARY_CHOICES}, "
                             f"got {self.primary.model!r}")
        if self.secondary.model not in SECONDARY_CHOICES:
            raise ValueError(f"secondary.model must be one of {SECONDARY_CHOICES}, "
                             f"got {self.secondary.model!r}")
        if self.idle.method not in IDLE_CHOICES:
            raise ValueError(f"idle.method must be one of {IDLE_CHOICES}, got {self.idle.method!r}")
        if self.run.mode not in MODE_CHOICES:
            raise ValueError(f"run.mode must be one of {MODE_CHOICES}, got {self.run.mode!r}")
        if self.primary.scale_estimator not in SCALE_CHOICES:
            raise ValueError(f"primary.scale_estimator must be one of {SCALE_CHOICES}, "
                             f"got {self.primary.scale_estimator!r}")
        if not 0.0 < self.secondary.contamination < 0.5:
            raise ValueError("secondary.contamination must be in (0, 0.5); it is a prior on "
                             f"the affected fraction, got {self.secondary.contamination}")
        if self.primary.sigma_multiple <= 0:
            raise ValueError("primary.sigma_multiple must be positive")
        if self.primary.scale_trim_sigma <= 0:
            raise ValueError("primary.scale_trim_sigma must be positive")
        if not 0.0 < self.job_flag.min_anomalous_fraction <= 1.0:
            raise ValueError("job_flag.min_anomalous_fraction must be in (0, 1]")
        if not 0.0 < self.job_flag.score_quantile <= 1.0:
            raise ValueError("job_flag.score_quantile must be in (0, 1]")
        if not 0.0 <= self.idle.percentile <= 100.0:
            raise ValueError("idle.percentile must be in [0, 100]")
        for pair in self.primary.interactions:
            if len(pair) != 2:
                raise ValueError(f"primary.interactions entries must be [numeric, categorical] "
                                 f"pairs, got {pair!r}")
        return self


def _build(cls, raw: Optional[Dict[str, Any]]):
    """Instantiate a config dataclass, rejecting unknown keys."""
    if not raw:
        return cls()
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}. "
                         f"Valid keys: {sorted(known)}")
    return cls(**{k: v for k, v in raw.items() if v is not None})


def _apply_overrides(raw: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge CLI overrides into the raw mapping, ignoring None (i.e. unset flags)."""
    if not overrides:
        return raw
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in raw.items()}
    for section, values in overrides.items():
        if not isinstance(values, dict):
            if values is not None:
                merged[section] = values
            continue
        target = merged.setdefault(section, {})
        if not isinstance(target, dict):
            target = {}
            merged[section] = target
        for key, value in values.items():
            if value is not None:
                target[key] = value
    return merged


def load_config(path: str | Path = DEFAULT_CONFIG_PATH,
                overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load and validate the pipeline configuration.

    Relative paths inside the config are resolved against the config file's own
    directory, so the pipeline can be invoked from anywhere.
    """
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {config_path} must contain a YAML mapping")

    raw = _apply_overrides(raw, overrides)
    known_sections = {f.name for f in fields(Config)} - {"base_dir"}
    unknown = set(raw) - known_sections
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}. "
                         f"Valid sections: {sorted(known_sections)}")

    cfg = Config(
        paths=_build(PathsConfig, raw.get("paths")),
        cleaning=_build(CleaningConfig, raw.get("cleaning")),
        idle=_build(IdleConfig, raw.get("idle")),
        primary=_build(PrimaryConfig, raw.get("primary")),
        secondary=_build(SecondaryConfig, raw.get("secondary")),
        job_flag=_build(JobFlagConfig, raw.get("job_flag")),
        run=_build(RunConfig, raw.get("run")),
        base_dir=config_path.parent,
    )
    # "robust" is a friendly alias for the consensus (RANSAC) regressor.
    if cfg.primary.model == "robust":
        cfg.primary.model = "ransac"
    return cfg.validate()
