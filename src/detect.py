"""The two-stage tool-wear detector.

Stage 1 (primary, physics-informed): regress vibration RMS on cutting power and
job context, then flag readings whose *residual* sits far above the fitted
healthy baseline. This is the 'high vibration for the power drawn' test, so a
heavy titanium cut is judged against other heavy titanium cuts.

Stage 2 (secondary, unsupervised): a multivariate outlier score on the raw
sensor space, which knows nothing about the regression.

A reading is called anomalous only when *both* stages agree, which suppresses
single-model artefacts at the cost of some sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import features as feat
from models import PRIMARY_MODELS, SECONDARY_MODELS, OneHotEncoder


@dataclass
class DetectorBundle:
    """Everything needed to score a new shift with an already-fitted detector."""

    primary_name: str
    secondary_name: str
    primary_model: Any
    secondary_model: Any
    encoder: OneHotEncoder
    feature_names: List[str]
    residual_center: float
    residual_scale: float
    sigma_multiple: float
    one_sided: bool
    primary_config: Dict[str, Any] = field(default_factory=dict)
    secondary_config: Dict[str, Any] = field(default_factory=dict)
    fit_metadata: Dict[str, Any] = field(default_factory=dict)


def robust_scale(residuals: np.ndarray, estimator: str = "trimmed_mad",
                 trim_sigma: float = 3.0, max_iter: int = 20) -> float:
    """Spread of the residuals, estimated so wear itself cannot inflate it.

    Using the sample standard deviation here would be circular: the chattering
    readings we are hunting would widen sigma and hide themselves. MAD and IQR
    are ~50% breakdown estimators and fix most of that.

    ``trimmed_mad`` (the default) goes one step further. A plain MAD is computed
    over healthy *and* worn readings together, so with ~13% of readings affected
    it still overstates the healthy scatter -- which raises the 6-sigma bar and
    makes the detector under-sensitive to mildly worn tools. We therefore iterate:
    estimate the scale, discard readings beyond ``trim_sigma``, re-estimate. This
    converges to the scatter of the *healthy* population, which is what the
    6-sigma rule is meant to be measured against.
    """
    r = residuals[np.isfinite(residuals)]
    if r.size == 0:
        return 1e-9
    if estimator == "std":
        return max(float(r.std()), 1e-9)
    if estimator == "iqr":
        q1, q3 = np.percentile(r, [25, 75])
        return max(float((q3 - q1) / 1.349), 1e-9)

    mad = max(float(1.4826 * np.median(np.abs(r - np.median(r)))), 1e-9)
    if estimator == "mad":
        return mad

    scale = mad
    centre = float(np.median(r))
    for _ in range(max_iter):
        keep = np.abs(r - centre) <= trim_sigma * scale
        if keep.sum() < max(30, 0.2 * r.size):
            break
        kept = r[keep]
        new_centre = float(np.median(kept))
        new_scale = max(float(1.4826 * np.median(np.abs(kept - new_centre))), 1e-9)
        if abs(new_scale - scale) <= 1e-12 * max(scale, 1.0):
            scale, centre = new_scale, new_centre
            break
        scale, centre = new_scale, new_centre
    return scale


def _instantiate_primary(cfg) -> Any:
    name = cfg.primary.model
    cls = PRIMARY_MODELS[name]
    if name == "ridge":
        return cls(alpha=cfg.primary.ridge_alpha)
    if name == "huber":
        return cls(epsilon=cfg.primary.huber_epsilon)
    return cls(n_trials=cfg.primary.ransac_trials, random_state=cfg.run.random_state)


def _instantiate_secondary(cfg) -> Any:
    name = cfg.secondary.model
    cls = SECONDARY_MODELS[name]
    s = cfg.secondary
    if name == "isolation_forest":
        return cls(n_estimators=s.if_n_estimators, max_samples=s.if_max_samples,
                   contamination=s.contamination, random_state=cfg.run.random_state)
    if name == "lof":
        return cls(n_neighbors=s.lof_n_neighbors, contamination=s.contamination,
                   reference_size=s.lof_reference_size, random_state=cfg.run.random_state)
    return cls(contamination=s.contamination, random_state=cfg.run.random_state)


def fit_detector(cutting: pd.DataFrame, cfg) -> DetectorBundle:
    p, s = cfg.primary, cfg.secondary

    X, names, encoder = feat.build_design_matrix(
        cutting, p.numeric_features, p.categorical_features, p.interactions
    )
    y = cutting[p.target].to_numpy(dtype=float)

    primary = _instantiate_primary(cfg).fit(X, y)
    residuals = y - primary.predict(X)
    center = float(np.median(residuals))
    scale = robust_scale(residuals - center, p.scale_estimator)

    secondary = _instantiate_secondary(cfg).fit(cutting[s.features].to_numpy(dtype=float))

    ss_res = float(np.sum((residuals - residuals.mean()) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    metadata = {
        "n_fit_readings": int(len(cutting)),
        "r2_in_sample": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "residual_center": center,
        "residual_scale": scale,
        "residual_scale_std": float(residuals.std()),
        "machines": sorted(cutting["machine_id"].unique().tolist()),
        "part_types": sorted(cutting["part_type"].dropna().unique().tolist()),
    }

    return DetectorBundle(
        primary_name=p.model,
        secondary_name=s.model,
        primary_model=primary,
        secondary_model=secondary,
        encoder=encoder,
        feature_names=names,
        residual_center=center,
        residual_scale=scale,
        sigma_multiple=p.sigma_multiple,
        one_sided=p.one_sided,
        primary_config={"target": p.target, "numeric": p.numeric_features,
                        "categorical": p.categorical_features, "interactions": p.interactions,
                        "scale_estimator": p.scale_estimator},
        secondary_config={"features": s.features, "contamination": s.contamination},
        fit_metadata=metadata,
    )


def score_readings(cutting: pd.DataFrame, bundle: DetectorBundle) -> pd.DataFrame:
    """Attach per-reading primary/secondary scores and the combined flag."""
    df = cutting.copy()
    pc = bundle.primary_config

    X, _, _ = feat.build_design_matrix(
        df, pc["numeric"], pc["categorical"], pc["interactions"], encoder=bundle.encoder
    )
    predicted = bundle.primary_model.predict(X)
    residual = df[pc["target"]].to_numpy(dtype=float) - predicted

    df["predicted_rms_g"] = predicted
    df["residual_g"] = residual
    df["residual_sigma"] = (residual - bundle.residual_center) / bundle.residual_scale

    if bundle.one_sided:
        # Only *excess* vibration indicates wear; unusually smooth cutting does not.
        df["primary_anomalous"] = df["residual_sigma"] >= bundle.sigma_multiple
        df["primary_score"] = df["residual_sigma"]
    else:
        df["primary_anomalous"] = df["residual_sigma"].abs() >= bundle.sigma_multiple
        df["primary_score"] = df["residual_sigma"].abs()

    sec_features = df[bundle.secondary_config["features"]].to_numpy(dtype=float)
    df["secondary_score"] = bundle.secondary_model.score_samples(sec_features)
    df["secondary_anomalous"] = df["secondary_score"] >= bundle.secondary_model.threshold_

    df["anomalous"] = df["primary_anomalous"] & df["secondary_anomalous"]
    return df


def aggregate_reading_flags(scored: pd.DataFrame, cfg) -> pd.DataFrame:
    """Roll reading-level flags and scores up to one row per job."""
    q = cfg.job_flag.score_quantile

    grouped = scored.groupby("job_id", observed=True)
    out = grouped.agg(
        cutting_readings=("primary_score", "size"),
        wear_score=("primary_score", lambda s: float(np.quantile(s, q))),
        max_primary_sigma=("primary_score", "max"),
        mean_primary_sigma=("primary_score", "mean"),
        median_primary_sigma=("primary_score", "median"),
        n_primary_anomalous=("primary_anomalous", "sum"),
        n_secondary_anomalous=("secondary_anomalous", "sum"),
        n_anomalous=("anomalous", "sum"),
        mean_residual_g=("residual_g", "mean"),
        mean_vib_per_kw=("vib_per_kw", "mean"),
    ).reset_index()

    for col in ("n_primary_anomalous", "n_secondary_anomalous", "n_anomalous"):
        out[col] = out[col].astype(int)
    out["anomalous_fraction"] = out["n_anomalous"] / out["cutting_readings"].clip(lower=1)
    return out


def flag_jobs(jobs: pd.DataFrame, cfg) -> pd.DataFrame:
    """Final per-job verdict, rank, and human-readable notes."""
    jf = cfg.job_flag
    df = jobs.copy()

    df["cutting_readings"] = df["cutting_readings"].fillna(0).astype(int)
    df["n_anomalous"] = df["n_anomalous"].fillna(0).astype(int)
    df["anomalous_fraction"] = df["anomalous_fraction"].fillna(0.0)

    assessable = df["cutting_readings"] >= jf.min_readings_for_verdict
    df["assessable"] = assessable
    df["flagged"] = (
        assessable
        & (df["n_anomalous"] >= jf.min_anomalous_readings)
        & (df["anomalous_fraction"] >= jf.min_anomalous_fraction)
    )

    # Rank on the primary detector score only, as specified in the brief.
    # method="first" and na_option="bottom" give a strict 1..n ordering with no ties,
    # and push jobs that could not be assessed to the end rather than dropping them.
    df["rank"] = (
        df["wear_score"]
        .where(assessable)
        .rank(ascending=False, method="first", na_option="bottom")
        .astype(int)
    )

    notes: List[str] = []
    for row in df.itertuples():
        parts: List[str] = []
        existing = getattr(row, "data_quality_notes", "") or ""
        if existing:
            parts.append(str(existing))
        if not row.assessable:
            parts.append(f"only {row.cutting_readings} cutting readings; no verdict issued")
        if getattr(row, "power_readings", 1) == 0:
            parts.append("no power readings in window")
        if getattr(row, "vibration_readings", 1) == 0:
            parts.append("no vibration readings in window")
        if row.flagged:
            parts.append(
                f"{row.n_anomalous} of {row.cutting_readings} cutting readings "
                f"({row.anomalous_fraction:.1%}) exceed the {cfg.primary.sigma_multiple:g}-sigma "
                f"baseline and the secondary detector"
            )
        elif row.assessable and row.n_primary_anomalous > 0 and row.n_anomalous == 0:
            parts.append("primary detector fired but secondary did not; not flagged")
        notes.append("; ".join(parts))
    df["notes"] = notes
    return df


# The schema the brief asks for: required columns plus the encouraged optional ones.
# This is what output/job_summary.csv contains, so it is machine-readable exactly as
# specified. The wider table below is written alongside it for human inspection.
SPEC_COLUMNS = [
    "job_id", "machine_id", "mean_power_kw", "peak_vibration_g",
    "wear_score", "flagged", "rank", "notes",
]

OUTPUT_COLUMNS = [
    "job_id", "machine_id", "part_type", "operator", "quantity",
    "start_time", "end_time", "duration_s",
    "mean_power_kw", "peak_vibration_g", "wear_score", "flagged", "rank",
    "mean_power_raw_kw", "idle_baseline_kw", "mean_vibration_rms_g", "p95_vibration_rms_g",
    "cutting_readings", "n_primary_anomalous", "n_secondary_anomalous", "n_anomalous",
    "anomalous_fraction", "max_primary_sigma", "mean_primary_sigma", "mean_vib_per_kw",
    "power_readings", "vibration_readings", "assessable", "notes",
]


def build_output_table(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    cols = [c for c in (columns or OUTPUT_COLUMNS) if c in df.columns]
    out = df[cols].copy()
    for col in ("mean_power_kw", "peak_vibration_g", "wear_score", "mean_power_raw_kw",
                "idle_baseline_kw", "mean_vibration_rms_g", "p95_vibration_rms_g",
                "max_primary_sigma", "mean_primary_sigma", "mean_vib_per_kw", "anomalous_fraction"):
        if col in out.columns:
            out[col] = out[col].astype(float).round(4)
    if "flagged" in out.columns:
        out["flagged"] = out["flagged"].map({True: "true", False: "false"})
    return out.sort_values(["rank", "job_id"], na_position="last").reset_index(drop=True)
