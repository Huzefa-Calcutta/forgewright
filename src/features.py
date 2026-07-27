"""Feature construction for the detectors and per-job aggregation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from models import OneHotEncoder


def build_reading_table(
    vibration_with_power: pd.DataFrame,
    production_log: pd.DataFrame,
    idle_baseline: pd.Series,
) -> pd.DataFrame:
    """One row per in-job vibration reading, joined to its power and job context."""
    df = vibration_with_power[vibration_with_power["job_id"].notna()].copy()
    df = df[df["power_kw"].notna()].copy()

    job_cols = ["job_id", "part_type", "operator", "quantity", "duration_s"]
    df = df.merge(production_log[job_cols], on="job_id", how="left", validate="many_to_one")

    df["idle_baseline_kw"] = df["machine_id"].map(idle_baseline).astype(float)
    df["net_power_kw"] = (df["power_kw"] - df["idle_baseline_kw"]).clip(lower=0.0)
    # Vibration per unit of cutting work -- the raw physical ratio behind the
    # whole exercise. Guarded against division by ~0 during idle stretches.
    df["vib_per_kw"] = df["vibration_rms_g"] / df["net_power_kw"].clip(lower=0.05)
    df["peak_to_rms"] = df["vibration_peak_g"] / df["vibration_rms_g"].clip(lower=1e-6)

    # Log-space companions. Vibration scales multiplicatively with cutting power
    # (see notebooks/01_eda.ipynb), so modelling in logs makes the error term
    # additive and roughly homoscedastic, and makes a residual directly
    # interpretable as "x% more vibration than a healthy tool would give".
    df["log_net_power_kw"] = np.log(df["net_power_kw"].clip(lower=0.05))
    df["log_vibration_rms_g"] = np.log(df["vibration_rms_g"].clip(lower=1e-4))
    df["log_vib_per_kw"] = df["log_vibration_rms_g"] - df["log_net_power_kw"]
    return df.reset_index(drop=True)


def select_cutting_readings(readings: pd.DataFrame, power_floor_kw: float) -> pd.DataFrame:
    """Keep readings where the machine is actually cutting.

    A job window includes short air-cut / repositioning moments at near-idle
    power. Vibration-per-power is meaningless there and would dominate the
    residual distribution, so those readings are excluded from the model.
    """
    return readings[readings["net_power_kw"] >= power_floor_kw].copy()


def build_design_matrix(
    readings: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    interactions: List[List[str]],
    encoder: OneHotEncoder | None = None,
) -> Tuple[np.ndarray, List[str], OneHotEncoder]:
    """Assemble [numeric | one-hot | numeric x one-hot] features.

    The interaction terms let each machine and each material have its own
    vibration-vs-power slope, which is what stops a legitimately aggressive
    titanium cut from looking like chatter.
    """
    numeric = readings[numeric_features].astype(float)
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    blocks = [numeric.to_numpy()]
    names = list(numeric_features)

    cat_frame = readings[categorical_features] if categorical_features else readings[[]]
    if encoder is None:
        encoder = OneHotEncoder(drop_first=True).fit(cat_frame)
    cat_matrix = encoder.transform(cat_frame)
    if cat_matrix.shape[1]:
        blocks.append(cat_matrix)
        names += list(encoder.feature_names_)

    # Column span of each categorical variable's one-hot block, recorded in the
    # same order the encoder stacks them, so interaction terms slice the right
    # columns even when a variable has no levels left after drop_first.
    spans: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for col, cats in encoder.categories_.items():
        spans[col] = (cursor, cursor + len(cats))
        cursor += len(cats)

    for num_col, cat_col in interactions or []:
        if num_col not in numeric.columns or cat_col not in spans:
            continue
        start, stop = spans[cat_col]
        if start == stop:
            continue
        block = cat_matrix[:, start:stop] * numeric[num_col].to_numpy()[:, None]
        blocks.append(block)
        names += [f"{num_col}*{cat_col}={c}" for c in encoder.categories_[cat_col]]

    X = np.hstack(blocks)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names, encoder


def aggregate_jobs(
    production_log: pd.DataFrame,
    power_with_jobs: pd.DataFrame,
    vibration_with_jobs: pd.DataFrame,
    idle_baseline: pd.Series,
) -> pd.DataFrame:
    """Per-job power and vibration metrics.

    ``mean_power_kw`` follows the brief: the mean of the job's power readings
    minus that machine's idle-minimum draw, i.e. the net power actually spent
    cutting. ``peak_vibration_g`` is the max peak-amplitude reading in the job.
    """
    pw = power_with_jobs[power_with_jobs["job_id"].notna()]
    power_stats = pw.groupby("job_id", observed=True).agg(
        mean_power_raw_kw=("power_kw", "mean"),
        median_power_raw_kw=("power_kw", "median"),
        max_power_raw_kw=("power_kw", "max"),
        power_readings=("power_kw", "size"),
    )

    vb = vibration_with_jobs[vibration_with_jobs["job_id"].notna()]
    vib_stats = vb.groupby("job_id", observed=True).agg(
        peak_vibration_g=("vibration_peak_g", "max"),
        mean_vibration_rms_g=("vibration_rms_g", "mean"),
        p95_vibration_rms_g=("vibration_rms_g", lambda s: float(np.percentile(s, 95))),
        max_vibration_rms_g=("vibration_rms_g", "max"),
        vibration_readings=("vibration_rms_g", "size"),
    )

    jobs = production_log.copy()
    jobs = jobs.merge(power_stats, on="job_id", how="left").merge(vib_stats, on="job_id", how="left")
    jobs["idle_baseline_kw"] = jobs["machine_id"].map(idle_baseline).astype(float)
    jobs["mean_power_kw"] = jobs["mean_power_raw_kw"] - jobs["idle_baseline_kw"]
    jobs["power_readings"] = jobs["power_readings"].fillna(0).astype(int)
    jobs["vibration_readings"] = jobs["vibration_readings"].fillna(0).astype(int)

    expected = jobs["duration_s"].clip(lower=1e-9)
    jobs["power_coverage"] = (jobs["power_readings"] / expected).round(3)
    jobs["vibration_coverage"] = (jobs["vibration_readings"] / (2.0 * expected)).round(3)
    return jobs


def summarise_data_quality(jobs: pd.DataFrame, min_readings: int) -> Dict[str, Any]:
    return {
        "jobs": int(len(jobs)),
        "jobs_without_power": int((jobs["power_readings"] == 0).sum()),
        "jobs_without_vibration": int((jobs["vibration_readings"] == 0).sum()),
        "jobs_below_min_readings": int((jobs["vibration_readings"] < min_readings).sum()),
        "job_ids_without_sensor_data": jobs.loc[
            (jobs["power_readings"] == 0) | (jobs["vibration_readings"] == 0), "job_id"
        ].tolist(),
    }
