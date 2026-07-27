"""Cleaning, sensor/MES clock alignment, and the reading-to-job join.

Everything in here is learned from the data that is passed in -- machine ids,
part types, clock offsets and idle baselines are all discovered at run time, so
the same code runs unchanged on a different shift or a different machine fleet.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Generic cleaning helpers
# --------------------------------------------------------------------------- #
def robust_upper_bound(values: np.ndarray, iqr_multiple: float) -> float:
    """Very loose upper plausibility bound used to catch sentinel codes.

    Set wide on purpose: a genuinely heavy titanium cut must survive, while a
    hard-coded fault value such as 9999 must not.
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.inf
    q1, q3 = np.percentile(v, [25, 75])
    iqr = max(q3 - q1, 1e-9)
    return float(q3 + iqr_multiple * iqr)


def clean_power(power: pd.DataFrame, cfg) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report: Dict[str, Any] = {"rows_in": int(len(power))}
    df = power.copy()

    report["dropped_unparsed_timestamp"] = int(df["timestamp"].isna().sum())
    report["dropped_null_value"] = int(df["power_kw"].isna().sum())
    df = df.dropna(subset=["timestamp", "machine_id", "power_kw"])

    if cfg.cleaning.drop_duplicate_readings:
        before = len(df)
        df = df.drop_duplicates(subset=["machine_id", "timestamp"], keep="first")
        report["dropped_duplicate_readings"] = int(before - len(df))

    neg = df["power_kw"] < cfg.cleaning.power_min_kw
    report["dropped_negative_power"] = int(neg.sum())
    df = df[~neg]

    bound = min(
        cfg.cleaning.power_abs_max_kw,
        robust_upper_bound(df["power_kw"].to_numpy(dtype=float), cfg.cleaning.robust_outlier_iqr_multiple),
    )
    sentinel = df["power_kw"] > bound
    report["sentinel_upper_bound_kw"] = float(bound)
    report["dropped_sentinel_power"] = int(sentinel.sum())
    report["sentinel_values"] = sorted(df.loc[sentinel, "power_kw"].unique().tolist())
    df = df[~sentinel]

    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    report["rows_out"] = int(len(df))
    report["machines"] = sorted(df["machine_id"].unique().tolist())
    return df, report


def clean_vibration(vibration: pd.DataFrame, cfg) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report: Dict[str, Any] = {"rows_in": int(len(vibration))}
    df = vibration.copy()

    report["dropped_unparsed_timestamp"] = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp", "machine_id", "vibration_rms_g", "vibration_peak_g"])

    if cfg.cleaning.drop_duplicate_readings:
        before = len(df)
        df = df.drop_duplicates(subset=["machine_id", "timestamp"], keep="first")
        report["dropped_duplicate_readings"] = int(before - len(df))

    neg = (df["vibration_rms_g"] < 0) | (df["vibration_peak_g"] < 0)
    report["dropped_negative_vibration"] = int(neg.sum())
    df = df[~neg]

    # Physically, a peak within an interval cannot be below the interval RMS.
    inconsistent = df["vibration_peak_g"] < df["vibration_rms_g"]
    report["dropped_peak_below_rms"] = int(inconsistent.sum())
    df = df[~inconsistent]

    bounds = {}
    for col in ("vibration_rms_g", "vibration_peak_g"):
        bound = robust_upper_bound(df[col].to_numpy(dtype=float), cfg.cleaning.robust_outlier_iqr_multiple)
        bounds[col] = float(bound)
        bad = df[col] > bound
        report[f"dropped_sentinel_{col}"] = int(bad.sum())
        df = df[~bad]
    report["sentinel_upper_bounds"] = bounds

    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    report["rows_out"] = int(len(df))
    report["machines"] = sorted(df["machine_id"].unique().tolist())
    return df, report


def clean_production_log(log: pd.DataFrame, cfg) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report: Dict[str, Any] = {"rows_in": int(len(log))}
    df = log.copy()
    df["data_quality_notes"] = ""

    def note(mask: pd.Series, text: str) -> None:
        if mask.any():
            df.loc[mask, "data_quality_notes"] = (
                df.loc[mask, "data_quality_notes"].str.cat([text] * int(mask.sum()), sep="; ").str.strip("; ")
            )

    bad_ts = df["start_time"].isna() | df["end_time"].isna()
    report["dropped_unparsed_job_times"] = int(bad_ts.sum())
    report["dropped_job_ids_unparsed_times"] = df.loc[bad_ts, "job_id"].tolist()
    df = df[~bad_ts].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["job_id"], keep="first")
    report["dropped_duplicate_job_ids"] = int(before - len(df))

    reversed_mask = df["end_time"] < df["start_time"]
    report["reversed_job_times"] = int(reversed_mask.sum())
    report["reversed_job_ids"] = df.loc[reversed_mask, "job_id"].tolist()
    if cfg.cleaning.swap_reversed_job_times and reversed_mask.any():
        s = df.loc[reversed_mask, "start_time"].copy()
        df.loc[reversed_mask, "start_time"] = df.loc[reversed_mask, "end_time"]
        df.loc[reversed_mask, "end_time"] = s
        note(reversed_mask, "start/end transposed in MES; swapped")
    else:
        df = df[~reversed_mask]

    df["duration_s"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    too_short = df["duration_s"] < cfg.cleaning.min_job_duration_s
    report["dropped_zero_duration_jobs"] = int(too_short.sum())
    report["dropped_job_ids_zero_duration"] = df.loc[too_short, "job_id"].tolist()
    df = df[~too_short].copy()

    missing_qty = df["quantity"].isna()
    report["missing_quantity"] = int(missing_qty.sum())
    report["missing_quantity_job_ids"] = df.loc[missing_qty, "job_id"].tolist()
    if cfg.cleaning.impute_missing_quantity and missing_qty.any():
        by_part = df.groupby("part_type")["quantity"].transform("median")
        overall = df["quantity"].median()
        df["quantity"] = df["quantity"].fillna(by_part).fillna(overall)
        df["quantity_imputed"] = missing_qty
        note(missing_qty, "quantity missing in MES; imputed with part-type median")
    else:
        df["quantity_imputed"] = False
        df = df[~missing_qty]

    df = df.sort_values(["machine_id", "start_time"]).reset_index(drop=True)

    overlaps: List[str] = []
    for _, g in df.groupby("machine_id"):
        prev_end, prev_id = None, None
        for row in g.itertuples():
            if prev_end is not None and row.start_time < prev_end:
                overlaps.append(f"{prev_id}->{row.job_id}")
            prev_end, prev_id = row.end_time, row.job_id
    report["overlapping_job_pairs"] = overlaps

    report["rows_out"] = int(len(df))
    report["machines"] = sorted(df["machine_id"].unique().tolist())
    report["part_types"] = sorted(df["part_type"].unique().tolist())
    return df, report


# --------------------------------------------------------------------------- #
# Sensor / MES clock alignment
# --------------------------------------------------------------------------- #
def _coverage_separation(power: pd.DataFrame, log: pd.DataFrame, offset_s: float) -> float:
    """Mean (in-job power - out-of-job power) across machines for one offset.

    Cutting draws power and idling does not, so the true offset is the one that
    makes MES job windows line up with the high-power stretches. This is a
    physics-based objective rather than an assumed timezone rule.
    """
    shift = pd.to_timedelta(offset_s, unit="s")
    separations: List[float] = []
    for machine, g in power.groupby("machine_id", observed=True):
        jobs = log[log["machine_id"] == machine]
        if jobs.empty:
            continue
        t = g["timestamp"].to_numpy()
        inside = np.zeros(len(g), dtype=bool)
        starts = (jobs["start_time"] + shift).to_numpy()
        ends = (jobs["end_time"] + shift).to_numpy()
        for s, e in zip(starts, ends):
            inside |= (t >= s) & (t <= e)
        if inside.sum() == 0 or (~inside).sum() == 0:
            continue
        p = g["power_kw"].to_numpy(dtype=float)
        separations.append(float(p[inside].mean() - p[~inside].mean()))
    if not separations:
        return -np.inf
    return float(np.mean(separations))


def estimate_clock_offset(power: pd.DataFrame, log: pd.DataFrame, cfg) -> Dict[str, Any]:
    """Coarse-to-fine search for the MES->sensor clock offset."""
    c = cfg.cleaning
    if not c.align_clocks:
        return {"applied_offset_s": 0.0, "searched": False}

    coarse_candidates = [
        h * c.alignment_coarse_step_s
        for h in range(-c.alignment_coarse_range_h, c.alignment_coarse_range_h + 1)
    ]
    coarse_scores = {off: _coverage_separation(power, log, off) for off in coarse_candidates}
    best_coarse = max(coarse_scores, key=lambda k: coarse_scores[k])

    fine_candidates = [
        best_coarse + step
        for step in range(-c.alignment_fine_window_s, c.alignment_fine_window_s + 1, c.alignment_fine_step_s)
    ]
    fine_scores = {off: _coverage_separation(power, log, off) for off in fine_candidates}
    best = max(fine_scores, key=lambda k: fine_scores[k])

    per_machine: Dict[str, float] = {}
    for machine in sorted(power["machine_id"].unique().tolist()):
        pm = power[power["machine_id"] == machine]
        lm = log[log["machine_id"] == machine]
        if lm.empty:
            continue
        scores = {off: _coverage_separation(pm, lm, off) for off in coarse_candidates}
        per_machine[str(machine)] = float(max(scores, key=lambda k: scores[k]))

    return {
        "applied_offset_s": float(best),
        "applied_offset_h": float(best) / 3600.0,
        "searched": True,
        "best_separation_kw": float(fine_scores[best]),
        "zero_offset_separation_kw": float(coarse_scores.get(0, np.nan)),
        "coarse_scores_kw": {str(k): round(v, 4) for k, v in coarse_scores.items()},
        "per_machine_best_offset_s": per_machine,
        "per_machine_agreement": bool(len(set(per_machine.values())) <= 1),
    }


def apply_clock_offset(log: pd.DataFrame, offset_s: float) -> pd.DataFrame:
    df = log.copy()
    shift = pd.to_timedelta(offset_s, unit="s")
    df["start_time_aligned"] = df["start_time"] + shift
    df["end_time_aligned"] = df["end_time"] + shift
    return df


# --------------------------------------------------------------------------- #
# Interval join
# --------------------------------------------------------------------------- #
def assign_jobs(readings: pd.DataFrame, log: pd.DataFrame, edge_trim_s: float = 0.0) -> pd.DataFrame:
    """Label each sensor reading with the job running on its machine, if any.

    Implemented as a per-machine ``searchsorted`` over job start times, which is
    O(n log m) and does not assume any particular machine naming or ordering.
    Readings outside every job window get ``job_id = NA`` and are treated as
    idle (they are what the idle baseline is estimated from).
    """
    out = readings.copy()
    out["job_id"] = pd.Series(pd.NA, index=out.index, dtype="string")

    trim = pd.to_timedelta(edge_trim_s, unit="s")
    for machine, jobs in log.groupby("machine_id", observed=True):
        mask = out["machine_id"] == machine
        if not mask.any():
            continue
        jobs = jobs.sort_values("start_time_aligned")
        starts = (jobs["start_time_aligned"] + trim).to_numpy()
        ends = (jobs["end_time_aligned"] - trim).to_numpy()
        ids = jobs["job_id"].to_numpy()

        idx = np.flatnonzero(mask.to_numpy())
        t = out.loc[mask, "timestamp"].to_numpy()
        pos = np.searchsorted(starts, t, side="right") - 1
        valid = (pos >= 0) & (t <= ends[np.clip(pos, 0, len(ends) - 1)])
        labels = np.full(len(t), None, dtype=object)
        labels[valid] = ids[pos[valid]]
        out.iloc[idx, out.columns.get_loc("job_id")] = pd.array(labels, dtype="string")

    return out


def pair_vibration_with_power(vibration: pd.DataFrame, power: pd.DataFrame, tolerance_s: float) -> pd.DataFrame:
    """Attach the nearest-in-time power reading to every vibration reading.

    Vibration logs ~2 Hz and power ~1 Hz, so the sensors are matched with an
    as-of join per machine rather than resampled onto an arbitrary grid. This
    keeps the native vibration resolution, which is what the peak metric needs.
    """
    left = vibration.sort_values("timestamp")
    right = power[["timestamp", "machine_id", "power_kw"]].sort_values("timestamp")
    merged = pd.merge_asof(
        left,
        right,
        on="timestamp",
        by="machine_id",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_s),
    )
    return merged


def estimate_idle_baseline(power_with_jobs: pd.DataFrame, cfg) -> Tuple[pd.Series, Dict[str, Any]]:
    """Per-machine idle power draw, measured on readings outside any job."""
    idle = power_with_jobs[power_with_jobs["job_id"].isna()]
    method, pct = cfg.idle.method, cfg.idle.percentile

    if method == "min":
        baseline = idle.groupby("machine_id", observed=True)["power_kw"].min()
        global_value = float(idle["power_kw"].min()) if len(idle) else 0.0
    else:
        baseline = idle.groupby("machine_id", observed=True)["power_kw"].quantile(pct / 100.0)
        global_value = float(np.percentile(idle["power_kw"], pct)) if len(idle) else 0.0

    all_machines = pd.Index(sorted(power_with_jobs["machine_id"].unique().tolist()), name="machine_id")
    baseline = baseline.reindex(all_machines)
    filled = baseline.isna()
    if cfg.idle.fallback_to_global:
        baseline = baseline.fillna(global_value)
    else:
        baseline = baseline.fillna(0.0)

    report = {
        "method": method,
        "percentile": pct if method == "percentile" else None,
        "idle_reading_count": int(len(idle)),
        "idle_share_of_readings": float(len(idle) / max(len(power_with_jobs), 1)),
        "baseline_kw": {str(k): float(v) for k, v in baseline.items()},
        "machines_without_idle_data": [str(m) for m in all_machines[filled.to_numpy()]],
    }
    return baseline, report
