import pandas as pd
from datetime import datetime, timezone
from typing import Any, Dict, Tuple
import data_io
import preprocess_utils
import features as feat


def prepare_data(cfg, logger) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Load, clean, clock-align, join, and feature-build. Returns tables + diagnostics."""
    diagnostics: Dict[str, Any] = {"run_started_utc": datetime.now(timezone.utc).isoformat()}

    raw = data_io.load_raw_data(cfg)
    logger.info(
        "Loaded %s power, %s vibration, %s production rows",
        f"{len(raw['power']):,}", f"{len(raw['vibration']):,}", f"{len(raw['production_log']):,}",
    )

    power, power_report = preprocess_utils.clean_power(raw["power"], cfg)
    vibration, vib_report = preprocess_utils.clean_vibration(raw["vibration"], cfg)
    log, log_report = preprocess_utils.clean_production_log(raw["production_log"], cfg)
    diagnostics["cleaning"] = {"power": power_report, "vibration": vib_report, "production_log": log_report}
    logger.info(
        "Cleaning dropped %s power and %s vibration readings; %s of %s jobs retained",
        power_report["rows_in"] - power_report["rows_out"],
        vib_report["rows_in"] - vib_report["rows_out"],
        log_report["rows_out"], log_report["rows_in"],
    )

    alignment = preprocess_utils.estimate_clock_offset(power, log, cfg)
    diagnostics["clock_alignment"] = alignment
    logger.info(
        "Estimated MES->sensor clock offset: %+.2f h (in-job power lift %.2f kW vs %.2f kW unaligned)",
        alignment.get("applied_offset_h", 0.0),
        alignment.get("best_separation_kw", float("nan")),
        alignment.get("zero_offset_separation_kw", float("nan")),
    )
    log = preprocess_utils.apply_clock_offset(log, alignment["applied_offset_s"])

    trim = cfg.cleaning.job_edge_trim_s
    power = preprocess_utils.assign_jobs(power, log, trim)
    vibration = preprocess_utils.assign_jobs(vibration, log, trim)

    idle_baseline, idle_report = preprocess_utils.estimate_idle_baseline(power, cfg)
    diagnostics["idle_baseline"] = idle_report
    logger.info("Idle baselines (kW): %s", idle_report["baseline_kw"])

    paired = preprocess_utils.pair_vibration_with_power(vibration, power, cfg.cleaning.pair_tolerance_s)
    unpaired = int(paired["power_kw"].isna().sum())
    diagnostics["pairing"] = {
        "vibration_readings": int(len(paired)),
        "unpaired_within_tolerance": unpaired,
        "tolerance_s": cfg.cleaning.pair_tolerance_s,
    }

    readings = feat.build_reading_table(paired, log, idle_baseline)
    cutting = feat.select_cutting_readings(readings, cfg.primary.cutting_power_floor_kw)
    diagnostics["readings"] = {
        "in_job_vibration_readings": int(len(readings)),
        "cutting_readings": int(len(cutting)),
        "cutting_power_floor_kw": cfg.primary.cutting_power_floor_kw,
        "near_idle_readings_excluded": int(len(readings) - len(cutting)),
    }
    logger.info(
        "Joined %s in-job vibration readings; %s above the %.2f kW cutting floor",
        f"{len(readings):,}", f"{len(cutting):,}", cfg.primary.cutting_power_floor_kw,
    )

    tables = {
        "power": power,
        "vibration": vibration,
        "production_log": log,
        "readings": readings,
        "cutting": cutting,
        "idle_baseline": idle_baseline,
    }
    return tables, diagnostics

