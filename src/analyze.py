#!/usr/bin/env python3
"""Forgewright CNC tool-wear analysis -- scikit-learn entry point.

combines the power, vibration and MES exports,
cleans them, computes per-job mean cutting power and peak vibration, runs the
two-stage detector and writes ``output/job_summary.csv``.

this script is orchestration only and defines no functions
of its own. Every user input comes from ``config.yaml`` or a CLI override, and
nothing is hardcoded to the machines, jobs or dates of a particular shift.

"""

from __future__ import annotations

import os,sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arg_utils
import data_io
from config import load_config
import preprocess
import sklearn_pipeline as skl


# --- 1. Configuration ------------------------------------------------------- #
args = arg_utils.parse_cli_args()
cfg = load_config(args.config, overrides=arg_utils.cli_overrides(args))
logger = data_io.get_logger(cfg.run.verbose)
logger.info("Config: %s | backend: scikit-learn (sklearn.pipeline)", args.config)
logger.info("Input dir: %s | Output dir: %s", cfg.resolve(cfg.paths.input_dir), cfg.output_dir)

# --- 2. Load, clean, align clocks, join sensors to jobs (shared) ------------- #
tables, diagnostics = preprocess.prepare_data(cfg, logger)

# --- 3. Fit (or load) the two-stage sklearn detector ------------------------ #
bundle_path = data_io.ensure_dir(cfg.model_dir) / "detector_bundle_sklearn.pkl"
bundle = skl.resolve_detector(tables["cutting"], cfg, logger, bundle_path)

# --- 4. Score readings and roll up to per-job verdicts ---------------------- #
scored, jobs, output = skl.run_detection(tables, bundle, cfg, logger)
data_io.write_table(output, os.path.join(cfg.output_dir, "job_summary.csv"))

# --- 6. Console report ------------------------------------------------------ #
logger.info("=" * 78)
logger.info("Ranked tool-wear candidates (top 10 by primary score):")
for row in output.head(10).itertuples():
    logger.info(
        "  #%-3s %-8s %-7s %-11s mean_power=%6.2f kW  peak_vib=%5.2f g  wear_score=%6.2f  flagged=%s",
        row.rank, row.job_id, row.machine_id, row.part_type,
        row.mean_power_kw, row.peak_vibration_g, row.wear_score, row.flagged,
    )
logger.info("=" * 78)