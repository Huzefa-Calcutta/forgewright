"""Input/output helpers: raw CSV loading, artefact persistence, logging."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd

LOGGER_NAME = "toolwear"

POWER_SCHEMA = {"timestamp", "machine_id", "power_kw"}
VIBRATION_SCHEMA = {"timestamp", "machine_id", "vibration_rms_g", "vibration_peak_g"}
PRODUCTION_SCHEMA = {
    "job_id", "machine_id", "part_type", "operator", "quantity", "start_time", "end_time",
}


def get_logger(verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False
    return logger


def _require_columns(df: pd.DataFrame, expected: set, source: Path) -> None:
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{source.name} is missing required columns: {sorted(missing)}")


def _parse_utc(series: pd.Series) -> pd.Series:
    """Parse timestamps to tz-naive UTC.

    Sensor exports carry an explicit UTC offset ('...Z'); the MES export is
    naive. Everything is normalised to naive-UTC here and any residual clock
    offset is estimated later from the data itself.
    """
    parsed = pd.to_datetime(series, format="mixed", utc=True, errors="coerce")
    return parsed.dt.tz_localize(None)


def load_power(path: str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    _require_columns(df, POWER_SCHEMA, path)
    df = df[["timestamp", "machine_id", "power_kw"]].copy()
    df["timestamp"] = _parse_utc(df["timestamp"])
    df["machine_id"] = df["machine_id"].astype("string").str.strip()
    df["power_kw"] = pd.to_numeric(df["power_kw"], errors="coerce")
    return df


def load_vibration(path: str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    _require_columns(df, VIBRATION_SCHEMA, path)
    df = df[["timestamp", "machine_id", "vibration_rms_g", "vibration_peak_g"]].copy()
    df["timestamp"] = _parse_utc(df["timestamp"])
    df["machine_id"] = df["machine_id"].astype("string").str.strip()
    for col in ("vibration_rms_g", "vibration_peak_g"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_production_log(path: str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    _require_columns(df, PRODUCTION_SCHEMA, path)
    df = df[sorted(PRODUCTION_SCHEMA, key=list(PRODUCTION_SCHEMA).index)].copy() if False else df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], format="mixed", errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], format="mixed", errors="coerce")
    for col in ("job_id", "machine_id", "part_type", "operator"):
        df[col] = df[col].astype("string").str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    return df


def load_raw_data(cfg) -> Dict[str, pd.DataFrame]:
    return {
        "power": load_power(cfg.power_path),
        "vibration": load_vibration(cfg.vibration_path),
        "production_log": load_production_log(cfg.production_log_path),
    }


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_table(df: pd.DataFrame, path: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path


def write_json(payload: Dict[str, Any], path: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def save_artifact(obj: Any, path: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_artifact(path: Path) -> Any:
    path = Path(path)
    with open(path, "rb") as fh:
        return pickle.load(fh)
