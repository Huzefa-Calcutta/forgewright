"""Forgewright CNC tool-wear detection pipeline.

Modules
-------
config      typed YAML configuration
data_io     CSV loading, artefact persistence, logging
preprocess  cleaning, sensor/MES clock alignment, reading-to-job join
features    design matrix construction and per-job aggregation
models      NumPy implementations of the regressors and outlier detectors
detect      the two-stage detector and job-level verdicts
pipeline    orchestration steps called by analyse.py
"""

__version__ = "1.0.0"

__all__ = ["config", "data_io", "preprocess", "features", "models", "detect", "pipeline"]
