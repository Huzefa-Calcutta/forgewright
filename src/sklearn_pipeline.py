"""scikit-learn implementation of the two-stage detector, built with ``sklearn.pipeline``.

This is the compact counterpart to the dependency-free NumPy detector in
``models.py`` + ``detect.py``. Everything that is *not* estimator-specific --
loading, cleaning, clock alignment, the interval join, per-job aggregation, the
robust scale estimator and the job-level decision rule -- is reused from the
existing modules rather than duplicated. Only the modelling layer is rewritten.

Why this file is short: ``Pipeline`` + ``ColumnTransformer`` absorb the
preprocessing that the NumPy backend had to do by hand.

* One-hot encoding, scaling and the design matrix become a declarative
  ``ColumnTransformer``, so the primary detector is a single object that takes the
  raw reading DataFrame and returns predictions -- no manual matrix assembly, no
  encoder to persist separately, and no risk of train/score column drift.
* All three secondary detectors (``IsolationForest``, ``EllipticEnvelope``,
  ``LocalOutlierFactor``) share scikit-learn's outlier-detector API, so one code
  path fits and scores any of them: ``score_samples`` (higher = more normal),
  ``decision_function`` (negative = outlier) and a native ``contamination``
  threshold exposed as ``offset_``.

Sign convention: scikit-learn's ``score_samples`` is higher-is-more-normal, so
the stored ``secondary_score`` negates it to keep "higher = more anomalous"
consistent with the rest of the codebase and the reading-level detail file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor, Ridge
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import detect, features as feat


class NumericByCategory(BaseEstimator, TransformerMixin):
    """Products of a numeric column with the one-hot levels of a categorical column.

    The only custom transformer here, because it is the one thing
    ``ColumnTransformer`` cannot express: it multiplies *across* two of its own
    branches. It gives each machine and each material its own vibration-vs-power
    slope, so an aggressive titanium cut is judged against titanium.

    Deliberately NOT done with ``PolynomialFeatures(interaction_only=True)``, which
    would also generate machine x part_type products. That interaction is harmful
    here: several worn jobs share a machine/material cell (3 of 5 CNC-07
    ST-housing jobs are worn), so such a term would absorb the very signal we are
    trying to detect into the baseline.

    Unseen levels at scoring time produce zeros, i.e. they fall back to the global
    slope instead of raising -- the behaviour we want on a held-out shift.
    """

    def __init__(self, pairs: Sequence[Sequence[str]] = ()) -> None:
        self.pairs = pairs

    def fit(self, X: pd.DataFrame, y=None) -> "NumericByCategory":
        self.pairs_ = [tuple(p) for p in (self.pairs or []) if len(p) == 2]
        self.levels_: Dict[str, List[Any]] = {}
        for _, cat_col in self.pairs_:
            if cat_col not in self.levels_:
                # drop the first level for identifiability: it is the reference slope
                self.levels_[cat_col] = sorted(X[cat_col].dropna().unique().tolist())[1:]
        self.feature_names_out_ = [f"{num}:{cat}={lvl}"
                                   for num, cat in self.pairs_
                                   for lvl in self.levels_[cat]]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for num_col, cat_col in self.pairs_:
            values = pd.to_numeric(X[num_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            for level in self.levels_[cat_col]:
                cols.append(values * (X[cat_col] == level).to_numpy(dtype=float))
        if not cols:
            return np.zeros((len(X), 0), dtype=float)
        return np.column_stack(cols)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)


def make_design(cfg) -> ColumnTransformer:
    """Declarative design matrix: scaled numerics | one-hot categoricals | interactions."""
    p = cfg.primary
    return ColumnTransformer(
        [
            ("num", StandardScaler(), list(p.numeric_features)),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
             list(p.categorical_features)),
            ("inter", NumericByCategory(p.interactions),
             list(dict.fromkeys(list(p.numeric_features) + list(p.categorical_features)))),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_primary(cfg) -> Pipeline:
    """Design matrix + a linear estimator, as one fit/predict object.

    ``huber`` is the recommended primary: it downweights the ~13% contaminated
    readings smoothly instead of letting them define the healthy baseline, and
    unlike RANSAC it is deterministic, which matters for a pipeline that must run
    unchanged on a held-out shift.
    """
    p, seed = cfg.primary, cfg.run.random_state
    estimators = {
        "ridge": lambda: Ridge(alpha=p.ridge_alpha),
        "huber": lambda: HuberRegressor(epsilon=p.huber_epsilon, alpha=1e-4, max_iter=300),
        "ransac": lambda: RANSACRegressor(estimator=LinearRegression(), min_samples=0.15,
                                          max_trials=p.ransac_trials, random_state=seed),
    }
    name = "ransac" if p.model == "robust" else p.model  # "robust" is the brief's alias
    if name not in estimators:
        raise ValueError(f"Unknown primary.model {p.model!r}; choose from {sorted(estimators)} or 'robust'")
    return Pipeline([("design", make_design(cfg)), ("model", estimators[name]())])


def make_secondary(cfg) -> Pipeline:
    """Scaler + a scikit-learn outlier detector, as one fit/score object.

    ``mahalanobis`` maps to ``EllipticEnvelope`` (a robust Minimum Covariance
    Determinant fit): the healthy cloud in log space is a compact ellipsoid, which
    is exactly this estimator's assumption, and it is deterministic and cheap.
    """
    s, seed = cfg.secondary, cfg.run.random_state
    detectors = {
        "isolation_forest": lambda: IsolationForest(n_estimators=s.if_n_estimators,
                                                   max_samples=s.if_max_samples,
                                                   contamination=s.contamination,
                                                   random_state=seed, n_jobs=-1),
        "mahalanobis": lambda: EllipticEnvelope(contamination=s.contamination,
                                                support_fraction=0.75, random_state=seed),
        "lof": lambda: LocalOutlierFactor(n_neighbors=s.lof_n_neighbors,
                                          contamination=s.contamination, novelty=True),
    }
    if s.model not in detectors:
        raise ValueError(f"Unknown secondary.model {s.model!r}; choose from {sorted(detectors)}")
    return Pipeline([("scale", StandardScaler()), ("detector", detectors[s.model]())])


@dataclass
class SklearnBundle:
    """A fitted detector: two pipelines plus the residual scale they are judged against."""

    primary_name: str
    secondary_name: str
    primary: Pipeline
    secondary: Pipeline
    target: str
    secondary_features: List[str]
    residual_center: float
    residual_scale: float
    sigma_multiple: float
    one_sided: bool
    feature_names: List[str] = field(default_factory=list)
    fit_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def secondary_model(self):
        """The bare detector, for callers that want ``offset_``/``threshold_``."""
        return self.secondary["detector"]


def fit_detector(cutting: pd.DataFrame, cfg) -> SklearnBundle:
    """Fit both stages on this shift's cutting readings."""
    p, s = cfg.primary, cfg.secondary

    primary = make_primary(cfg).fit(cutting, cutting[p.target].to_numpy(dtype=float))
    y = cutting[p.target].to_numpy(dtype=float)
    residuals = y - primary.predict(cutting)
    center = float(np.median(residuals))
    # Reuse the shared trimmed-MAD estimator: the scale must describe the HEALTHY
    # scatter, or the worn readings inflate the very sigma used to catch them.
    scale = detect.robust_scale(residuals - center, p.scale_estimator, p.scale_trim_sigma)

    secondary = make_secondary(cfg).fit(cutting[s.features])

    try:
        names = list(primary["design"].get_feature_names_out())
    except Exception:  # feature names are diagnostics only; never fail the run for them
        names = []

    ss_res = float(np.sum((residuals - residuals.mean()) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    metadata = {
        "backend": "sklearn",
        "n_fit_readings": int(len(cutting)),
        "n_features": len(names),
        "r2_in_sample": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "residual_center": center,
        "residual_scale": scale,
        "residual_scale_std": float(residuals.std()),
        "machines": sorted(cutting["machine_id"].unique().tolist()),
        "part_types": sorted(cutting["part_type"].dropna().unique().tolist()),
    }

    return SklearnBundle(
        primary_name=p.model, secondary_name=s.model,
        primary=primary, secondary=secondary,
        target=p.target, secondary_features=list(s.features),
        residual_center=center, residual_scale=scale,
        sigma_multiple=p.sigma_multiple, one_sided=p.one_sided,
        feature_names=names, fit_metadata=metadata,
    )


def score_readings(cutting: pd.DataFrame, bundle: SklearnBundle) -> pd.DataFrame:
    """Attach per-reading primary/secondary scores and the combined AND flag."""
    df = cutting.copy()

    predicted = bundle.primary.predict(df)
    residual = df[bundle.target].to_numpy(dtype=float) - predicted
    df["predicted_rms_g"] = predicted
    df["residual_g"] = residual
    df["residual_sigma"] = (residual - bundle.residual_center) / bundle.residual_scale

    if bundle.one_sided:
        # Only EXCESS vibration indicates wear; unusually smooth cutting does not.
        df["primary_score"] = df["residual_sigma"]
    else:
        df["primary_score"] = df["residual_sigma"].abs()
    df["primary_anomalous"] = df["primary_score"] >= bundle.sigma_multiple

    features = df[bundle.secondary_features]
    # Negated so higher = more anomalous, matching the NumPy backend's convention.
    df["secondary_score"] = -bundle.secondary.score_samples(features)
    # predict() == -1 applies scikit-learn's own contamination threshold.
    df["secondary_anomalous"] = bundle.secondary.predict(features) == -1

    df["anomalous"] = df["primary_anomalous"] & df["secondary_anomalous"]
    return df


def run_detection(tables: Dict[str, pd.DataFrame], bundle: SklearnBundle, cfg, logger):
    """Score readings, roll up to jobs, and assemble the deliverable table.

    The aggregation and job-level decision rule are the shared ones from
    ``detect``/``features``, so the sklearn and NumPy backends are compared on the
    modelling layer alone.
    """
    scored = score_readings(tables["cutting"], bundle)
    per_job_flags = detect.aggregate_reading_flags(scored, cfg)

    jobs = feat.aggregate_jobs(
        tables["production_log"], tables["power"], tables["vibration"], tables["idle_baseline"]
    )
    jobs = jobs.merge(per_job_flags, on="job_id", how="left")
    jobs = detect.flag_jobs(jobs, cfg)
    output = detect.build_output_table(jobs)

    flagged = int((output["flagged"] == "true").sum())
    logger.info("Flagged %s of %s jobs as showing tool wear", flagged, len(output))

    return scored, jobs, output


def resolve_detector(cutting: pd.DataFrame, cfg, logger, bundle_path) -> SklearnBundle:
    """Fit a fresh detector on this shift, or reload a persisted one."""
    import data_io

    if cfg.run.mode == "load":
        if not bundle_path.exists():
            raise FileNotFoundError(f"run.mode='load' but no model at {bundle_path}")
        bundle = data_io.load_artifact(bundle_path)
        logger.info("Loaded sklearn detector from %s (%s + %s)",
                    bundle_path, bundle.primary_name, bundle.secondary_name)
        return bundle

    logger.info("Fitting %s (primary) + %s (secondary) on %s cutting readings [sklearn]",
                cfg.primary.model, cfg.secondary.model, f"{len(cutting):,}")
    bundle = fit_detector(cutting, cfg)
    logger.info("Primary in-sample R2=%.3f over %s features, robust residual sigma=%.4f",
                bundle.fit_metadata["r2_in_sample"], bundle.fit_metadata["n_features"],
                bundle.residual_scale)
    if cfg.run.save_model:
        data_io.save_artifact(bundle, bundle_path)
        logger.info("Saved detector to %s", bundle_path)
    return bundle
