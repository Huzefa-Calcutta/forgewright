"""Self-contained NumPy implementations of the regressors and outlier detectors.

The sandbox has no scikit-learn (no network access), so the estimators used by
the tool-wear pipeline are implemented here with a scikit-learn-like API
(``fit`` / ``predict`` / ``score_samples``). Every estimator is deterministic
given ``random_state``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Encoding helpers
# --------------------------------------------------------------------------- #
@dataclass
class OneHotEncoder:
    """Minimal one-hot encoder that is safe on unseen categories.

    Categories are learned at ``fit`` time from whatever is present in the
    data, so nothing is hardcoded to a particular set of machines or parts.
    Unseen categories encode as an all-zero row (they fall back to the model
    intercept) instead of raising.
    """

    drop_first: bool = True
    categories_: Dict[str, List[str]] = field(default_factory=dict)
    feature_names_: List[str] = field(default_factory=list)

    def fit(self, frame) -> "OneHotEncoder":
        self.categories_ = {}
        self.feature_names_ = []
        for col in frame.columns:
            cats = sorted(frame[col].astype("string").dropna().unique().tolist())
            if self.drop_first and len(cats) > 1:
                cats = cats[1:]  # first level absorbed into the intercept
            self.categories_[col] = cats
            self.feature_names_ += [f"{col}={c}" for c in cats]
        return self

    def transform(self, frame) -> np.ndarray:
        blocks = []
        for col, cats in self.categories_.items():
            values = frame[col].astype("string").to_numpy()
            block = np.zeros((len(frame), len(cats)), dtype=float)
            for j, cat in enumerate(cats):
                block[:, j] = (values == cat).astype(float)
            blocks.append(block)
        if not blocks:
            return np.zeros((len(frame), 0), dtype=float)
        return np.hstack(blocks)

    def fit_transform(self, frame) -> np.ndarray:
        return self.fit(frame).transform(frame)


@dataclass
class StandardScaler:
    """Zero-mean unit-variance scaler (constant columns are left alone)."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
# Primary detectors: regressors of vibration on power + job context
# --------------------------------------------------------------------------- #
class _LinearModelBase:
    """Shared plumbing: standardize inputs, fit an intercept separately."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(X)
        return Z @ self.coef_ + self.intercept_


class RidgeRegressor(_LinearModelBase):
    """L2-penalised least squares, solved in closed form on centred data."""

    name = "ridge"

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        Z = self.scaler.fit_transform(X)
        y = np.asarray(y, dtype=float)
        y_mean = y.mean()
        yc = y - y_mean
        n_features = Z.shape[1]
        gram = Z.T @ Z + self.alpha * np.eye(n_features)
        self.coef_ = np.linalg.solve(gram, Z.T @ yc)
        self.intercept_ = float(y_mean)
        self.n_iter_ = 1
        return self


class HuberRegressor(_LinearModelBase):
    """Huber M-estimator fitted by iteratively reweighted least squares.

    Squared loss inside ``epsilon`` robust scales, linear outside, so a handful
    of chattering readings cannot drag the healthy-baseline fit toward
    themselves the way ordinary least squares would.
    """

    name = "huber"

    def __init__(self, epsilon: float = 1.35, alpha: float = 1e-4, max_iter: int = 100, tol: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HuberRegressor":
        Z = self.scaler.fit_transform(X)
        y = np.asarray(y, dtype=float)
        Zi = np.hstack([np.ones((Z.shape[0], 1)), Z])
        penalty = self.alpha * np.eye(Zi.shape[1])
        penalty[0, 0] = 0.0  # never penalise the intercept
        beta = np.linalg.solve(Zi.T @ Zi + penalty, Zi.T @ y)
        for it in range(self.max_iter):
            resid = y - Zi @ beta
            sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
            sigma = max(sigma, 1e-9)
            scaled = np.abs(resid) / sigma
            w = np.ones_like(scaled)
            big = scaled > self.epsilon
            w[big] = self.epsilon / scaled[big]
            ZW = Zi * w[:, None]
            new_beta = np.linalg.solve(Zi.T @ ZW + penalty, ZW.T @ y)
            shift = np.max(np.abs(new_beta - beta))
            beta = new_beta
            self.n_iter_ = it + 1
            if shift < self.tol:
                break
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self


class RansacRegressor(_LinearModelBase):
    """RANSAC consensus fit -- the 'robust' candidate.

    Repeatedly fits on random minimal subsets, keeps the fit with the largest
    inlier consensus set, then refits on those inliers only. This gives a
    baseline that is fitted almost purely on healthy readings.
    """

    name = "ransac"

    def __init__(self, n_trials: int = 120, residual_threshold: float | None = None,
                 min_samples_frac: float = 0.15, random_state: int = 0) -> None:
        super().__init__()
        self.n_trials = int(n_trials)
        self.residual_threshold = residual_threshold
        self.min_samples_frac = float(min_samples_frac)
        self.random_state = int(random_state)
        self.inlier_mask_: np.ndarray | None = None

    @staticmethod
    def _solve(Zi: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
        penalty = ridge * np.eye(Zi.shape[1])
        penalty[0, 0] = 0.0
        return np.linalg.solve(Zi.T @ Zi + penalty, Zi.T @ y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RansacRegressor":
        Z = self.scaler.fit_transform(X)
        y = np.asarray(y, dtype=float)
        Zi = np.hstack([np.ones((Z.shape[0], 1)), Z])
        n, p = Zi.shape
        rng = np.random.default_rng(self.random_state)

        beta_full = self._solve(Zi, y)
        resid_full = y - Zi @ beta_full
        mad = 1.4826 * np.median(np.abs(resid_full - np.median(resid_full)))
        threshold = self.residual_threshold if self.residual_threshold else max(2.0 * mad, 1e-9)

        m = max(int(self.min_samples_frac * n), p + 1, 50)
        m = min(m, n)
        best_beta, best_count = beta_full, int((np.abs(resid_full) <= threshold).sum())
        for _ in range(self.n_trials):
            idx = rng.choice(n, size=m, replace=False)
            try:
                beta = self._solve(Zi[idx], y[idx])
            except np.linalg.LinAlgError:
                continue
            count = int((np.abs(y - Zi @ beta) <= threshold).sum())
            if count > best_count:
                best_beta, best_count = beta, count

        inliers = np.abs(y - Zi @ best_beta) <= threshold
        if inliers.sum() > p + 1:
            best_beta = self._solve(Zi[inliers], y[inliers])
            inliers = np.abs(y - Zi @ best_beta) <= threshold
        self.inlier_mask_ = inliers
        self.intercept_ = float(best_beta[0])
        self.coef_ = best_beta[1:]
        self.n_iter_ = self.n_trials
        return self


PRIMARY_MODELS = {
    "ridge": RidgeRegressor,
    "huber": HuberRegressor,
    "ransac": RansacRegressor,
    "robust": RansacRegressor,  # alias used in the brief
}


# --------------------------------------------------------------------------- #
# Secondary detectors: unsupervised multivariate outlier scores
# --------------------------------------------------------------------------- #
class IsolationForestDetector:
    """Isolation Forest. Higher ``score_samples`` output == more anomalous."""

    name = "isolation_forest"

    def __init__(self, n_estimators: int = 100, max_samples: int = 256,
                 contamination: float = 0.02, random_state: int = 0) -> None:
        self.n_estimators = int(n_estimators)
        self.max_samples = int(max_samples)
        self.contamination = float(contamination)
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.trees_: List[Dict[str, np.ndarray]] = []
        self.threshold_: float = np.inf
        self._psi: int = 0

    @staticmethod
    def _c(n: int) -> float:
        if n <= 1:
            return 1.0
        h = np.log(n - 1) + 0.5772156649015329
        return 2.0 * h - 2.0 * (n - 1) / n

    def _build_tree(self, X: np.ndarray, rng: np.random.Generator, max_depth: int) -> Dict[str, np.ndarray]:
        n = X.shape[0]
        cap = 2 * n + 1
        feature = np.full(cap, -1, dtype=np.int32)
        threshold = np.zeros(cap, dtype=float)
        left = np.full(cap, -1, dtype=np.int32)
        right = np.full(cap, -1, dtype=np.int32)
        size = np.zeros(cap, dtype=np.int32)
        depth = np.zeros(cap, dtype=np.int32)

        node_count = 1
        stack: List[Tuple[int, np.ndarray, int]] = [(0, np.arange(n), 0)]
        while stack:
            node, idx, d = stack.pop()
            size[node] = len(idx)
            depth[node] = d
            if len(idx) <= 1 or d >= max_depth:
                continue
            sub = X[idx]
            spread = sub.max(axis=0) - sub.min(axis=0)
            usable = np.flatnonzero(spread > 1e-12)
            if usable.size == 0:
                continue
            f = int(rng.choice(usable))
            lo, hi = sub[:, f].min(), sub[:, f].max()
            t = float(rng.uniform(lo, hi))
            mask = sub[:, f] < t
            if mask.all() or (~mask).all():
                continue
            feature[node], threshold[node] = f, t
            l, r = node_count, node_count + 1
            node_count += 2
            left[node], right[node] = l, r
            stack.append((l, idx[mask], d + 1))
            stack.append((r, idx[~mask], d + 1))

        s = slice(0, node_count)
        return {"feature": feature[s], "threshold": threshold[s], "left": left[s],
                "right": right[s], "size": size[s], "depth": depth[s]}

    def _path_length(self, tree: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
        node = np.zeros(X.shape[0], dtype=np.int32)
        active = np.ones(X.shape[0], dtype=bool)
        feature, threshold, left, right = tree["feature"], tree["threshold"], tree["left"], tree["right"]
        while active.any():
            cur = node[active]
            f = feature[cur]
            internal = f >= 0
            if not internal.any():
                break
            rows = np.flatnonzero(active)
            move = rows[internal]
            fm, cm = f[internal], cur[internal]
            go_left = X[move, fm] < threshold[cm]
            node[move] = np.where(go_left, left[cm], right[cm])
            stop = rows[~internal]
            active[stop] = False
        leaf_depth = tree["depth"][node].astype(float)
        leaf_size = tree["size"][node].astype(float)
        adj = np.array([self._c(int(s)) for s in leaf_size])
        return leaf_depth + adj

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        Z = self.scaler.fit_transform(X)
        rng = np.random.default_rng(self.random_state)
        psi = min(self.max_samples, Z.shape[0])
        self._psi = psi
        max_depth = int(np.ceil(np.log2(max(psi, 2))))
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(Z.shape[0], size=psi, replace=False)
            self.trees_.append(self._build_tree(Z[idx], rng, max_depth))
        scores = self.score_samples(X)
        self.threshold_ = float(np.quantile(scores, 1.0 - self.contamination))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(X)
        total = np.zeros(Z.shape[0], dtype=float)
        for tree in self.trees_:
            total += self._path_length(tree, Z)
        mean_path = total / len(self.trees_)
        return 2.0 ** (-mean_path / self._c(self._psi))

    def predict_outlier(self, X: np.ndarray) -> np.ndarray:
        return self.score_samples(X) >= self.threshold_


class MahalanobisDetector:
    """Robust Mahalanobis distance using an iteratively trimmed covariance."""

    name = "mahalanobis"

    def __init__(self, contamination: float = 0.02, support_fraction: float = 0.75,
                 n_iter: int = 15, random_state: int = 0) -> None:
        self.contamination = float(contamination)
        self.support_fraction = float(support_fraction)
        self.n_iter = int(n_iter)
        self.random_state = int(random_state)
        self.location_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None
        self.threshold_: float = np.inf

    def fit(self, X: np.ndarray) -> "MahalanobisDetector":
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        h = max(int(self.support_fraction * n), p + 1)
        loc = np.median(X, axis=0)
        cov = np.cov(X, rowvar=False) + 1e-9 * np.eye(p)
        for _ in range(self.n_iter):
            prec = np.linalg.pinv(cov)
            d = np.einsum("ij,jk,ik->i", X - loc, prec, X - loc)
            keep = np.argsort(d)[:h]  # the h most central points
            new_loc = X[keep].mean(axis=0)
            new_cov = np.cov(X[keep], rowvar=False) + 1e-9 * np.eye(p)
            if np.allclose(new_loc, loc, atol=1e-10) and np.allclose(new_cov, cov, atol=1e-10):
                loc, cov = new_loc, new_cov
                break
            loc, cov = new_loc, new_cov
        self.location_, self.precision_ = loc, np.linalg.pinv(cov)
        scores = self.score_samples(X)
        self.threshold_ = float(np.quantile(scores, 1.0 - self.contamination))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        diff = X - self.location_
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, self.precision_, diff), 0.0))

    def predict_outlier(self, X: np.ndarray) -> np.ndarray:
        return self.score_samples(X) >= self.threshold_


class LocalOutlierFactorDetector:
    """LOF against a fitted reference subsample (novelty-style scoring).

    Exact LOF is O(n^2); with ~170k readings we fit LOF on a stratified random
    reference sample and score every reading against it in chunks. Distances
    are computed brute force in NumPy because SciPy/sklearn are unavailable.
    """

    name = "lof"

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.02,
                 reference_size: int = 4000, chunk_size: int = 8000, random_state: int = 0) -> None:
        self.n_neighbors = int(n_neighbors)
        self.contamination = float(contamination)
        self.reference_size = int(reference_size)
        self.chunk_size = int(chunk_size)
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.reference_: np.ndarray | None = None
        self.k_distance_: np.ndarray | None = None
        self.lrd_ref_: np.ndarray | None = None
        self.lof_reference_: np.ndarray | None = None
        self.threshold_: float = np.inf
        self._k: int = int(n_neighbors)

    @staticmethod
    def _dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
        return np.sqrt(np.maximum(d2, 0.0))

    def fit(self, X: np.ndarray) -> "LocalOutlierFactorDetector":
        Z = self.scaler.fit_transform(X)
        rng = np.random.default_rng(self.random_state)
        size = min(self.reference_size, Z.shape[0])
        idx = rng.choice(Z.shape[0], size=size, replace=False)
        R = Z[idx]
        self.reference_ = R
        k = min(self.n_neighbors, size - 1)

        D = self._dist(R, R)
        np.fill_diagonal(D, np.inf)
        nn_idx = np.argsort(D, axis=1)[:, :k]
        nn_dist = np.take_along_axis(D, nn_idx, axis=1)
        self.k_distance_ = nn_dist[:, -1]

        reach = np.maximum(nn_dist, self.k_distance_[nn_idx])
        self.lrd_ref_ = 1.0 / (reach.mean(axis=1) + 1e-12)

        lof_ref = (self.lrd_ref_[nn_idx].mean(axis=1)) / (self.lrd_ref_ + 1e-12)
        self._k = k
        scores = self.score_samples(X)
        self.threshold_ = float(np.quantile(scores, 1.0 - self.contamination))
        self.lof_reference_ = lof_ref
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(X)
        k = self._k
        out = np.empty(Z.shape[0], dtype=float)
        for start in range(0, Z.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, Z.shape[0])
            D = self._dist(Z[start:stop], self.reference_)
            nn_idx = np.argpartition(D, k - 1, axis=1)[:, :k]
            nn_dist = np.take_along_axis(D, nn_idx, axis=1)
            reach = np.maximum(nn_dist, self.k_distance_[nn_idx])
            lrd = 1.0 / (reach.mean(axis=1) + 1e-12)
            out[start:stop] = self.lrd_ref_[nn_idx].mean(axis=1) / (lrd + 1e-12)
        return out

    def predict_outlier(self, X: np.ndarray) -> np.ndarray:
        return self.score_samples(X) >= self.threshold_


SECONDARY_MODELS = {
    "isolation_forest": IsolationForestDetector,
    "mahalanobis": MahalanobisDetector,
    "lof": LocalOutlierFactorDetector,
}
