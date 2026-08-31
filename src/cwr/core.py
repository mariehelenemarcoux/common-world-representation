"""Frozen linear CWR v0.1 core.

Research prototype for multiview representation under local frame drift.
"""
from __future__ import annotations
import numpy as np


def _ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    return np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T @ y)


def _invsqrt_psd(a: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((a + a.T) / 2)
    vals = np.maximum(vals, 1e-10)
    return (vecs * (1 / np.sqrt(vals))) @ vecs.T


class PlainGCCA:
    def __init__(self, latent_dim: int = 6, ridge_lambda: float = 1.5):
        self.latent_dim = latent_dim
        self.ridge_lambda = ridge_lambda

    def fit(self, views, weights=None):
        v = len(views)
        n = len(views[0])
        if weights is None:
            weights = np.ones(v)
        self.weights_ = np.asarray(weights, dtype=float)
        self.means_ = []
        centered = []
        blocks = []
        for i, x in enumerate(views):
            mu = x.mean(0)
            xc = x - mu
            self.means_.append(mu)
            centered.append(xc)
            w = _invsqrt_psd(
                xc.T @ xc + self.ridge_lambda * np.eye(x.shape[1])
            )
            blocks.append(np.sqrt(self.weights_[i]) * xc @ w)
        u, _, _ = np.linalg.svd(np.concatenate(blocks, axis=1), full_matrices=False)
        self.latent_ = u[:, : self.latent_dim] * np.sqrt(n)
        self.encoders_ = []
        self.residuals_ = []
        for xc in centered:
            b = _ridge(xc, self.latent_, self.ridge_lambda)
            self.encoders_.append(b)
            self.residuals_.append(np.mean((xc @ b - self.latent_) ** 2))
        self.residuals_ = np.asarray(self.residuals_)
        return self

    def transform(self, views, available=None, means=None, encoders=None, weights=None):
        if available is None:
            available = list(range(len(views)))
        means = self.means_ if means is None else means
        encoders = self.encoders_ if encoders is None else encoders
        weights = self.weights_ if weights is None else np.asarray(weights, dtype=float)
        preds = np.stack([(views[i] - means[i]) @ encoders[i] for i in available])
        w = weights[available].astype(float)
        w /= w.sum() + 1e-12
        return np.tensordot(w, preds, axes=(0, 0))


class CWR:
    """CWR v0.1: weighted GCCA plus local view-map realignment."""

    def __init__(
        self,
        latent_dim: int = 6,
        ridge_lambda: float = 1.5,
        reliability_power: float = 1.5,
        change_threshold: float = 1.65,
    ):
        self.latent_dim = latent_dim
        self.ridge_lambda = ridge_lambda
        self.reliability_power = reliability_power
        self.change_threshold = change_threshold

    def fit(self, views):
        base = PlainGCCA(self.latent_dim, self.ridge_lambda).fit(views)
        rel = 1.0 / (base.residuals_ + 0.05)
        rel = (rel / rel.mean()) ** self.reliability_power
        model = PlainGCCA(self.latent_dim, self.ridge_lambda).fit(views, rel)
        self.means_ = model.means_
        self.encoders_ = model.encoders_
        self.weights_ = rel
        self.latent_ = model.latent_
        return self

    def transform(self, views, available=None):
        if available is None:
            available = list(range(len(views)))
        preds = np.stack(
            [(views[i] - self.means_[i]) @ self.encoders_[i] for i in available]
        )
        w = self.weights_[available].astype(float)
        w /= w.sum() + 1e-12
        return np.tensordot(w, preds, axes=(0, 0))

    def adapt_local_maps(self, adaptation_views):
        """Return adapted means/encoders without changing the fitted shared latent."""
        v = len(adaptation_views)
        means = [x.copy() for x in self.means_]
        encoders = [x.copy() for x in self.encoders_]
        pred = np.stack(
            [(adaptation_views[i] - means[i]) @ encoders[i] for i in range(v)]
        )
        residuals = np.asarray([
            np.mean(
                (
                    pred[i]
                    - np.mean(pred[[j for j in range(v) if j != i]], axis=0)
                ) ** 2
            )
            for i in range(v)
        ])
        threshold = self.change_threshold * (np.median(residuals) + 1e-12)
        suspects = np.where(residuals > threshold)[0]
        if len(suspects) > 1:
            suspects = np.asarray([int(np.argmax(residuals))])

        for i in suspects:
            stable = [j for j in range(v) if j != i and j not in suspects]
            target = np.mean(pred[stable], axis=0)
            mu = adaptation_views[i].mean(0)
            encoders[i] = _ridge(
                adaptation_views[i] - mu, target, self.ridge_lambda
            )
            means[i] = mu
        return means, encoders, suspects, residuals

    def transform_with_maps(self, views, means, encoders, available=None):
        if available is None:
            available = list(range(len(views)))
        preds = np.stack([(views[i] - means[i]) @ encoders[i] for i in available])
        w = self.weights_[available].astype(float)
        w /= w.sum() + 1e-12
        return np.tensordot(w, preds, axes=(0, 0))
