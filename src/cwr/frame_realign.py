"""Minimal local frame-realignment utilities used by the agent smoke test.

This module is intentionally small and deterministic. It captures the
mechanism-level operation that is tested in the smoke benchmark:

1. encode each view into a common latent gauge;
2. measure cross-view disagreement;
3. identify at most one locally inconsistent view;
4. refit only that view into the gauge defined by the stable views.

The external MHEALTH/PAMAP2 benchmark scripts remain the scientific
reference implementations for the published experiments.
"""

from __future__ import annotations

import numpy as np


def ridge(X: np.ndarray, Y: np.ndarray, lam: float = 1.5) -> np.ndarray:
    """Closed-form ridge regression."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    return np.linalg.solve(
        X.T @ X + lam * np.eye(X.shape[1]),
        X.T @ Y,
    )


def encode_views(
    views: list[np.ndarray],
    means: list[np.ndarray],
    encoders: list[np.ndarray],
) -> np.ndarray:
    """Return shape (n_views, n_samples, latent_dim)."""
    return np.stack([
        (np.asarray(views[i]) - means[i]) @ encoders[i]
        for i in range(len(views))
    ])


def disagreement_scores(predictions: np.ndarray) -> np.ndarray:
    """MSE of each latent prediction versus the mean of the other views."""
    V = predictions.shape[0]
    scores = []
    for i in range(V):
        others = [j for j in range(V) if j != i]
        target = predictions[others].mean(axis=0)
        scores.append(np.mean((predictions[i] - target) ** 2))
    return np.asarray(scores)


def detect_changed_view(
    predictions: np.ndarray,
    threshold_ratio: float = 1.65,
) -> np.ndarray:
    """Detect at most one locally inconsistent view."""
    scores = disagreement_scores(predictions)
    med = np.median(scores) + 1e-12
    suspects = np.where(scores > threshold_ratio * med)[0]

    if len(suspects) > 1:
        suspects = np.array([int(np.argmax(scores))])

    return suspects


def realign_local_view(
    adaptation_views: list[np.ndarray],
    means: list[np.ndarray],
    encoders: list[np.ndarray],
    threshold_ratio: float = 1.65,
    lam: float = 1.5,
):
    """Conditionally refit only the detected local view into the old gauge."""
    new_means = [np.asarray(m).copy() for m in means]
    new_encoders = [np.asarray(B).copy() for B in encoders]

    P = encode_views(adaptation_views, new_means, new_encoders)
    scores = disagreement_scores(P)
    suspects = detect_changed_view(P, threshold_ratio=threshold_ratio)

    for i in suspects:
        stable = [j for j in range(len(adaptation_views)) if j != i]
        target = P[stable].mean(axis=0)

        mu = np.asarray(adaptation_views[i]).mean(axis=0)
        B = ridge(np.asarray(adaptation_views[i]) - mu, target, lam=lam)

        new_means[i] = mu
        new_encoders[i] = B

    return new_means, new_encoders, suspects, scores


def fuse_views(
    views: list[np.ndarray],
    means: list[np.ndarray],
    encoders: list[np.ndarray],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted mean of per-view latent predictions."""
    P = encode_views(views, means, encoders)

    if weights is None:
        weights = np.ones(P.shape[0], dtype=float)

    weights = np.asarray(weights, dtype=float)
    weights = weights / (weights.sum() + 1e-12)

    return np.tensordot(weights, P, axes=(0, 0))
