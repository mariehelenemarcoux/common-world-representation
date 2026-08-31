#!/usr/bin/env python3
"""Fast deterministic CWR mechanism smoke test.

This is NOT scientific evidence. It is a small synthetic code-path check
designed for CI and AI agents.

Expected behavior:
- stable: frozen and adaptive are similar;
- frame drift: adaptive degrades less than frozen;
- changed view is detected.
"""

from __future__ import annotations

import numpy as np

from cwr.frame_realign import ridge, realign_local_view, fuse_views


def proper_rotation(rng: np.random.Generator, d: int) -> np.ndarray:
    A = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def make_problem(seed: int = 1234):
    rng = np.random.default_rng(seed)

    n_cal = 600
    n_adapt = 160
    n_test = 500
    latent_dim = 4
    view_dim = 4
    V = 3

    def latent(n):
        return rng.normal(size=(n, latent_dim))

    z_cal = latent(n_cal)
    z_adapt = latent(n_adapt)
    z_test = latent(n_test)

    maps = [proper_rotation(rng, view_dim) for _ in range(V)]

    def observe(z, M):
        return z @ M.T + 0.06 * rng.normal(size=(len(z), view_dim))

    cal = [observe(z_cal, M) for M in maps]
    adapt_stable = [observe(z_adapt, M) for M in maps]
    test_stable = [observe(z_test, M) for M in maps]

    # Calibration consensus: a fixed gauge learned from stable multiview data.
    # For this smoke test, view 1 and 2 define a simple consensus target.
    # The test is mechanism-level only; external benchmarks provide evidence.
    consensus_cal = (cal[0] + cal[1] + cal[2]) / 3.0

    means = [X.mean(axis=0) for X in cal]
    encoders = [
        ridge(X - means[i], consensus_cal, lam=1.5)
        for i, X in enumerate(cal)
    ]

    # Downstream target: sign of a stable calibration consensus direction.
    w = rng.normal(size=latent_dim)
    y_test = (z_test @ w > 0).astype(int)

    # A large proper rotation changes only local view 0.
    R = proper_rotation(rng, view_dim)
    adapt_drift = [adapt_stable[0] @ R.T, adapt_stable[1], adapt_stable[2]]
    test_drift = [test_stable[0] @ R.T, test_stable[1], test_stable[2]]

    return (
        means, encoders,
        adapt_stable, test_stable,
        adapt_drift, test_drift,
        y_test
    )


def nearest_linear_accuracy(G_train, y_train, G_test, y_test):
    # Tiny dependency-free linear classifier via ridge regression on {-1,+1}.
    t = 2 * y_train - 1
    B = ridge(
        np.c_[np.ones(len(G_train)), G_train],
        t[:, None],
        lam=0.5,
    )
    score = np.c_[np.ones(len(G_test)), G_test] @ B
    pred = (score.ravel() > 0).astype(int)
    return float((pred == y_test).mean())


def run_smoke(seed: int = 1234):
    rng = np.random.default_rng(seed)

    (
        means, encoders,
        adapt_stable, test_stable,
        adapt_drift, test_drift,
        y_test,
    ) = make_problem(seed)

    # Build a small synthetic downstream calibration set from stable fused views.
    n_train = 700
    z_train = rng.normal(size=(n_train, 4))
    w = rng.normal(size=4)
    y_train = (z_train @ w > 0).astype(int)

    # For the smoke classifier, construct a stable latent dataset in the same
    # fused gauge using independently generated view observations.
    view_maps = [np.eye(4), np.eye(4), np.eye(4)]
    train_views = [
        z_train @ M.T + 0.06 * rng.normal(size=(n_train, 4))
        for M in view_maps
    ]
    G_train = np.mean(train_views, axis=0)

    # Accuracy is secondary here; latent continuity is the primary smoke metric.
    G_stable_frozen = fuse_views(test_stable, means, encoders)
    G_drift_frozen = fuse_views(test_drift, means, encoders)

    ma, ea, suspects, scores = realign_local_view(
        adapt_drift, means, encoders
    )
    G_drift_adaptive = fuse_views(test_drift, ma, ea)

    # Compare drift relative to the frozen stable representation.
    frozen_shift = float(np.mean((G_drift_frozen - G_stable_frozen) ** 2))
    adaptive_shift = float(np.mean((G_drift_adaptive - G_stable_frozen) ** 2))

    result = {
        "suspects": suspects.tolist(),
        "scores": scores.tolist(),
        "frozen_latent_shift": frozen_shift,
        "adaptive_latent_shift": adaptive_shift,
        "relative_improvement": (
            (frozen_shift - adaptive_shift) / (frozen_shift + 1e-12)
        ),
    }

    return result


def main():
    r = run_smoke()

    print("CWR AGENT SMOKE TEST")
    print("--------------------")
    print("detected changed view:", r["suspects"])
    print("frozen latent shift:  ", f'{r["frozen_latent_shift"]:.6f}')
    print("adaptive latent shift:", f'{r["adaptive_latent_shift"]:.6f}')
    print("relative improvement: ", f'{100*r["relative_improvement"]:.2f}%')

    passed = (
        r["suspects"] == [0]
        and r["adaptive_latent_shift"] < r["frozen_latent_shift"]
        and r["relative_improvement"] > 0.20
    )

    print("RESULT:", "PASS" if passed else "FAIL")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
