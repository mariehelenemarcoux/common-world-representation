import numpy as np

from cwr.frame_realign import (
    detect_changed_view,
    fuse_views,
    realign_local_view,
    ridge,
)


def proper_rotation(rng, d):
    A = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def test_detector_and_realign_reduce_local_frame_shift():
    rng = np.random.default_rng(20260831)

    n_cal = 500
    n_adapt = 180
    n_test = 400
    d = 4
    V = 3

    maps = [proper_rotation(rng, d) for _ in range(V)]

    def obs(z, M):
        return z @ M.T + 0.04 * rng.normal(size=(len(z), d))

    zc = rng.normal(size=(n_cal, d))
    za = rng.normal(size=(n_adapt, d))
    zt = rng.normal(size=(n_test, d))

    cal = [obs(zc, M) for M in maps]
    adapt = [obs(za, M) for M in maps]
    test = [obs(zt, M) for M in maps]

    consensus = sum(cal) / V
    means = [X.mean(0) for X in cal]
    encoders = [
        ridge(X - means[i], consensus, lam=1.5)
        for i, X in enumerate(cal)
    ]

    R = proper_rotation(rng, d)
    adapt_drift = [adapt[0] @ R.T, adapt[1], adapt[2]]
    test_drift = [test[0] @ R.T, test[1], test[2]]

    frozen_stable = fuse_views(test, means, encoders)
    frozen_drift = fuse_views(test_drift, means, encoders)

    ma, ea, suspects, _ = realign_local_view(
        adapt_drift, means, encoders
    )
    adaptive_drift = fuse_views(test_drift, ma, ea)

    frozen_error = np.mean((frozen_drift - frozen_stable) ** 2)
    adaptive_error = np.mean((adaptive_drift - frozen_stable) ** 2)

    assert suspects.tolist() == [0]
    assert adaptive_error < frozen_error


def test_stable_views_do_not_force_realign():
    rng = np.random.default_rng(9876)
    P = rng.normal(size=(3, 200, 5))
    # Make all three predictions close to the same latent consensus.
    center = rng.normal(size=(200, 5))
    P = np.stack([
        center + 0.01 * rng.normal(size=center.shape)
        for _ in range(3)
    ])

    suspects = detect_changed_view(P, threshold_ratio=1.65)

    # With symmetric small noise, the detector should generally remain closed.
    assert len(suspects) == 0
