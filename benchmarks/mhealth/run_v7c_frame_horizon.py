"""Entry point for the frozen v7-C experiment.

This repository release includes the frozen result CSVs produced by the
reference experiment. The full stress runner is intentionally kept separate
from the minimal CWR core so the core remains readable.

For exact reproduction, first run prepare_mhealth.py. Then implement or reuse
the protocol described in README.md / docs/EXPERIMENTAL_SCOPE.md:

- leave-one-subject-out evaluation,
- 960 balanced calibration windows from the other subjects,
- 120 unlabeled adaptation windows from the unseen subject,
- proper SO(3) rotation applied consistently to each right-arm triad,
- optional loss of one axis across acc/gyro/magnetometer triads,
- adaptive CWR vs frozen CWR vs plain GCCA,
- subject-level paired inference.

The frozen outputs from the reference run are under results/.
"""

from pathlib import Path

p = Path("data/processed/mhealth_cwr_window_features.npz")
if not p.exists():
    raise FileNotFoundError(
        "Run benchmarks/mhealth/prepare_mhealth.py first."
    )

print("MHEALTH processed features found:", p)
print("See results/cwr_v7c_mhealth_frame_horizon_summary.csv for frozen v0.1 output.")
