# CWR — Common World Representation

**Status:** research prototype / v0.1

CWR is an experimental multiview representation method designed for **robustness to local changes of observation frame or information horizon**. It is not presented as a generally superior learning architecture.

## Main empirical claim

The current evidence supports the narrower hypothesis that CWR can preserve a shared latent representation when one local view changes its mapping to the world.

On the external MHEALTH dataset:

- CWR is **not** the best general supervised activity classifier.
- Under complete sensor loss, CWR is substantially more robust than PCA and low-rank ALS, but roughly comparable to GCCA.
- Under controlled physically coherent 3D frame rotation of the right-arm sensor, CWR degrades much less than frozen CWR and plain GCCA.
- The benefit is therefore concentrated in **local view/frame drift**, not general task performance.

## Core idea

For views \(x_i\), CWR separates a shared representation from local observation maps:

```text
views -> local maps -> conditional reliability -> common representation [z]
```

When a view appears to have changed, CWR attempts **local realignment** instead of globally relearning the shared representation.

A useful conceptual distinction is:

```text
observation != observed structure
```

## Repository layout

```text
CWR_repo/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── src/cwr/
│   ├── __init__.py
│   └── core.py
├── benchmarks/mhealth/
│   ├── README.md
│   ├── prepare_mhealth.py
│   └── run_v7c_frame_horizon.py
├── results/
│   └── selected frozen CSV outputs
└── docs/
    └── EXPERIMENTAL_SCOPE.md
```

## Reproducing the MHEALTH benchmark

1. Download MHEALTH from the UCI Machine Learning Repository.
2. Extract `mHealth_subject1.log` ... `mHealth_subject10.log` into:

```text
data/raw/mhealth/
```

3. Install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

4. Prepare window features:

```bash
python benchmarks/mhealth/prepare_mhealth.py
```

5. Run the frozen frame/horizon benchmark:

```bash
python benchmarks/mhealth/run_v7c_frame_horizon.py
```

## Frozen v0.1 parameters

The parameters below are frozen from the exploratory phase and should not be retuned on MHEALTH when reproducing the reported result:

```text
latent dimension K = 6
GCCA ridge lambda = 1.5
reliability power = 1.5
local change threshold = 1.65 * median disagreement
adaptation budget = 120 unlabeled windows per unseen subject
```

## Reported MHEALTH v7-C result

Mean classification accuracy:

| Scenario | CWR | Frozen CWR | Plain GCCA |
|---|---:|---:|---:|
| Stable | 0.7596 | 0.7596 | 0.6549 |
| Arm 3D rotation | **0.7250** | 0.5274 | 0.3755 |
| Arm axis loss | **0.6746** | 0.6456 | 0.4801 |
| Rotation + axis loss | **0.7317** | 0.4535 | 0.3666 |

Across the three stressors, adaptive CWR showed lower accuracy degradation than frozen CWR and plain GCCA for **10/10 held-out subjects** in the aggregate subject-level analysis.

See `results/` for the frozen CSV summaries and audits.

## Scientific scope

This repository does **not** claim that CWR demonstrates consciousness, ethics, universal morality, or a general theory of intelligence. The current evidence concerns a much narrower machine-learning question: robustness of shared multiview representations under local observation-map changes.

See [`docs/EXPERIMENTAL_SCOPE.md`](docs/EXPERIMENTAL_SCOPE.md).

## Next validation step

The next meaningful test is replication on a second external multiview or multisensor dataset with the CWR v0.1 mechanism and parameters frozen before evaluation.

## License

## PAMAP2 external replication

To test whether the MHEALTH result generalized beyond a single dataset, CWR v0.1 was evaluated on the PAMAP2 Physical Activity Monitoring dataset without changing the core CWR mechanism or its frozen hyperparameters.

The primary comparison used subjects 101–108 in a leave-one-subject-out protocol. Each subject was treated as an unseen test individual. The three PAMAP2 IMUs — wrist, chest, and ankle — were treated as three distinct perspectives of the same underlying physical activity.

The main hypothesis was defined before evaluating PAMAP2:

- CWR was not expected to outperform all baselines under stable observation conditions.
- CWR was expected to degrade less when one local sensor observation frame changed.

Controlled, physically coherent perturbations were applied to the wrist IMU:
- 3D rotation of the local sensor frame;
- loss of one axis;
- 3D rotation combined with loss of one axis.

The same rotation was applied consistently to the accelerometer, gyroscope, and magnetometer triads of the perturbed sensor.

### Mean accuracy

| Condition | CWR | CWR frozen | GCCA | ALS | PCA | Late fusion |
|---|---:|---:|---:|---:|---:|---:|
| Stable | 0.556 | 0.553 | 0.511 | 0.639 | 0.622 | 0.807 |
| Wrist rotation | 0.557 | 0.411 | 0.357 | 0.435 | 0.417 | 0.663 |
| Axis loss | 0.555 | 0.538 | 0.463 | 0.567 | 0.557 | 0.743 |
| Rotation + axis loss | 0.556 | 0.439 | 0.361 | 0.430 | 0.413 | 0.662 |

### Accuracy degradation relative to stable condition

| Condition | CWR | CWR frozen | GCCA | ALS | PCA | Late fusion |
|---|---:|---:|---:|---:|---:|---:|
| Wrist rotation | -0.001 | 0.142 | 0.154 | 0.203 | 0.205 | 0.144 |
| Axis loss | 0.000 | 0.015 | 0.048 | 0.072 | 0.065 | 0.064 |
| Rotation + axis loss | 0.000 | 0.114 | 0.150 | 0.209 | 0.209 | 0.144 |

Across the three perturbation conditions, CWR showed substantially lower degradation than GCCA, ALS, PCA, and the frozen CWR control.

For 3D wrist rotation, CWR showed lower degradation than every comparison baseline for all 8 evaluated subjects.

The wrist realignment detector triggered in:
- 100% of wrist-rotation conditions;
- 91.7% of rotation-plus-axis-loss conditions;
- 41.7% of axis-loss-only conditions;
- 4.2% of stable conditions.

These results replicate the qualitative pattern previously observed on MHEALTH: CWR is not a generally superior classifier under stable conditions, but its local realignment mechanism appears to provide substantial robustness when a single observation frame changes while the underlying activity remains the same.

### Scope of this result

The PAMAP2 measurements are real sensor data collected from human participants. However, the frame rotations and axis-loss perturbations used in this benchmark were experimentally injected. They should therefore be interpreted as controlled robustness tests rather than naturally observed sensor reorientation events.

MIT. Dataset licenses remain those of their original providers; MHEALTH data are not redistributed in this repository.
