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

MIT. Dataset licenses remain those of their original providers; MHEALTH data are not redistributed in this repository.
