# Experimental scope

## What is supported so far

The current experiments support a candidate mechanism for multiview robustness under local changes in observation maps.

The strongest external result to date uses the UCI MHEALTH dataset. A held-out subject is never used for supervised fitting. Controlled, physically coherent perturbations are applied to the tri-axial right-arm sensor:

- 3D frame rotation,
- loss of one axis,
- 3D rotation plus loss of one axis.

Adaptive CWR is compared with frozen CWR and plain GCCA using the same unlabeled adaptation budget.

## What is not supported

The experiments do not establish:

- general superiority over standard machine-learning methods,
- general intelligence,
- consciousness or phenomenology,
- universal morality or ethical agency,
- metaphysical claims.

## Interpretation discipline

A good working statement is:

> CWR is a candidate multiview robustness mechanism for local frame/horizon drift.

A statement that is currently too strong is:

> CWR is a generally superior cognitive architecture.

## Validation roadmap

1. Freeze v0.1.
2. Reproduce MHEALTH from a clean environment.
3. Replicate on a second external dataset without retuning the core mechanism.
4. Add stronger contemporary baselines where feasible.
5. Only then consider neural implementations or a formal paper.
