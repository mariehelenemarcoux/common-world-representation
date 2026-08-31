# PAMAP2 benchmark

This directory contains the PAMAP2 replication benchmark for CWR.

## Dataset

PAMAP2 Physical Activity Monitoring Dataset.

The benchmark uses the Protocol recordings from subjects 101–108.

The three IMU locations are treated as three perspectives:

- wrist
- chest
- ankle

Subject 109 is excluded from the primary multiclass benchmark because the available Protocol recording is not comparable to the other subjects for the selected endpoint.

## Evaluation protocol

Leave-one-subject-out evaluation is used.

For each test subject:

1. The remaining subjects provide calibration data.
2. CWR and baseline representations are fit without using the held-out subject labels.
3. A small unlabeled adaptation block from the unseen subject is provided.
4. Evaluation is performed on the remaining windows.

The frozen CWR v0.1 mechanism is used without retuning.

## Conditions

The following conditions are evaluated:

- stable
- 3D wrist-frame rotation
- wrist axis loss
- 3D wrist-frame rotation + axis loss

The 3D rotation is applied consistently to the wrist accelerometer, gyroscope, and magnetometer triads.

## Baselines

- frozen CWR
- plain GCCA
- low-rank ALS
- concatenated PCA
- supervised late fusion

## Primary hypothesis

CWR is not expected to outperform all baselines under stable conditions.

The primary hypothesis is that CWR should show lower degradation when one local observation frame changes while the underlying physical activity remains unchanged.

## Important limitation

The PAMAP2 sensor measurements are real.

The frame-drift and axis-loss perturbations are experimentally injected and are not naturally occurring sensor failures documented in the original PAMAP2 collection.
The PAMAP2 sensor measurements are real.

The frame-drift and axis-loss perturbations are experimentally injected and are not naturally occurring sensor failures documented in the original PAMAP2 collection.
