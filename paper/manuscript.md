# Common World Representation for Robust Multiview Learning Under Local Observation-Frame Drift

**Marie-Hélène Marcoux**

## Abstract

Multiview learning systems often assume that the mapping from each sensor or view to a shared representation remains sufficiently stable at deployment. In practice, a local observation frame can change while the underlying world remains unchanged: a wearable sensor can rotate, a camera can move, or a modality can lose access to part of its original signal. We study **Common World Representation (CWR)**, a lightweight multiview mechanism designed to preserve a shared latent representation while conditionally realigning a locally changed perspective.

CWR is not proposed as a generally superior classifier. Its central hypothesis is narrower: when the underlying task remains stable but one local observation mapping changes, local perspective realignment should preserve downstream performance better than globally refitting or freezing the original representation.

We evaluate this hypothesis on two external human-sensor datasets, MHEALTH and PAMAP2. On MHEALTH, CWR does not outperform supervised late fusion under stable conditions, but it is substantially more robust than plain GCCA and a frozen-CWR control under controlled 3D sensor-frame rotations. Under arm-frame rotation, CWR accuracy decreases by only 0.035, compared with 0.279 for plain GCCA. Across three MHEALTH frame/horizon stressors, adaptive CWR shows lower degradation than plain GCCA for all 10 held-out subjects.

The preregistered qualitative prediction is then replicated on PAMAP2 without retuning the CWR core. In stable conditions, CWR accuracy is 0.555, below supervised late fusion at 0.807. Under controlled wrist-frame rotation, CWR remains at 0.557 accuracy, while plain GCCA falls to 0.357. Mean accuracy degradation is approximately -0.001 for CWR versus 0.153 for plain GCCA, 0.203 for ALS, 0.205 for concatenated PCA, and 0.143 for late fusion. CWR degrades less than plain GCCA for all 8 comparable PAMAP2 subjects.

These results support a specific claim: **CWR is a candidate robustness mechanism for multiview systems subject to local observation-frame drift.** The real sensor recordings are external, but the frame-drift and axis-loss perturbations are experimentally injected; naturally occurring reorientation remains to be tested.

---

## 1. Introduction

Multiview systems combine several partial observations of the same underlying process. Typical examples include wearable sensor arrays, multi-camera systems, robot sensor suites, and multimodal inference pipelines. A common assumption in many representation-learning pipelines is that the relationship between each input view and the latent representation remains sufficiently stable between calibration and deployment.

That assumption can fail even when the underlying world does not change.

A sensor may rotate. A camera may move. A modality may lose access to one axis or subspace. The resulting observations can change substantially although the latent physical activity, object, or scene remains the same.

This distinction motivates the central problem studied here:

> **How can a multiview system preserve a shared representation when one local observation frame changes while the underlying world remains stable?**

We refer to the proposed mechanism as **Common World Representation (CWR)**. The intended computational distinction is:

\[
\text{change in world} \neq \text{change in observation map}.
\]

CWR therefore does not treat every large prediction residual as evidence that the shared latent world itself must be relearned. Instead, it estimates whether one local perspective has become inconsistent with the remaining views and, when warranted, realigns that perspective into the existing latent gauge.

The contribution of this work is deliberately narrow. CWR is not claimed to be a general cognitive architecture or a universally better classifier. In fact, our external benchmarks repeatedly show that standard supervised late-fusion systems can outperform CWR in stable conditions. The contribution studied here is **robustness to local view-map drift**.

The paper makes four empirical contributions:

1. It evaluates CWR on real multiview human-sensor data rather than only synthetic latent-variable simulations.
2. It compares adaptive CWR against frozen CWR, plain GCCA, low-rank ALS, concatenated PCA, and supervised late fusion.
3. It tests physically coherent 3D sensor-frame transformations and axis loss while holding the underlying activity unchanged.
4. It replicates the qualitative CWR robustness pattern across MHEALTH and PAMAP2, with the PAMAP2 test performed after the CWR v0.1 mechanism was publicly frozen.

---

## 2. Problem Formulation

Let an underlying latent world state be \(z\), observed through \(V\) local perspectives:

\[
x_i = f_i(z) + \epsilon_i,
\qquad i=1,\dots,V.
\]

A conventional multiview model often learns a joint representation

\[
g(x_1,\dots,x_V) \rightarrow [z].
\]

The deployment problem considered here is not necessarily a change in \(z\), but a change in one local observation map:

\[
f_i \rightarrow \tilde f_i.
\]

Thus,

\[
z \text{ is approximately stable}
\quad\text{while}\quad
x_i \rightarrow \tilde x_i.
\]

The desired behavior is therefore not unrestricted global adaptation. Instead, the system should preserve the common representation when possible while repairing the locally changed mapping.

We summarize this objective as:

\[
\boxed{
\text{WORLD stable}
+
\text{VIEW mapping changes}
\Rightarrow
\text{local view realignment}
}
\]

rather than:

\[
\text{global representation relearning}.
\]

---

## 3. Common World Representation

### 3.1 Shared latent representation

CWR begins from a linear MAXVAR-style generalized canonical correlation analysis (GCCA) representation. Given standardized views \(X_i\), a shared latent matrix \(G\) is estimated from the joint view geometry.

For each view, a linear encoder maps the centered input into the common latent gauge:

\[
\hat G_i = (X_i - \mu_i)B_i.
\]

### 3.2 View reliability

A first shared representation provides a per-view latent residual. View reliability is then estimated as:

\[
r_i = \frac{1}{e_i + 0.05},
\]

where \(e_i\) is the view-specific latent residual.

The frozen CWR v0.1 implementation rescales reliability as:

\[
\tilde r_i =
\left(
\frac{r_i}{\bar r}
\right)^{1.5}.
\]

A weighted GCCA fit then constructs the calibration representation.

### 3.3 Local disagreement detector

For an unlabeled deployment adaptation block, each view predicts the common latent representation:

\[
P_i = (X_i - \mu_i)B_i.
\]

For view \(i\), disagreement is measured relative to the other perspectives:

\[
d_i =
\operatorname{MSE}
\left(
P_i,
\frac{1}{V-1}
\sum_{j\neq i} P_j
\right).
\]

A view is considered locally inconsistent when:

\[
d_i > 1.65 \cdot \operatorname{median}(d_1,\dots,d_V).
\]

With three views, at most one perspective is realigned.

### 3.4 Local realignment

If view \(i\) is identified as inconsistent, CWR preserves the latent gauge of the remaining perspectives and relearns only the mapping from the changed view into that gauge.

Let

\[
T_i =
\frac{1}{V-1}
\sum_{j\neq i} P_j.
\]

The new local encoder is obtained by ridge regression:

\[
B_i' =
\arg\min_B
\left\|
(X_i-\mu_i')B - T_i
\right\|_2^2
+
\lambda\|B\|_2^2.
\]

No global latent refit is required.

This architecture embodies the operational distinction:

\[
\boxed{
\text{observation} \neq \text{structure observed}
}
\]

and specifically:

\[
\boxed{
\text{local frame drift} \neq \text{world change}
}.
\]

---

## 4. Experimental Design

### 4.1 General protocol

Both external benchmarks use leave-one-subject-out evaluation. For each held-out participant:

- the remaining subjects provide calibration data;
- the held-out subject is never used to fit the supervised classifier;
- a small unlabeled adaptation block is available for representation adaptation;
- inference is performed on the remaining windows;
- statistical inference is conducted at the subject level rather than the window level.

The primary performance metric is classification accuracy. Macro-F1 is reported as a secondary metric.

The key robustness quantity is:

\[
\Delta_{\text{drift}}
=
Acc_{\text{stable}}
-
Acc_{\text{transformed}}.
\]

Lower degradation is better.

### 4.2 Baselines

The primary comparison methods are:

- **CWR**: weighted shared representation with conditional local realignment;
- **CWR frozen**: the same calibrated CWR representation without deployment realignment;
- **plain GCCA**: GCCA fully refit on the same unlabeled adaptation block, followed by affine gauge alignment;
- **low-rank ALS**: global low-rank factorization with adaptation and gauge alignment;
- **concatenated PCA**: PCA on concatenated views with adaptation and gauge alignment;
- **supervised late fusion**: one classifier per view with averaged class probabilities.

The supervised late-fusion baseline is intentionally strong. It tests whether explicit construction of a common latent representation is necessary for high downstream classification performance.

---

## 5. MHEALTH

### 5.1 Dataset and views

MHEALTH contains wearable recordings from multiple body locations. The benchmark uses three perspectives corresponding to:

- chest;
- left ankle;
- right arm.

Window-level features are constructed from fixed, label-free summaries of each sensor block.

The evaluation uses 10 subjects in leave-one-subject-out fashion.

### 5.2 Natural cross-subject evaluation

Under natural subject heterogeneity, CWR is not the best activity classifier. Mean accuracy is approximately 0.764, while supervised late fusion reaches approximately 0.932.

This result is important because it falsifies a broader interpretation in which CWR would be expected to dominate standard supervised fusion in general.

However, in a held-out sensor reconstruction task, CWR reconstructs an absent body-location view with lower mean MSE than ALS and PCA:

\[
MSE_{CWR} = 0.598,
\quad
MSE_{ALS} = 0.757,
\quad
MSE_{PCA} = 0.757.
\]

CWR outperforms ALS and PCA on this reconstruction metric for all 10 held-out subjects.

### 5.3 Controlled frame/horizon stress

To isolate the local-frame hypothesis, physically coherent perturbations are applied to the right-arm sensor while leaving the activity sequence unchanged.

The tested perturbations are:

- proper 3D rotation of the arm sensor frame;
- loss of one axis;
- 3D rotation plus axis loss.

The same 3D rotation is applied consistently to the tri-axial sensor groups.

### 5.4 MHEALTH results

Mean accuracy:

| Condition | CWR | CWR frozen | plain GCCA |
|---|---:|---:|---:|
| Stable | 0.760 | 0.760 | 0.655 |
| Arm rotation | 0.725 | 0.527 | 0.376 |
| Axis loss | 0.675 | 0.646 | 0.480 |
| Rotation + axis loss | 0.732 | 0.454 | 0.367 |

Accuracy degradation relative to stable:

| Condition | CWR | CWR frozen | plain GCCA |
|---|---:|---:|---:|
| Arm rotation | 0.035 | 0.232 | 0.279 |
| Axis loss | 0.085 | 0.114 | 0.175 |
| Rotation + axis loss | 0.028 | 0.306 | 0.288 |

For arm rotation, CWR shows lower degradation than plain GCCA for all 10 subjects. The one-sided subject-level sign test and Wilcoxon signed-rank test both yield \(p=0.000977\).

Across all three stressors, CWR shows lower average degradation than plain GCCA for all 10 subjects, with mean degradation advantage 0.198 and \(p=0.000977\) for both the sign and Wilcoxon tests.

Adaptive CWR also outperforms the frozen-CWR control across all three stressors for all 10 subjects, with mean degradation advantage 0.168.

### 5.5 Detector behavior

The arm-view detector realigns the arm in approximately:

- 0.0% of arm-rotation conditions;
- 0.0% of rotation-plus-axis-loss conditions;
- 0.0% of axis-loss-only conditions;
- 0.0% of stable conditions.

This supports the interpretation that the gain is not produced by indiscriminate continuous adaptation.

---

## 6. PAMAP2 Replication

### 6.1 Dataset and frozen prediction

PAMAP2 provides a second external wearable-sensor benchmark with three IMU locations:

- wrist;
- chest;
- ankle.

The primary comparable cohort uses subjects 101–108. Subject 109 is excluded from the primary multiclass analysis because its available Protocol recording is not comparable to the remaining subjects for the selected endpoint.

Before evaluating PAMAP2, the qualitative prediction was fixed:

\[
\boxed{
\text{CWR need not be best in stable conditions}
}
\]

but:

\[
\boxed{
\Delta_{drift}^{CWR}
<
\Delta_{drift}^{baseline}
}
\]

when one local sensor frame changes.

No CWR hyperparameter was retuned for PAMAP2.

### 6.2 Controlled perturbations

The wrist IMU is perturbed while the chest and ankle views remain unchanged.

Conditions are:

- stable;
- proper 3D wrist-frame rotation;
- loss of one wrist axis;
- 3D rotation plus axis loss.

The same 3D rotation is applied consistently to the wrist accelerometer, gyroscope, and magnetometer vector groups.

### 6.3 PAMAP2 results

Mean accuracy:

| Condition | CWR | CWR frozen | GCCA | ALS | PCA | Late fusion |
|---|---:|---:|---:|---:|---:|---:|
| Stable | 0.555 | 0.553 | 0.511 | 0.639 | 0.622 | 0.807 |
| Wrist rotation | 0.557 | 0.411 | 0.357 | 0.435 | 0.417 | 0.663 |
| Axis loss | 0.555 | 0.538 | 0.462 | 0.567 | 0.557 | 0.743 |
| Rotation + axis loss | 0.556 | 0.439 | 0.361 | 0.430 | 0.413 | 0.662 |

Accuracy degradation relative to stable:

| Condition | CWR | CWR frozen | GCCA | ALS | PCA | Late fusion |
|---|---:|---:|---:|---:|---:|---:|
| Wrist rotation | -0.001 | 0.142 | 0.153 | 0.203 | 0.205 | 0.143 |
| Axis loss | 0.000 | 0.015 | 0.048 | 0.072 | 0.065 | 0.064 |
| Rotation + axis loss | -0.000 | 0.114 | 0.150 | 0.209 | 0.209 | 0.144 |

For wrist rotation, adaptive CWR degrades less than each comparison baseline for all 8 subjects. Against plain GCCA:

\[
\Delta_{advantage} = 0.155,
\]

with 8/8 subject-level wins, one-sided sign-test \(p=0.003906\), and one-sided Wilcoxon \(p=0.003906\).

Aggregated across the three stressors, CWR shows lower degradation than plain GCCA for all 8 subjects, with mean degradation advantage 0.118. Adaptive CWR also outperforms frozen CWR on 7/8 subjects with mean degradation advantage 0.091 and one-sided Wilcoxon \(p=0.007812\).

### 6.4 Detector replication

The wrist detector realigns the perturbed wrist in approximately:

- 100.0% of rotation conditions;
- 91.7% of rotation-plus-axis-loss conditions;
- 41.7% of axis-loss-only conditions;
- 4.2% of stable conditions.

This reproduces the qualitative MHEALTH pattern.

---

## 7. Cross-Dataset Interpretation

The most important result is not that CWR achieves the highest stable-condition accuracy. It does not.

On PAMAP2, supervised late fusion reaches 0.807 stable accuracy compared with 0.555 for CWR. MHEALTH shows the same general pattern: direct supervised fusion can be substantially better for the downstream task.

The replicated signal is instead an interaction between **method** and **observation-frame drift**.

Across both MHEALTH and PAMAP2:

1. CWR is not generally superior in stable conditions.
2. A local 3D sensor-frame rotation causes a large degradation for frozen/global baselines.
3. Adaptive CWR preserves substantially more downstream performance.
4. The local detector activates primarily under the intended frame-change conditions.
5. Simple axis loss produces a weaker and less consistent adaptive-CWR advantage than full frame rotation.

This last point is important. It suggests that CWR is not merely a generic corruption-resistance method. The strongest evidence is specifically associated with **changes in the local observation mapping**.

The empirical claim supported by the current results is therefore:

\[
\boxed{
\text{CWR provides robustness to local observation-frame drift
in multiview sensor systems.}
}
\]

---

## 8. Ablation and Failure Analysis

### 8.1 CWR versus frozen CWR

The frozen-CWR ablation isolates the role of online local realignment.

On both external datasets, adaptive CWR shows substantially lower degradation under rotation-based perturbations than the frozen representation.

This indicates that the result cannot be attributed solely to the weighted shared latent representation.

### 8.2 CWR versus GCCA

Plain GCCA is an especially important comparison because CWR uses a GCCA-style shared latent space.

Under missing-view conditions, GCCA can be nearly as robust as CWR. However, under local frame rotation, CWR consistently outperforms plain GCCA in both MHEALTH and PAMAP2.

This suggests that the distinctive mechanism is not merely construction of a shared representation, but **conditional repair of the changed local map while preserving the existing common gauge**.

### 8.3 Stable-condition failure to dominate

CWR is often worse than strong supervised baselines in stable conditions. This is not treated as an anomaly to be tuned away. It is part of the empirical characterization of the method.

CWR trades some direct supervised task optimization for robustness of the shared representation under perspective drift.

### 8.4 Axis loss

Axis loss by itself produces a weaker adaptive advantage than full 3D frame rotation.

This is consistent with the detector's behavior: a subspace reduction does not always create the same cross-view inconsistency signature as a full local gauge change.

---

## 9. Limitations

Several limitations constrain the interpretation of the present results.

### 9.1 Injected perturbations

MHEALTH and PAMAP2 contain real human sensor measurements, but the tested frame rotations and axis-loss events are injected experimentally. Therefore, the current experiments test controlled robustness, not naturally documented sensor reorientation.

### 9.2 Small number of subjects

The inferential unit is the subject. MHEALTH provides 10 subjects and the primary comparable PAMAP2 cohort provides 8. The strong paired consistency is encouraging, but larger cohorts are required.

### 9.3 Linear implementation

The current CWR implementation is intentionally simple and largely linear. The results do not establish that the same mechanism will scale to high-dimensional image, video, language, or nonlinear multimodal representations.

### 9.4 Hand-designed detector

The view-change detector uses a fixed residual threshold and a hand-designed realignment rule. These choices were frozen before the PAMAP2 replication, which reduces post-hoc tuning concerns, but they are not theoretically optimal.

### 9.5 No claim of general superiority

CWR should not be interpreted as a generally superior multiview classifier. Supervised late fusion remains substantially stronger in stable absolute classification performance in the evaluated wearable-sensor tasks.

### 9.6 No claim about consciousness or ethics

The experiments evaluate a computational robustness mechanism. They do not establish claims about consciousness, moral agency, universal ethics, or phenomenology.

---

## 10. Discussion

The central computational lesson from the experiments is a distinction between the world and a local map of the world.

A system that treats every large discrepancy as evidence of global world change risks unnecessary relearning. Conversely, a system that assumes all local mappings are permanently stable can fail catastrophically when a sensor frame shifts.

CWR introduces an intermediate option:

\[
\boxed{
\text{detect local perspective inconsistency}
\rightarrow
\text{repair local map}
\rightarrow
\text{preserve common latent gauge}
}
\]

The cross-dataset replication suggests that this decomposition is useful in at least two wearable-sensor environments.

This can also be framed as a restricted form of gauge robustness. The latent representation need not be identified with any one sensor coordinate system. If a single view changes coordinates while the underlying shared structure remains compatible with the other views, the local mapping can be updated without replacing the common representation.

This interpretation suggests several future directions:

- naturally occurring sensor reorientation datasets;
- multi-camera systems with camera relocation;
- robot sensor suites with calibration drift;
- nonlinear neural encoders with explicit local realignment;
- change-point detection for time-varying perspective reliability;
- uncertainty-aware decisions about when to realign versus globally revise.

The next empirical priority should not be further tuning on MHEALTH or PAMAP2. It should be testing the frozen mechanism on additional domains with independent data-generation processes.

---

## 11. Conclusion

This work studies a narrow problem in multiview learning: preserving a shared representation when one local observation frame changes while the underlying world remains stable.

CWR combines a shared multiview latent representation with a conditional local disagreement detector and view-specific realignment mechanism.

Across two external wearable-sensor datasets, MHEALTH and PAMAP2, CWR does not emerge as a generally superior classifier in stable conditions. However, under controlled 3D sensor-frame drift, adaptive CWR consistently shows substantially lower degradation than plain GCCA, low-rank ALS, concatenated PCA, frozen CWR, and, in PAMAP2, even a strong supervised late-fusion baseline in terms of **relative robustness**.

The PAMAP2 experiment reproduces the qualitative prediction after the CWR v0.1 mechanism had been frozen publicly, reducing the scope for post-hoc adaptation of the method to the second dataset.

The current evidence therefore supports the following limited conclusion:

\[
\boxed{
\text{CWR is a candidate robustness mechanism
for multiview systems under local observation-frame drift.}
}
\]

Further validation should focus on naturally occurring frame changes, larger subject cohorts, and nonlinear multiview systems.

---

## Reproducibility

The public repository contains:

- the frozen CWR implementation;
- MHEALTH benchmark scripts;
- PAMAP2 preparation and benchmark scripts;
- benchmark CSV outputs;
- subject-level paired statistical audits.

The raw MHEALTH and PAMAP2 datasets are not redistributed in the repository and should be obtained from their original sources.

---

## References to add if ever submited

This draft intentionally leaves the formal bibliography incomplete. A submission-ready version, which will be adjusted if ever submitted will add and verify references for:

- GCCA / MAXVAR GCCA;
- multiview representation learning;
- sensor-domain adaptation and calibration drift;
- MHEALTH dataset;
- PAMAP2 dataset;
- wearable human-activity recognition;
- robust sensor fusion and missing-modality learning.

