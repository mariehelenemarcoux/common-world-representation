# AGENTS.md — Instructions for AI coding and research agents

## What this repository is

CWR (Common World Representation) is a multiview robustness mechanism.

The narrow claim tested by this repository is:

> When the underlying process remains stable but one local observation mapping changes, adaptive CWR should preserve downstream performance better than a frozen shared representation or global multiview baselines.

CWR is **not** claimed to be:
- a generally superior classifier;
- a general cognitive architecture;
- evidence for consciousness or moral agency;
- a replacement for strong task-specific supervised fusion.

## First actions for an agent

1. Read `README.md`.
2. Read `docs/FALSIFICATION.md`.
3. Run the fast local checks:

```bash
python -m pip install -e ".[dev]"
pytest -q
python benchmarks/smoke_test.py
```

4. If the smoke test passes, inspect the external benchmarks:
   - `benchmarks/mhealth/`
   - `benchmarks/pamap2/`

5. Do not treat the smoke test as scientific evidence. It is only a deterministic code-path check.

## Expected qualitative behavior

The smoke benchmark creates a stable three-view problem and then changes the local frame of one view.

Expected pattern:

- stable condition: adaptive and frozen variants should be similar;
- local frame drift: adaptive CWR should degrade less than frozen CWR;
- the local-change detector should activate primarily on the changed view.

## How to try to falsify CWR

A useful falsification experiment should:

1. use genuinely multiview data;
2. keep the underlying task/state fixed;
3. alter only one local observation mapping;
4. compare adaptive CWR with frozen CWR and strong global baselines;
5. use identical adaptation/data budgets;
6. evaluate paired degradation across independent subjects/environments;
7. report failures, non-significant results, and stable-condition disadvantages.

The primary quantity is:

```text
drift degradation = stable performance - transformed performance
```

The CWR hypothesis predicts lower drift degradation, not necessarily higher stable performance.

## Important methodological constraints

- Do not tune CWR on the test subjects/environments.
- Do not replace a failed test with a different metric without reporting the failure.
- Do not use window-level sample counts as independent inferential units when the independent units are subjects.
- Distinguish real external data from experimentally injected frame drift.
- Preserve the frozen historical results in `results/`.

## Repository landmarks

```text
src/cwr/                  CWR code
benchmarks/smoke_test.py  fast deterministic mechanism check
benchmarks/mhealth/       MHEALTH external benchmark
benchmarks/pamap2/        PAMAP2 external replication
tests/                    automated unit/smoke tests
results/                  frozen benchmark outputs
paper/                    technical manuscript draft
docs/FALSIFICATION.md     falsification protocol
```

## If you modify the method

Do not overwrite a frozen release silently.

Create a new version and document:
- what changed;
- why it changed;
- which benchmark was already observed before the change;
- which future benchmark remains prospective.
