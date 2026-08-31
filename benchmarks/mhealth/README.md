# MHEALTH benchmark

The raw MHEALTH dataset is intentionally not redistributed here.

Download it from the UCI Machine Learning Repository and place:

```text
mHealth_subject1.log
...
mHealth_subject10.log
```

under:

```text
data/raw/mhealth/
```

The benchmark uses 100-sample windows with stride 50 and retains windows with at least 95% purity for a nonzero activity label.
