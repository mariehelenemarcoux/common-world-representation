from pathlib import Path
import numpy as np

RAW = Path("data/raw/mhealth")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

VIEW_SLICES = [slice(0, 5), slice(5, 14), slice(14, 23)]


def feature_block(x):
    return np.concatenate([
        x.mean(0), x.std(0), x.min(0), x.max(0),
        np.sqrt(np.mean(x * x, axis=0)),
    ])


def make_windows(path, subject, win=100, stride=50, purity=0.95):
    a = np.loadtxt(path, dtype=np.float32)
    y = a[:, 23].astype(np.int16)
    x = a[:, :23]
    rows = []
    for start in range(0, len(a) - win + 1, stride):
        yy = y[start:start + win]
        nz = yy[yy > 0]
        if len(nz) < int(purity * win):
            continue
        vals, cnts = np.unique(nz, return_counts=True)
        lab = int(vals[np.argmax(cnts)])
        if np.max(cnts) < int(purity * win):
            continue
        feats = [feature_block(x[start:start + win, sl]) for sl in VIEW_SLICES]
        rows.append((subject, lab, *feats))
    return rows


all_rows = []
for subject in range(1, 11):
    p = RAW / f"mHealth_subject{subject}.log"
    if not p.exists():
        raise FileNotFoundError(p)
    all_rows.extend(make_windows(p, subject))

subjects = np.asarray([r[0] for r in all_rows], dtype=np.int16)
y = np.asarray([r[1] for r in all_rows], dtype=np.int16)
xs = [np.stack([r[2 + i] for r in all_rows]).astype(np.float64) for i in range(3)]

np.savez_compressed(
    OUT / "mhealth_cwr_window_features.npz",
    subjects=subjects, y=y, X0=xs[0], X1=xs[1], X2=xs[2],
)
print(f"saved {len(y)} windows")
