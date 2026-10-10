"""Shared constants and helpers."""
import numpy as np

TERRAINS = ["asphalt", "cork", "grass", "gravel", "lab", "laminate_wood", "pebble", "sand"]  # label order of the dataset
CLASSES = ["HF", "LF", "D", "G"]            # high-friction, low-friction, deformable, granular
TERRAIN_TO_CLASS = [0, 2, 2, 2, 1, 1, 2, 3]  # asphalt->HF, lab+laminate_wood->LF, cork/grass/gravel/pebble->D, sand->G

FEATURE_NAMES = (
    ["peak_amp_sum", "area_sum", "gait_motor_rpm", "gait_ratio_Ts_Tc", "avg_amp_sum", "gait_omega_slow"]
    + [f"peak_taxel{i+1}" for i in range(5)]
    + [f"peak_ratio_taxel{i+1}" for i in range(5)]
    + ["shear_min", "shear_max", "shear_avg"]
    + ["rpm_at_peak_sum", "current_at_peak_sum", "current_avg", "current_range"]
    + [f"avg_force_taxel{i+1}" for i in range(5)]
    + [f"rpm_at_peak_taxel{i+1}" for i in range(5)]
)
assert len(FEATURE_NAMES) == 33
CONTROL_IDX = [2, 3, 5]   # the 3 gait-parameter control inputs (features 3, 4, 6 in Table 1)


def load(path="data/tactile.npz"):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def split_indices(y_terrain, seed=0, n_each=112):
    """Split like the authors: 112 validation + 112 test samples per terrain (896 each), the rest for training."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for t in np.unique(y_terrain):
        idx = rng.permutation(np.where(y_terrain == t)[0])
        va += list(idx[:n_each]); te += list(idx[n_each:2 * n_each]); tr += list(idx[2 * n_each:])
    return np.array(tr), np.array(va), np.array(te)


def plot_confusion(cm, labels, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    plt.xlabel("Predicted label"); plt.ylabel("True label"); plt.title(title)
    plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
