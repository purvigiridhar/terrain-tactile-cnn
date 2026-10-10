"""Live demo: pick random held-out steps, show the 6 tactile channels and the model's predictions."""
import argparse, numpy as np, torch
import matplotlib
import matplotlib.pyplot as plt
from common import load, split_indices, TERRAINS, CLASSES
from train_cnn import TerrainCNN, VARIANTS

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--variant", default="all")
ap.add_argument("--seed", type=int, default=None)
a = ap.parse_args()

d = load()
_, _, te = split_indices(d["y_terrain"])
rng = np.random.default_rng(a.seed)
models = {}
for label, names in [("class", CLASSES), ("terrain", TERRAINS)]:
    ck = torch.load(f"outputs/models/{a.variant}_{label}.pt", weights_only=False)
    m = TerrainCNN(ck["n_feats"], len(names)); m.load_state_dict(ck["state"]); m.eval()
    models[label] = (m, ck, names)

for i in rng.choice(te, a.n, replace=False):
    res = {}
    for label, (m, ck, names) in models.items():
        s = ck["stats"]
        x = torch.from_numpy(((d["X"][i:i+1] - s["mu"]) / s["sd"])[:, :, None, :].astype(np.float32))
        f = None
        if s["fmu"] is not None:
            f = torch.from_numpy(((d["feats"][i:i+1][:, VARIANTS[a.variant]] - s["fmu"]) / s["fsd"]).astype(np.float32))
        with torch.no_grad():
            p = torch.softmax(m(x, f), 1)[0]
        res[label] = (names[int(p.argmax())], float(p.max()))
    t_true, c_true = TERRAINS[d["y_terrain"][i]], CLASSES[d["y_class"][i]]
    print(f"step {i}: TRUE {t_true} ({c_true}) | PRED terrain {res['terrain'][0]} ({res['terrain'][1]:.0%}), class {res['class'][0]} ({res['class'][1]:.0%})")
    fig, ax = plt.subplots(figsize=(7, 3))
    for k in range(6):
        ax.plot(d["X"][i, k], label=("normal taxel %d" % (k + 1)) if k < 5 else "shear taxel")
    ax.set_title(f"true: {t_true} | predicted: {res['terrain'][0]}"); ax.set_xlabel("time sample"); ax.legend(fontsize=6, ncol=3)
    plt.tight_layout(); plt.savefig(f"outputs/demo_step_{i}.png", dpi=130)
    plt.show() if matplotlib.get_backend().lower() != "agg" else plt.close()
