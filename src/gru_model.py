"""GRU terrain classifier (paper's future-work idea): same data/split/features as the CNN, recurrent feature extractor."""
import argparse, copy, json, numpy as np, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from common import load, split_indices, plot_confusion, TERRAINS, CLASSES
from train_cnn import prepare, predict


class TerrainGRU(nn.Module):
    def __init__(self, n_feats, n_out, in_ch=6, hid=64):
        super().__init__()
        self.gru = nn.GRU(in_ch, hid, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hid + n_feats, 50)
        self.fc2 = nn.Linear(50, n_out)

    def forward(self, x, f=None):
        _, h = self.gru(x.squeeze(2).permute(0, 2, 1))   # (N,6,1,69) -> (N,69,6)
        z = self.drop(h[-1])
        if f is not None and f.shape[1] > 0:
            z = torch.cat([z, f], dim=1)
        return self.fc2(torch.relu(self.fc1(z)))


def train(variant, label, epochs=60, lr=1e-3, batch=16, seed=0, data_path="data/tactile.npz", out="outputs"):
    torch.manual_seed(seed); np.random.seed(seed)
    d = load(data_path)
    y = d["y_class"] if label == "class" else d["y_terrain"]
    names = CLASSES if label == "class" else TERRAINS
    tr, va, te = split_indices(d["y_terrain"])
    X, F, stats = prepare(d, variant, tr)
    model = TerrainGRU(0 if F is None else F.shape[1], len(names))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X); Ft = None if F is None else torch.from_numpy(F); yt = torch.from_numpy(y).long()
    best = (-1, None)
    for ep in range(epochs):
        model.train()
        perm = torch.from_numpy(np.random.permutation(tr))
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xt[b], None if Ft is None else Ft[b]), yt[b])
            loss.backward(); opt.step()
        a_va = accuracy_score(y[va], predict(model, X[va], None if F is None else F[va]))
        if a_va > best[0]:
            best = (a_va, copy.deepcopy(model.state_dict()))
        if (ep + 1) % 10 == 0:
            print(f"[GRU {variant}/{label}] epoch {ep+1:3d}  val {a_va:.3f}", flush=True)
    model.load_state_dict(best[1])
    pred = predict(model, X[te], None if F is None else F[te])
    acc = float(accuracy_score(y[te], pred))
    print(f"[GRU {variant}/{label}] TEST accuracy = {acc:.3f}  (best val {best[0]:.3f})")
    plot_confusion(confusion_matrix(y[te], pred), names, f"GRU {variant} ({label}) test acc {acc:.1%}", f"{out}/cm_gru_{variant}_{label}.png")
    return dict(test_acc=acc, best_val=float(best[0]), epochs=epochs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    res = {}
    for variant in ["raw", "all"]:
        for label in ["class", "terrain"]:
            res[f"{variant}_{label}"] = train(variant, label, a.epochs)
    json.dump(res, open("outputs/results_gru.json", "w"), indent=2)
