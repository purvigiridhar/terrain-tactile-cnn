"""1D CNN terrain classifier (Choi & Thomasson): 4 conv layers (-> 1368 learned features) + 2 FC layers, optional hand-designed features."""
import argparse, copy, json, numpy as np, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from common import load, split_indices, plot_confusion, TERRAINS, CLASSES, CONTROL_IDX

VARIANTS = {"raw": None, "control": CONTROL_IDX, "all": list(range(33))}   # which of the 33 features are appended


class TerrainCNN(nn.Module):
    def __init__(self, n_feats, n_out, in_ch=6, ch=24, T=69):
        super().__init__()
        layers, c = [], in_ch
        for _ in range(4):   # conv (1x5, stride 1, pad 1) -> ReLU -> max-pool (1x2, stride 1) -> dropout; width 69 -> 57
            layers += [nn.Conv2d(c, ch, kernel_size=(1, 5), stride=1, padding=(0, 1)), nn.ReLU(),
                       nn.MaxPool2d((1, 2), stride=1), nn.Dropout(0.5)]
            c = ch
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            n_conv = self.conv(torch.zeros(1, in_ch, 1, T)).numel()
        self.n_conv = n_conv
        self.fc1 = nn.Linear(n_conv + n_feats, 50)
        self.fc2 = nn.Linear(50, n_out)

    def forward(self, x, f=None):
        z = self.conv(x).flatten(1)
        if f is not None and f.shape[1] > 0:
            z = torch.cat([z, f], dim=1)     # append hand-designed features to the learned ones
        return self.fc2(torch.relu(self.fc1(z)))


def prepare(d, variant, tr):
    """Standardise raw channels and features using TRAIN statistics only."""
    X = d["X"]
    mu = X[tr].mean(axis=(0, 2), keepdims=True); sd = X[tr].std(axis=(0, 2), keepdims=True) + 1e-6
    Xn = ((X - mu) / sd)[:, :, None, :].astype(np.float32)           # (N, 6, 1, 69)
    idx = VARIANTS[variant]
    if idx is None:
        return Xn, None, dict(mu=mu, sd=sd, fmu=None, fsd=None)
    F = d["feats"][:, idx]
    fmu = F[tr].mean(0, keepdims=True); fsd = F[tr].std(0, keepdims=True) + 1e-6
    return Xn, ((F - fmu) / fsd).astype(np.float32), dict(mu=mu, sd=sd, fmu=fmu, fsd=fsd)


@torch.no_grad()
def predict(model, X, F):
    model.eval()
    out = model(torch.from_numpy(X), None if F is None else torch.from_numpy(F))
    return out.argmax(1).numpy()


def train(variant, label, epochs=150, lr=1e-3, batch=16, seed=0, data_path="data/tactile.npz", out="outputs"):
    torch.manual_seed(seed); np.random.seed(seed)
    d = load(data_path)
    y = d["y_class"] if label == "class" else d["y_terrain"]
    names = CLASSES if label == "class" else TERRAINS
    tr, va, te = split_indices(d["y_terrain"])
    X, F, stats = prepare(d, variant, tr)
    n_feats = 0 if F is None else F.shape[1]
    model = TerrainCNN(n_feats, len(names))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X); Ft = None if F is None else torch.from_numpy(F); yt = torch.from_numpy(y).long()
    hist = dict(train=[], val=[]); best = (-1, None)
    for ep in range(epochs):
        model.train()
        perm = torch.from_numpy(np.random.permutation(tr))
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xt[b], None if Ft is None else Ft[b]), yt[b])
            loss.backward(); opt.step()
        a_tr = accuracy_score(y[tr], predict(model, X[tr], None if F is None else F[tr]))
        a_va = accuracy_score(y[va], predict(model, X[va], None if F is None else F[va]))
        hist["train"].append(a_tr); hist["val"].append(a_va)
        if a_va > best[0]:
            best = (a_va, copy.deepcopy(model.state_dict()))
        if (ep + 1) % 10 == 0:
            print(f"[{variant}/{label}] epoch {ep+1:3d}  train {a_tr:.3f}  val {a_va:.3f}", flush=True)
    model.load_state_dict(best[1])                     # model selection on validation accuracy
    pred = predict(model, X[te], None if F is None else F[te])
    acc = float(accuracy_score(y[te], pred))
    print(f"[{variant}/{label}] TEST accuracy = {acc:.3f}  (best val {best[0]:.3f})")
    tag = f"{variant}_{label}"
    plot_confusion(confusion_matrix(y[te], pred), names, f"CNN {variant} ({label}) test acc {acc:.1%}", f"{out}/cm_cnn_{tag}.png")
    plt.figure(figsize=(6, 4)); plt.plot(np.array(hist["train"]) * 100, label="Train"); plt.plot(np.array(hist["val"]) * 100, label="Validation")
    plt.xlabel("epoch"); plt.ylabel("accuracy [%]"); plt.title(f"Training curves: {variant} ({label})"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{out}/curves_{tag}.png", dpi=150); plt.close()
    torch.save(dict(state=best[1], variant=variant, label=label, n_feats=n_feats, stats=stats), f"{out}/models/{tag}.pt")
    json.dump(dict(test_acc=acc, best_val=float(best[0]), epochs=epochs, lr=lr, batch=batch, n_learned_features=model.n_conv),
              open(f"{out}/results_cnn_{tag}.json", "w"), indent=2)
    return acc, float(best[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="all")
    ap.add_argument("--label", choices=["class", "terrain"], default="class")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    train(a.variant, a.label, a.epochs, a.lr, a.batch)
