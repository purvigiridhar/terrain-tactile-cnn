"""Extra baselines: Random Forest and XGBoost on the 33 hand-designed features (same split as the SVM)."""
import json, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from common import load, split_indices, plot_confusion, TERRAINS, CLASSES


def run(data_path="data/tactile.npz", out="outputs"):
    d = load(data_path)
    tr, va, te = split_indices(d["y_terrain"])
    F = d["feats"]
    models = {"rf": lambda: RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)}
    try:
        from xgboost import XGBClassifier
        models["xgb"] = lambda: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                              random_state=0, n_jobs=-1)
    except ImportError:
        print("xgboost not installed, running Random Forest only")

    res = {}
    for name, make in models.items():
        for label, y, names in [("class", d["y_class"], CLASSES), ("terrain", d["y_terrain"], TERRAINS)]:
            m = make().fit(F[tr], y[tr])
            val = accuracy_score(y[va], m.predict(F[va]))
            pred = m.predict(F[te])
            acc = accuracy_score(y[te], pred)
            res[f"{name}_{label}"] = dict(test_acc=float(acc), val_acc=float(val))
            plot_confusion(confusion_matrix(y[te], pred), names,
                           f"{name.upper()} ({label}) test acc {acc:.1%}", f"{out}/cm_{name}_{label}.png")
            print(f"{name.upper():4s} {label:8s} test acc = {acc:.3f}  (val {val:.3f})")
    json.dump(res, open(f"{out}/results_extra.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    run()

