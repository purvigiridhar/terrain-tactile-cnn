"""SVM baseline on the 33 hand-designed features with the Pearson VII universal kernel (PUK), as in the paper."""
import json, numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.spatial.distance import cdist
from common import load, split_indices, plot_confusion, TERRAINS, CLASSES


def puk(A, B, sigma, omega=1.0):
    d = cdist(A, B)
    return 1.0 / (1.0 + (2 * d * np.sqrt(2 ** (1 / omega) - 1) / sigma) ** 2) ** omega


def run(data_path="data/tactile.npz", out="outputs"):
    d = load(data_path)
    tr, va, te = split_indices(d["y_terrain"])
    sc = StandardScaler().fit(d["feats"][tr])
    F = sc.transform(d["feats"])
    res = {}
    for label, y, names in [("class", d["y_class"], CLASSES), ("terrain", d["y_terrain"], TERRAINS)]:
        best = (-1, None)                      # choose C, sigma on the validation split
        for sigma in [3, 10, 30, 100]:
            Ktr = puk(F[tr], F[tr], sigma)
            for C in [1, 10, 100]:
                m = SVC(kernel="precomputed", C=C).fit(Ktr, y[tr])
                a = accuracy_score(y[va], m.predict(puk(F[va], F[tr], sigma)))
                if a > best[0]:
                    best = (a, (sigma, C, m))
        sigma, C, m = best[1]
        pred = m.predict(puk(F[te], F[tr], sigma))
        acc = accuracy_score(y[te], pred)
        res[label] = dict(test_acc=float(acc), best_params=dict(sigma=sigma, C=C), best_val=float(best[0]))
        plot_confusion(confusion_matrix(y[te], pred), names, f"SVM ({label}) test acc {acc:.1%}", f"{out}/cm_svm_{label}.png")
        print(f"SVM  {label:8s} test acc = {acc:.3f}  sigma={sigma} C={C}")
    json.dump(res, open(f"{out}/results_svm.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    run()
