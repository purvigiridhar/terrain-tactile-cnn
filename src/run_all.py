"""Run SVM baseline + all 3 CNN variants x 2 label sets and write the summary table (Table 2 of the paper)."""
import argparse
import svm_baseline, train_cnn

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()

svm = svm_baseline.run()
rows = [("SVM baseline (33 features)", svm["class"]["test_acc"], svm["terrain"]["test_acc"], svm["class"]["best_val"], svm["terrain"]["best_val"])]
for variant, name in [("raw", "CNN: raw data"), ("all", "CNN: raw + all 33 features"), ("control", "CNN: raw + control-input features")]:
    c, cv = train_cnn.train(variant, "class", a.epochs, a.lr, a.batch)
    t, tv = train_cnn.train(variant, "terrain", a.epochs, a.lr, a.batch)
    rows.append((name, c, t, cv, tv))
table = ("| Model | Class acc (test) | Terrain acc (test) | Class acc (val) | Terrain acc (val) |\n|---|---|---|---|---|\n"
         + "\n".join(f"| {n} | {c:.1%} | {t:.1%} | {cv:.1%} | {tv:.1%} |" for n, c, t, cv, tv in rows))
open("outputs/results_table.md", "w").write(table + "\n")
print("\n" + table)
