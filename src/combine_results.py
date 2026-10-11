"""Combine all results (SVM, RF, XGBoost, CNN, GRU) into one table: outputs/final_results.md"""
import json

def j(p):
    return json.load(open("outputs/" + p))

svm, ex, gru = j("results_svm.json"), j("results_extra.json"), j("results_gru.json")
cnn = lambda v, l: j(f"results_cnn_{v}_{l}.json")["test_acc"]

rows = [
    ("SVM, PUK kernel (33 features)", svm["class"]["test_acc"], svm["terrain"]["test_acc"]),
    ("Random Forest (33 features)", ex["rf_class"]["test_acc"], ex["rf_terrain"]["test_acc"]),
    ("XGBoost (33 features)", ex["xgb_class"]["test_acc"], ex["xgb_terrain"]["test_acc"]),
    ("CNN: raw data", cnn("raw", "class"), cnn("raw", "terrain")),
    ("CNN: raw + control features", cnn("control", "class"), cnn("control", "terrain")),
    ("CNN: raw + all 33 features", cnn("all", "class"), cnn("all", "terrain")),
    ("GRU: raw data (20 epochs)", gru["raw_class"]["test_acc"], gru["raw_terrain"]["test_acc"]),
    ("GRU: raw + all features (20 epochs)", gru["all_class"]["test_acc"], gru["all_terrain"]["test_acc"]),
]
lines = ["| Model | Class acc (test) | Terrain acc (test) |", "|---|---|---|"]
lines += [f"| {n} | {c:.1%} | {t:.1%} |" for n, c, t in rows]
out = "\n".join(lines)
open("outputs/final_results.md", "w").write(out)
print(out)
