"""Convert the authors' CSVs (Choi & Thomasson, github.com/rthomasson23/CS229_TerrainClassification)
into data/tactile.npz.

dataSet_withFeatures.csv : 5990 rows x 448 cols = 414 raw values (6 channels x 69 samples, channel-major)
                           + 33 hand-designed features (Table 1) + 1 trailing label column
dataSet_withFeatures_labels.csv : terrain name per row
"""
import numpy as np, pandas as pd
from common import TERRAINS, TERRAIN_TO_CLASS

a = pd.read_csv("data/dataSet_withFeatures.csv", header=None)
lab = pd.read_csv("data/dataSet_withFeatures_labels.csv", header=None)[0].values
A = a.iloc[:, :447].values.astype(np.float32)
assert (a.iloc[:, 447].values == lab).all(), "label column mismatch"
y_terrain = np.array([TERRAINS.index(l) for l in lab])
np.savez("data/tactile.npz",
         X=A[:, :414].reshape(-1, 6, 69), feats=A[:, 414:447],
         y_terrain=y_terrain, y_class=np.array(TERRAIN_TO_CLASS)[y_terrain])
print("saved data/tactile.npz:", A.shape[0], "steps;", np.bincount(y_terrain), "per terrain")
