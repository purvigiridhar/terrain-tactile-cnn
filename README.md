# Terrain Classification for Small Legged Robots — 1D CNN on Tactile Data

UE24CS352A Machine Learning mini-project (Problem Statement 2). Re-implementation of
*Choi & Thomasson, "Terrain Classification for Small Legged Robots Using Deep Learning on Tactile Data"* (Stanford CS229, 2020).

**Task:** classify the terrain a small legged robot (SAIL-R) is walking on from one step of capacitive tactile data
(6 taxel channels x 69 time samples): 4 higher-level classes (HF, LF, D, G) and 8 individual terrains.

## Dataset
Real SAIL-R data released by the paper's authors: <https://github.com/rthomasson23/CS229_TerrainClassification>
(`dataSet_withFeatures.csv`, `dataSet_withFeatures_labels.csv`, copied into `data/`).
5,990 steps; each = 6 taxel channels (5 normal + 1 shear) x 69 samples + 33 hand-designed features (Table 1).
8 terrains: asphalt, cork, grass, gravel, lab, laminate_wood, pebble, sand.
Classes: HF = asphalt; LF = lab, laminate_wood; D = cork, grass, gravel, pebble; G = sand (mapping taken from the authors' `ProcessData.py`).
The dataset's terrain names differ from those printed in the paper (e.g. paper "concrete" = asphalt here); we use the dataset's names.

## Setup
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run (all commands from the repo root)
```bash
python src/prepare_data.py                 # CSVs -> data/tactile.npz
python src/run_all.py                      # SVM + all CNN variants (150 epochs, batch 16, lr 1e-3); writes outputs/results_table.md
```
Single model: `python src/train_cnn.py --variant all --label terrain --epochs 150`
(`--variant raw | control | all`, `--label class | terrain`). SVM only: `python src/svm_baseline.py`.
Live demo (after training): `python src/demo.py --n 3`

## Layout
| Path | Purpose |
|---|---|
| `src/prepare_data.py` | converts the authors' CSVs to `data/tactile.npz` |
| `src/svm_baseline.py` | SVM on the 33 features (PUK kernel, C and sigma chosen on the validation split) |
| `src/train_cnn.py` | 1D CNN (4 conv layers, 24 channels, kernel 1x5, max-pool, dropout 0.5, 2 FC layers, Adam 1e-4, cross-entropy) |
| `src/run_all.py` | runs everything, writes results table |
| `src/demo.py` | shows predictions on held-out steps |
| `outputs/` | confusion matrices, training curves, results JSON, saved models |

## Notes / differences from the paper
- Split follows the authors' `ProcessData.py`: 112 validation + 112 test samples per terrain (896 each), the rest (4,190) for training. The exact samples are random (seed 0).
- CNN matches the authors' notebook: four conv layers (24 channels, kernel 1x5, padding 1), each followed by max-pool (1x2, stride 1) and dropout 0.5, giving 24 x 57 = 1368 learned features; FC 50 -> output; cross-entropy; Adam.
- Training: the authors' code uses batch size 1 (their default) with lr 1e-4 for 100+ epochs, i.e. ~4,000 updates per epoch. We use batch 16, lr 1e-3, 150 epochs for speed. To replicate their setting: `python src/train_cnn.py --batch 1 --lr 1e-4 --epochs 150` (slow).
- Model selection uses best validation accuracy; we report **test** accuracy, which was evaluated once on the best checkpoint. The authors' published confusion matrices appear to come from the validation set, so their numbers are slightly optimistic compared with ours.
- SVM uses the PUK kernel with omega = 1; C and sigma are chosen on the validation split.
