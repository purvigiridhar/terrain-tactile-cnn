| Model | Class acc (test) | Terrain acc (test) |
|---|---|---|
| SVM, PUK kernel (33 features) | 91.6% | 76.6% |
| Random Forest (33 features) | 93.5% | 81.4% |
| XGBoost (33 features) | 95.0% | 85.2% |
| CNN: raw data | 87.5% | 64.3% |
| CNN: raw + control features | 94.2% | 73.1% |
| CNN: raw + all 33 features | 93.9% | 79.7% |
| GRU: raw data (150 epochs) | 86.8% | 66.1% |
| GRU: raw + all features (150 epochs) | 93.4% | 80.0% |