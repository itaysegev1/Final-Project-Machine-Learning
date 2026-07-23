# Epicurious Recipe Hit/Miss Classifier

Binary classification on the **Epicurious — Recipes with Rating and Nutrition**
dataset. Each recipe has a nutrition profile, ~674 binary editorial tags, and
free-text directions/ingredients. The task is to predict whether a recipe is a
**Hit** (rating ≥ 4.0) or a **Miss** (rating < 4.0).

Seven classifiers are trained and compared on two feature sets — a **Baseline**
(nutrition + tags) and an **Advanced** set that adds engineered culinary text
features — to see whether the engineered features help.


## Project layout

```
.
├── data/                          input files (unchanged)
│   ├── epi_r.csv                  nutrition + rating + binary tag matrix
│   └── full_format_recipes.json   raw directions / ingredients text
│
├── src/                           shared package; everything imports from here
│   ├── _constants.py              RANDOM_STATE = 42
│   ├── data_foundation.py         load, merge, target, train/test split, feature extractor
│   ├── preprocessing.py           imputation, scaling, Baseline + Advanced matrices
│   └── train_utils.py             evaluation / metrics-JSON / I/O / plotting helpers
│
├── train_perceptron.py            one training script per model; each trains on
├── train_logistic_regression.py   Baseline + Advanced and writes to results/<slug>/
├── train_adaboost.py
├── train_pca_knn.py
├── train_pca_knn_improved.py
├── train_random_forest.py
├── train_mlp.py                   also supports --no-early-stopping (overfit baseline)
├── train_Kmeans.py                Added After coversation with the superviser
├── train_rf_with_Kmeans.py        Added After coversation with the superviser
│
├── evaluate_all_results.py        reads every metrics.json, prints the summary table
├── analysis.py                    Logistic Regression coefficients + confidence buckets
├── advanced_tuning.py             decision-threshold selection + top-20 features
│
├── notebooks/                     one thin notebook per model; each runs the
│   ├── 01_Logistic_Regression.ipynb   matching script and shows the plots
│   ├── 02_Random_Forest.ipynb
│   ├── 03_MLP_Neural_Network.ipynb
│   ├── 04_Perceptron.ipynb
│   ├── 05_AdaBoost.ipynb
│   ├── 06_PCA_KNN.ipynb
│   ├── 07_PCA_KNN_Improved.ipynb
│   └── 08_Master_Comparison.ipynb     runs evaluate_all_results.py + plot gallery
│
├── results/                       per-model outputs (metrics.json, predictions, plots)
├── legacy/                        dataset author's original scraper (unused, kept for reference)
├── requirements.txt
└── README.md
```

Everything runs from the command line; the notebooks are just a thin viewer on
top of the same scripts.


## Setup

Requires Python 3.9–3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dependency versions are pinned because scikit-learn estimator behaviour
changes between releases; the exact numbers reproduce only with these versions.


## How to run

Run all commands from the project root, with the virtual environment activated.

### 1. Train the models

Each script is independent and can be run in any order. Every script trains on
both the Baseline and Advanced matrices and writes its outputs to
`results/<model>/`.

```bash
python train_perceptron.py
python train_logistic_regression.py
python train_adaboost.py
python train_pca_knn.py
python train_pca_knn_improved.py
python train_random_forest.py
python train_mlp.py
```

The MLP script has an extra mode that trains the same architecture **without**
early stopping, as an overfitting baseline (writes to `results/mlp_overfit/`):

```bash
python train_mlp.py --no-early-stopping
```

Run it once if you want the early-stopping ablation; `train_mlp.py` then prints a
before/after comparison automatically.

### 2. Compare the results

After training, print the side-by-side summary table across all models:

```bash
python evaluate_all_results.py
```

It reports whatever models it finds, so you can run it after training only some
of them.

### 3. Post-hoc analysis (optional)

```bash
python analysis.py          # LR signed coefficients + prediction confidence buckets
python advanced_tuning.py   # decision-threshold selection + top-20 features
```

### 4. Sanity-check the data pipeline (optional)

The `src/` modules can be run directly to print a summary of each stage:

```bash
python -m src.data_foundation    # load + merge + split summary
python -m src.preprocessing      # Baseline / Advanced matrix shapes + scaler stats
```

### 5. Notebooks

Each notebook runs its training script and displays the resulting plots.

```bash
jupyter lab notebooks/
```

Open `01`–`07` in any order; `08_Master_Comparison.ipynb` trains nothing and
just runs the aggregator over `results/`.


## Outputs

Each model writes to its own folder under `results/<model>/`:

| File | Description |
|---|---|
| `metrics.json` | accuracy, F1, confusion matrix, FP/FN rates, config, extras |
| `predictions_baseline.npy` | int8 test predictions on the Baseline matrix |
| `predictions_advanced.npy` | int8 test predictions on the Advanced matrix |
| `test_index.npy` | the test split's row index |
| `confusion_matrix.png` | confusion-matrix heatmap (Advanced fit) |
| `roc_curve.png` | ROC curve + AUC (Advanced fit) |
| `feature_importance.png` | top-20 importances (Random Forest only) |
| `loss_curve.png` | training loss + validation error (MLP only) |


## Author

Itay Segev — final project for a Machine Learning course.
