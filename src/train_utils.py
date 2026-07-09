"""
Shared helpers for all the train scripts and notebooks: loading the data,
scoring a model, saving metrics/predictions/plots to results/<slug>/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src._constants import RANDOM_STATE
from src.preprocessing import build_preprocessed_datasets


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# the order of the two feature matrices, kept stable for the json and npy names
DATASETS = ("Baseline", "Advanced")


def model_results_dir(model_slug):
    """
    returns results/<slug>/, creates it if needed
    """
    if not model_slug or not isinstance(model_slug, str):
        raise ValueError(f"Invalid model_slug: {model_slug!r}")
    path = RESULTS_DIR / model_slug
    path.mkdir(parents=True, exist_ok=True)
    return path


# Data loading
def load_preprocessed():
    """
    loads both matrices as a dict with "Baseline"/"Advanced" keys,
    returns (datasets, y_train, y_test)
    """
    (
        X_train_b, X_test_b,
        X_train_a, X_test_a,
        y_train, y_test,
    ) = build_preprocessed_datasets(verbose=False)

    datasets = {
        "Baseline": (X_train_b, X_test_b),
        "Advanced": (X_train_a, X_test_a),
    }
    return datasets, y_train, y_test


# Evaluation
def _confusion_to_dict(cm):
    """
    turns the 2x2 confusion matrix into a tn/fp/fn/tp dict
    """
    return {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def _compute_rates(cm_dict):
    """
    returns the (FP rate, FN rate) pair from the confusion dict
    """
    tn, fp, fn, tp = cm_dict["tn"], cm_dict["fp"], cm_dict["fn"], cm_dict["tp"]
    fp_rate = fp / (tn + fp) if (tn + fp) else 0.0   # P(pred=Hit | true=Miss)
    fn_rate = fn / (fn + tp) if (fn + tp) else 0.0   # P(pred=Miss | true=Hit)
    return fp_rate, fn_rate


def fit_and_score(model, X_train, y_train, X_test, y_test):
    """
    fits the model on train, predicts on test and packs everything to one dict.
    models with no predict_proba (like Perceptron) fall back to decision_function.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    proba_hit = None
    if hasattr(model, "predict_proba"):
        proba_hit = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        proba_hit = model.decision_function(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_dict = _confusion_to_dict(cm)
    fp_rate, fn_rate = _compute_rates(cm_dict)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": cm_dict,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "y_pred": y_pred,
        "proba_hit": proba_hit,
        "model": model,
    }


# Payload + persistence
def build_metrics_payload(
    model_name,
    display_name,
    model_config,
    n_train,
    n_test,
    random_state,
    per_dataset_results,
    extras=None,
):
    """
    packs the per dataset results into the json payload the aggregator reads
    """
    datasets_block = {}
    for ds in DATASETS:
        if ds not in per_dataset_results:
            continue
        r = per_dataset_results[ds]
        datasets_block[ds] = {
            "accuracy": r["accuracy"],
            "f1": r["f1"],
            "confusion_matrix": r["confusion_matrix"],
            "fp_rate": r["fp_rate"],
            "fn_rate": r["fn_rate"],
        }

    return {
        "model_name": model_name,
        "display_name": display_name,
        "model_config": model_config,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "random_state": int(random_state),
        "datasets": datasets_block,
        "extras": extras or {},
    }


def save_metrics(model_slug, payload):
    """
    writes the payload to results/<slug>/metrics.json
    """
    path = model_results_dir(model_slug) / "metrics.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=_json_default)
        fh.write("\n")
    return path


def save_predictions(model_slug, dataset, y_pred):
    """
    saves the test predictions as a small int8 npy file
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}.")
    path = model_results_dir(model_slug) / f"predictions_{dataset.lower()}.npy"
    np.save(path, np.asarray(y_pred, dtype=np.int8))
    return path


def save_test_index(model_slug, y_test):
    """
    saves the test rows index next to the predictions
    """
    path = model_results_dir(model_slug) / "test_index.npy"
    np.save(path, np.asarray(y_test.index, dtype=np.int64))
    return path


def save_figure(model_slug, filename, fig):
    """
    saves a figure into results/<slug>/<filename> at 200 dpi
    """
    path = model_results_dir(model_slug) / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def load_metrics(model_slug):
    """
    reads results/<slug>/metrics.json back to a dict
    """
    path = RESULTS_DIR / model_slug / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"No metrics for slug {model_slug!r} at {path}.")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj):
    """
    helper for json.dump, casts numpy types to native python types
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


# Plot helpers, they return the Figure and the caller shows or saves it
def confusion_matrix_figure(cm, title="Confusion Matrix", figsize=(5.5, 4.5)):
    """
    draws the confusion matrix heatmap and returns the Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Miss (0)", "Hit (1)"],
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    return fig


def roc_curve_figure(y_true, y_score, title="ROC Curve", figsize=(5.5, 4.5),
                     model_label="Model"):
    """
    draws the ROC curve and returns (Figure, auc)
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, lw=2.0, label=f"{model_label} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.0,
            label="Random (AUC = 0.50)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, auc


# Print helpers used by all the train scripts
def print_dataset_block(ds_name, X_shape, result):
    """
    prints the per dataset block that every train script shows
    """
    cm = result["confusion_matrix"]
    print(f"\n  --- {ds_name}  (X_train: {X_shape}) ---")
    print(f"     Test Accuracy : {result['accuracy']:.4f}")
    print(f"     Test F1-Score : {result['f1']:.4f}")
    print(f"     Confusion Matrix:")
    print(f"                       Pred:Miss  Pred:Hit")
    print(f"        True:Miss    {cm['tn']:>9}  {cm['fp']:>8}   "
          f"FP rate = {result['fp_rate']:.4f}")
    print(f"        True:Hit     {cm['fn']:>9}  {cm['tp']:>8}   "
          f"FN rate = {result['fn_rate']:.4f}")


def print_delta(per_dataset_results):
    """
    prints the difference between Advanced and Baseline (acc and F1)
    """
    if "Baseline" not in per_dataset_results or "Advanced" not in per_dataset_results:
        return
    b = per_dataset_results["Baseline"]
    a = per_dataset_results["Advanced"]
    print(
        f"\n  >> Δ (Advanced − Baseline):  "
        f"Acc {a['accuracy'] - b['accuracy']:+.4f}  |  "
        f"F1 {a['f1'] - b['f1']:+.4f}"
    )
