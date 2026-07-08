"""
This script does the threshold tuning and the top feature extraction for the
Logistic Regression. the threshold is selected on a validation split held out
of train (not on the test set), and then we report it once on the test set.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.data_foundation import CulinaryFeatureExtractor
from src.preprocessing import build_preprocessed_datasets
# we import the LR config instead of copying it so the numbers stay the same
from train_logistic_regression import MODEL_CONFIG as LR_MODEL_CONFIG


from src._constants import RANDOM_STATE
THRESHOLDS = np.linspace(0.30, 0.70, 9)   # 0.30, 0.35, ..., 0.70
TOP_K = 20
DEFAULT_THRESHOLD = 0.50                  # baseline we compare against
VALIDATION_FRACTION = 0.15                # held out of TRAIN for threshold selection

CULINARY_FEATURE_NAMES = (
    tuple(CulinaryFeatureExtractor.BASELINE_FEATURES)
    + tuple(f"has_{g}" for g in CulinaryFeatureExtractor.KEYWORD_GROUPS)
)


# Threshold evaluation
def _evaluate_at_threshold(y_true, proba_hit, threshold):
    """
    applies one decision threshold and returns all the metrics we report
    """
    y_pred = (proba_hit >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negatives, false_positives = int(cm[0, 0]), int(cm[0, 1])
    false_negatives, true_positives = int(cm[1, 0]), int(cm[1, 1])

    n_true_miss = true_negatives + false_positives
    n_true_hit = false_negatives + true_positives
    fp_rate = false_positives / n_true_miss if n_true_miss else 0.0   # P(pred=Hit | true=Miss)
    fn_rate = false_negatives / n_true_hit if n_true_hit else 0.0     # P(pred=Miss | true=Hit)
    balance = (fp_rate / fn_rate) if fn_rate > 0 else float("inf")

    return {
        "threshold": float(threshold),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "balance_ratio": balance,
        "pred_hits": int(y_pred.sum()),
        "n_total": len(y_pred),
    }


def _print_threshold_table(rows):
    """
    prints one aligned row per threshold
    """
    header = (
        f"  {'thresh':>6}  {'F1':>7}  {'Acc':>7}  "
        f"{'FP rate':>8}  {'FN rate':>8}  {'Bal ratio':>10}  "
        f"{'Predicted Hits':>16}"
    )
    print(header)
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*16}")
    for r in rows:
        bal = r["balance_ratio"]
        bal_str = f"{bal:10.3f}" if np.isfinite(bal) else f"{'inf':>10}"
        hits_pct = 100.0 * r["pred_hits"] / r["n_total"]
        hits_str = f"{r['pred_hits']:>5d} ({hits_pct:5.1f}%)"
        print(
            f"  {r['threshold']:>6.2f}  {r['f1']:>7.4f}  {r['accuracy']:>7.4f}  "
            f"{r['fp_rate']:>8.4f}  {r['fn_rate']:>8.4f}  {bal_str}  "
            f"{hits_str:>16}"
        )


def _log_distance_to_balanced(row):
    """
    returns |log(balance_ratio)| so 2.0 and 0.5 count as equally off
    """
    bal = row["balance_ratio"]
    if not np.isfinite(bal) or bal <= 0:
        return float("inf")
    return abs(np.log(bal))


def _summarize_row(label, r):
    """
    prints one highlighted threshold row
    """
    bal = r["balance_ratio"]
    bal_str = f"{bal:.3f}" if np.isfinite(bal) else "inf"
    print(
        f"  {label:<34} threshold={r['threshold']:.2f}   "
        f"F1={r['f1']:.4f}   Acc={r['accuracy']:.4f}   "
        f"FP/FN rates={r['fp_rate']:.4f}/{r['fn_rate']:.4f}   "
        f"balance={bal_str}"
    )


def _select_thresholds(rows):
    """
    picks the best F1 row and the most balanced row from the validation sweep
    """
    return {
        "best_f1": max(rows, key=lambda r: r["f1"]),
        "most_balanced": min(rows, key=_log_distance_to_balanced),
    }


# Top-coefficient extraction
def _print_top_coefficients(coefs, k):
    """
    prints the top k positive and negative coefficients with their ranks,
    and points out any culinary feature that made the lists
    """
    top_pos = coefs.sort_values(ascending=False).head(k)
    top_neg = coefs.sort_values(ascending=True).head(k)

    def _print_block(title, series):
        print(f"\n  {title}")
        print(f"  {'rank':>4}  {'coef':>10}   {'feature'}")
        print(f"  {'-'*4}  {'-'*10}   {'-'*55}")
        for rank, (name, val) in enumerate(series.items(), start=1):
            print(f"  {rank:>4}  {val:+10.4f}   {name}")

    _print_block(
        f"Top {k} POSITIVE — 'Recipe for Success' (strongest Hit indicators)",
        top_pos,
    )
    _print_block(
        f"Top {k} NEGATIVE — 'Recipe for Disaster' (strongest Miss indicators)",
        top_neg,
    )

    eng_in_pos = [n for n in top_pos.index if n in CULINARY_FEATURE_NAMES]
    eng_in_neg = [n for n in top_neg.index if n in CULINARY_FEATURE_NAMES]
    print()
    if eng_in_pos or eng_in_neg:
        print(f"  >> Engineered culinary features in the top {k}:")
        for n in eng_in_pos:
            print(f"     POSITIVE: {n} ({top_pos[n]:+.4f})")
        for n in eng_in_neg:
            print(f"     NEGATIVE: {n} ({top_neg[n]:+.4f})")
    else:
        print(
            f"  >> No engineered culinary features made the top {k} lists in "
            "either direction. The CSV's binary tag matrix dominates the "
            "interpretable signal."
        )


def main():
    print("=" * 72)
    print("  ADVANCED TUNING & TOP FEATURE EXTRACTION")
    print("=" * 72)

    print("\nLoading preprocessed Advanced datasets...")
    (
        _, _,
        X_train_adv, X_test_adv,
        y_train, y_test,
    ) = build_preprocessed_datasets(verbose=False)
    print(f"  X_train shape: {X_train_adv.shape}")
    print(f"  X_test  shape: {X_test_adv.shape}")

    # 1. Threshold tuning — selection on validation, evaluation on test
    print("\n" + "=" * 72)
    print("  1) THRESHOLD TUNING — addressing FP/FN asymmetry")
    print("=" * 72)
    print(
        "  Balance Ratio = FP_rate / FN_rate.\n"
        "    >1 → model favours 'Hit' (more false alarms than misses)\n"
        "    <1 → model favours 'Miss'\n"
        "    ≈1 → symmetric errors\n"
    )

    # 1a. selection sweep on a validation split held out of train
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_adv, y_train,
        test_size=VALIDATION_FRACTION,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    print(f"  1a) SELECTION sweep on a {VALIDATION_FRACTION:.0%} validation split "
          f"held out of train (n={len(y_val)}):\n")
    lr_select = LogisticRegression(**LR_MODEL_CONFIG)
    lr_select.fit(X_tr, y_tr)
    val_proba = lr_select.predict_proba(X_val)[:, 1]
    val_rows = [_evaluate_at_threshold(y_val, val_proba, t) for t in THRESHOLDS]
    _print_threshold_table(val_rows)

    selected = _select_thresholds(val_rows)
    print("\n  Selected on validation:")
    _summarize_row("Best F1 (val):", selected["best_f1"])
    _summarize_row("Most balanced (val, |log| min):", selected["most_balanced"])

    # 1b. final model on the full train split, the test sweep is just descriptive
    print("\nTraining the final LogisticRegression on the full train split "
          "(config imported from train_logistic_regression)...")
    lr = LogisticRegression(**LR_MODEL_CONFIG)
    lr.fit(X_train_adv, y_train)
    proba_hit = lr.predict_proba(X_test_adv)[:, 1]

    print(f"\n  1b) DESCRIPTIVE sweep on the test set (n={len(y_test)}) — "
          "not used for selection:\n")
    test_rows = [_evaluate_at_threshold(y_test, proba_hit, t) for t in THRESHOLDS]
    _print_threshold_table(test_rows)

    # 1c. the validation selected thresholds, evaluated once on the test set
    print("\n  1c) Validation-selected operating points, evaluated on TEST:")
    default_test = _evaluate_at_threshold(y_test, proba_hit, DEFAULT_THRESHOLD)
    _summarize_row("Default (t=0.50):", default_test)
    for label, sel_row in (
        ("Best F1 (selected on val):", selected["best_f1"]),
        ("Most balanced (selected on val):", selected["most_balanced"]),
    ):
        test_eval = _evaluate_at_threshold(y_test, proba_hit, sel_row["threshold"])
        _summarize_row(label, test_eval)

    # 2. Top coefficients
    print("\n" + "=" * 72)
    print(f"  2) FEATURE RANKING — top {TOP_K} per direction (n_features={X_train_adv.shape[1]})")
    print("=" * 72)
    print("  Coefficients are signed log-odds shifts per +1 IQR (or per +1 for "
          "0/1 binary features) of the feature.")
    coefs = pd.Series(lr.coef_[0], index=X_train_adv.columns)
    _print_top_coefficients(coefs, k=TOP_K)


if __name__ == "__main__":
    main()
