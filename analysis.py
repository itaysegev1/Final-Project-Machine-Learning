"""
This script is the interpretation analysis. we train one Logistic Regression on
the Advanced dataset and look at the top coefficients, the confidence buckets on
the test set, and save the confusion matrix + ROC plots.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.data_foundation import (
    CulinaryFeatureExtractor,
    clean_and_binarize,
    load_binary_matrix,
    load_recipe_text,
    merge_datasets,
)
from src.preprocessing import build_preprocessed_datasets
from src.train_utils import (
    confusion_matrix_figure,
    roc_curve_figure,
    save_figure,
)
# we import the LR config instead of copying it so the numbers stay the same
from train_logistic_regression import MODEL_CONFIG as LR_MODEL_CONFIG

TOP_K = 15           # top coefficients per direction
TOP_CONFIDENCE = 5   # rows printed per confidence bucket
TITLE_WIDTH = 70     # truncate titles so the printing stays tidy

LR_SLUG = "logistic_regression"

# the 9 culinary feature names, taken from the extractor so they cant drift
CULINARY_FEATURE_NAMES = (
    tuple(CulinaryFeatureExtractor.BASELINE_FEATURES)
    + tuple(f"has_{g}" for g in CulinaryFeatureExtractor.KEYWORD_GROUPS)
)


# Title recovery
def _recover_titles():
    """
    rebuilds the title series aligned to the same index as the split
    """
    bin_df = load_binary_matrix()
    txt_df = load_recipe_text()
    merged = merge_datasets(bin_df, txt_df, verbose=False)
    merged, _ = clean_and_binarize(merged, verbose=False)
    return merged["title"].astype(str)


# Printing helpers
def _truncate(s, width=TITLE_WIDTH):
    """
    truncates a title so the printing stays aligned
    """
    s = " ".join(s.split())
    return s if len(s) <= width else s[: width - 1] + "…"


def _print_top_coefficients(coefs, k):
    """
    prints the top k positive and negative coefficients and where our
    culinary features ended up
    """
    top_pos = coefs.sort_values(ascending=False).head(k)
    top_neg = coefs.sort_values(ascending=True).head(k)

    print(f"\n  Top {k} POSITIVE coefficients (strongest Hit indicators)")
    print(f"  {'coef':>10}   {'feature'}")
    print(f"  {'-' * 10}   {'-' * 50}")
    for name, val in top_pos.items():
        print(f"  {val:+10.4f}   {name}")

    print(f"\n  Top {k} NEGATIVE coefficients (strongest Miss indicators)")
    print(f"  {'coef':>10}   {'feature'}")
    print(f"  {'-' * 10}   {'-' * 50}")
    for name, val in top_neg.items():
        print(f"  {val:+10.4f}   {name}")

    # where did the 9 culinary features land
    eng_set = set(CULINARY_FEATURE_NAMES)
    eng_coefs = coefs[coefs.index.isin(eng_set)].sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    print("\n  >> ALL 9 engineered culinary features, ranked by |coef|:")
    n_features = len(coefs)
    abs_rank = coefs.abs().rank(ascending=False, method="min").astype(int)
    for name, val in eng_coefs.items():
        rank = int(abs_rank[name])
        direction = "Hit" if val > 0 else "Miss"
        print(
            f"     {val:+.4f}   rank {rank:>4}/{n_features}"
            f"   pushes toward {direction:<4}   {name}"
        )

    in_top_pos = [n for n in top_pos.index if n in eng_set]
    in_top_neg = [n for n in top_neg.index if n in eng_set]
    if in_top_pos or in_top_neg:
        print("\n  >> Culinary features that cracked the top-15 lists:")
        for n in in_top_pos:
            print(f"     POSITIVE: {n} ({top_pos[n]:+.4f})")
        for n in in_top_neg:
            print(f"     NEGATIVE: {n} ({top_neg[n]:+.4f})")
    else:
        print(
            "\n  >> None of the engineered culinary features cracked the "
            "top-15 lists — the binary tag matrix dominates."
        )


def _print_confidence_groups(proba_hit, y_true, titles, k):
    """
    prints the most confident Hits, most confident Misses and the borderline rows
    """
    df = pd.DataFrame({
        "title": titles.values,
        "true": y_true.values,
        "p_hit": proba_hit,
    })
    df["pred"] = (df["p_hit"] >= 0.5).astype(int)
    df["dist_to_05"] = (df["p_hit"] - 0.5).abs()
    df["correct"] = (df["true"] == df["pred"])

    def _emit(label, rows):
        print(f"\n  {label}")
        print(f"     {'p(Hit)':>7}  {'true':>4}  {'pred':>4}  {'verdict':>7}   title")
        print(f"     {'-'*7}  {'-'*4}  {'-'*4}  {'-'*7}   {'-'*TITLE_WIDTH}")
        for _, r in rows.iterrows():
            verdict = "OK" if r["correct"] else "WRONG"
            true_lbl = "Hit" if r["true"] == 1 else "Miss"
            pred_lbl = "Hit" if r["pred"] == 1 else "Miss"
            print(
                f"     {r['p_hit']:7.4f}  {true_lbl:>4}  {pred_lbl:>4}  "
                f"{verdict:>7}   {_truncate(r['title'])}"
            )

    most_hit = df.sort_values("p_hit", ascending=False).head(k)
    most_miss = df.sort_values("p_hit", ascending=True).head(k)
    borderline = df.sort_values("dist_to_05", ascending=True).head(k)

    _emit(f"Top {k} MOST CONFIDENT HIT predictions  (p closest to 1.0):", most_hit)
    _emit(f"Top {k} MOST CONFIDENT MISS predictions (p closest to 0.0):", most_miss)
    _emit(f"Top {k} MOST BORDERLINE predictions     (p closest to 0.5):", borderline)


def main():
    print("=" * 72)
    print("  INTERPRETATION & CONFIDENCE ANALYSIS")
    print("=" * 72)

    print("Loading preprocessed datasets...")
    (
        _, _,
        X_train_adv, X_test_adv,
        y_train, y_test,
    ) = build_preprocessed_datasets(verbose=False)

    print("Recovering recipe titles aligned to the test split...")
    titles_all = _recover_titles()
    test_titles = titles_all.loc[X_test_adv.index]

    assert len(test_titles) == len(X_test_adv), (
        "Title recovery mis-aligned — indices drifted."
    )

    # we use liblinear because lbfgs didnt converge on this input, the config
    # is imported from the train script so both stay the same
    print(f"\nTraining LogisticRegression on Advanced X_train {X_train_adv.shape}...")
    lr = LogisticRegression(**LR_MODEL_CONFIG)
    lr.fit(X_train_adv, y_train)

    y_pred = lr.predict(X_test_adv)
    proba_hit = lr.predict_proba(X_test_adv)[:, 1]

    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    test_auc = float(roc_auc_score(y_test, proba_hit))

    print(f"  Test Accuracy : {test_accuracy:.4f}")
    print(f"  Test F1-Score : {test_f1:.4f}")
    print(f"  Test ROC AUC  : {test_auc:.4f}")

    # 1. Feature importance
    print("\n" + "=" * 72)
    print("  1) FEATURE IMPORTANCE — signed LR coefficients")
    print("=" * 72)
    print("  Coefficients are signed log-odds shifts per +1 IQR of the feature.")
    coefs = pd.Series(lr.coef_[0], index=X_train_adv.columns)
    _print_top_coefficients(coefs, k=TOP_K)

    # 2. Confidence analysis
    print("\n" + "=" * 72)
    print("  2) CONFIDENCE ANALYSIS — predict_proba on test set")
    print("=" * 72)
    _print_confidence_groups(proba_hit, y_test, test_titles, k=TOP_CONFIDENCE)

    # 3. Plots (we reuse the shared figure helpers so the style stays the same)
    print("\n" + "=" * 72)
    print("  3) PLOTS")
    print("=" * 72)
    fig_cm = confusion_matrix_figure(
        cm, title="Logistic Regression — Confusion Matrix (Advanced)",
        figsize=(6.5, 5.5),
    )
    cm_path = save_figure(LR_SLUG, "confusion_matrix.png", fig_cm)
    plt.close(fig_cm)

    fig_roc, auc_check = roc_curve_figure(
        y_test, proba_hit,
        title="Logistic Regression — ROC Curve (Advanced)",
        figsize=(6.5, 5.5),
        model_label="LogReg",
    )
    roc_path = save_figure(LR_SLUG, "roc_curve.png", fig_roc)
    plt.close(fig_roc)

    assert abs(auc_check - test_auc) < 1e-9, "AUC mismatch between plot and metrics."
    print(f"  Wrote {cm_path}")
    print(f"  Wrote {roc_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
