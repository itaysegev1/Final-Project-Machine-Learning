"""
This script aggregates all the results under results/<slug>/ to one summary
table. it trains nothing, just reads the metrics.json of every model, so run
the train scripts (or the notebooks) first.
"""

import json
import sys

import pandas as pd

from src.train_utils import DATASETS, RESULTS_DIR


METRICS_BASENAME = "metrics.json"

# the order the seven models appear in the table
PREFERRED_ORDER = (
    "Perceptron",
    "LogisticRegression",
    "AdaBoost",
    "PCA(0.90) + KNN",
    "PCA(0.90) + KNN (Improved)",
    "RandomForest",
    "MLP (128,64)",
)


def load_metrics_files(results_dir):
    """
    reads every results/<slug>/metrics.json into a list of dicts
    """
    if not results_dir.exists():
        return []
    payloads = []
    for slug_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        path = slug_dir / METRICS_BASENAME
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            payload.setdefault("_slug", slug_dir.name)
            payloads.append(payload)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [warning] could not read {path}: {exc}", file=sys.stderr)
    return payloads


def order_payloads(payloads):
    """
    orders the payloads by PREFERRED_ORDER, the rest go after alphabetically
    """
    # duplicate model_names keep the last one (with a warning) instead of dropping
    name_to_payload = {}
    for p in payloads:
        name = p.get("model_name", "")
        if name in name_to_payload:
            print(
                f"  [warning] duplicate model_name {name!r} "
                f"(slugs: {name_to_payload[name].get('_slug')!r} and "
                f"{p.get('_slug')!r}) — keeping the latter.",
                file=sys.stderr,
            )
        name_to_payload[name] = p
    ordered = []
    for name in PREFERRED_ORDER:
        if name in name_to_payload:
            ordered.append(name_to_payload.pop(name))
    for name in sorted(name_to_payload):
        ordered.append(name_to_payload[name])
    return ordered


def missing_from_preferred(payloads):
    """
    returns which of the expected models we didnt find
    """
    present = {p.get("model_name") for p in payloads}
    return [name for name in PREFERRED_ORDER if name not in present]


# Summary table
def row_from_payload(payload):
    """
    builds one row of the summary table from a single payload
    """
    ds = payload.get("datasets", {})
    baseline_key, advanced_key = DATASETS
    baseline_metrics = ds.get(baseline_key, {})
    advanced_metrics = ds.get(advanced_key, {})
    return {
        "Model": payload.get("display_name", payload.get("model_name", "?")),
        "Acc (Baseline)": baseline_metrics.get("accuracy", float("nan")),
        "Acc (Advanced)": advanced_metrics.get("accuracy", float("nan")),
        "Δ Acc": _safe_delta(advanced_metrics.get("accuracy"), baseline_metrics.get("accuracy")),
        "F1 (Baseline)": baseline_metrics.get("f1", float("nan")),
        "F1 (Advanced)": advanced_metrics.get("f1", float("nan")),
        "Δ F1": _safe_delta(advanced_metrics.get("f1"), baseline_metrics.get("f1")),
    }


def _safe_delta(a, b):
    """
    returns a - b, or nan if one of them is missing
    """
    if a is None or b is None:
        return float("nan")
    return float(a) - float(b)


def _format_summary_table(rows):
    """
    formats the rows to an aligned text table (the delta columns get a sign)
    """
    df = pd.DataFrame(rows)
    formatters = {
        col: (lambda x: f"{x:+.4f}") if col.startswith("Δ") else (lambda x: f"{x: .4f}")
        for col in df.columns
        if col != "Model"
    }
    return df.to_string(index=False, formatters=formatters)


def build_summary_dataframe(results_dir=RESULTS_DIR):
    """
    returns the summary as a DataFrame (the comparison notebook uses this)
    """
    payloads = order_payloads(load_metrics_files(results_dir))
    return pd.DataFrame([row_from_payload(p) for p in payloads])


def _print_linear_vs_nonlinear_verdict(payloads, meaningful=0.01):
    """
    compares RF and MLP against the LR baseline on the Advanced set
    """
    by_name = {p.get("model_name"): p for p in payloads}
    lr = by_name.get("LogisticRegression")
    rf = by_name.get("RandomForest")
    mlp = by_name.get("MLP (128,64)")
    if lr is None or (rf is None and mlp is None):
        return

    lr_advanced = lr["datasets"]["Advanced"]
    print("\n" + "=" * 72)
    print("  CROSS-MODEL NOTE — Linear baseline vs Non-linear models")
    print("=" * 72)
    print(f"\n  Reference (calibrated linear): \nLR Advanced   "
          f"Acc {lr_advanced['accuracy']:.4f}   F1 {lr_advanced['f1']:.4f}")

    def _line(label, payload):
        ad = payload["datasets"]["Advanced"]
        d_acc = ad["accuracy"] - lr_advanced["accuracy"]
        d_f1 = ad["f1"] - lr_advanced["f1"]
        if d_acc >= meaningful and d_f1 >= meaningful:
            verdict = "BREAKS the linear ceiling on both metrics."
        elif d_acc >= meaningful or d_f1 >= meaningful:
            verdict = "PARTIAL gain (one metric only)."
        elif d_acc <= -meaningful or d_f1 <= -meaningful:
            verdict = "UNDERPERFORMS the linear baseline."
        else:
            verdict = "PLATEAUS within noise of LR — no meaningful gain."
        print(
            f"  {label:<22} Acc {ad['accuracy']:.4f} ({d_acc:+.4f})   "
            f"F1 {ad['f1']:.4f} ({d_f1:+.4f})   →  {verdict}"
        )

    if rf is not None:
        _line("RandomForest:", rf)
    if mlp is not None:
        _line("MLP (128,64):", mlp)


def main():
    print("=" * 72)
    print("  AGGREGATOR — model results summary")
    print("=" * 72)
    print(f"  Reading metrics from: {RESULTS_DIR}/<model_slug>/metrics.json")

    payloads = load_metrics_files(RESULTS_DIR)
    if not payloads:
        print(
            "\n  No metrics files found. Run the train_<model>.py scripts first.",
            file=sys.stderr,
        )
        return

    payloads = order_payloads(payloads)
    missing = missing_from_preferred(payloads)

    print(f"  Found {len(payloads)} model result file(s).")
    if missing:
        print(f"  Missing from the expected lineup: {missing}")
    print()

    rows = [row_from_payload(p) for p in payloads]
    print("=" * 72)
    print("  SUMMARY — Feature Engineering A/B comparison")
    print("=" * 72)
    print(_format_summary_table(rows))
    print(
        "\n  Reading the table: positive Δ means the engineered culinary "
        "features improved that metric over the baseline."
    )

    _print_linear_vs_nonlinear_verdict(payloads)


if __name__ == "__main__":
    main()
