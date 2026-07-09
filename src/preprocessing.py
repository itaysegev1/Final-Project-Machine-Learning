"""
In this module we build the two feature matrices: Baseline (scaled nutrition +
binary tags) and Advanced (Baseline + the 9 culinary features). everything is
fit on train only so there is no leakage.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.data_foundation import (
    CulinaryFeatureExtractor,
    build_dataset,
    make_culinary_extractor,
)


NUMERIC_COLUMNS = ("calories", "protein", "fat", "sodium")
TEXT_COLUMNS = ("directions", "ingredients", "categories")

# dropped only if they exist, the csv "date" tag column is kept
DROP_IF_PRESENT = ("title", "desc", "categories")

# the 3 numeric culinary features that need scaling, the 6 has_* pass as is
CULINARY_NUMERIC = CulinaryFeatureExtractor.BASELINE_FEATURES


# Column classification
def classify_columns(X):
    """
    splits the columns of X to (numeric, text, binary_tags, to_drop),
    every column lands in exactly one group and we verify it
    """
    cols = list(X.columns)

    numeric = [c for c in NUMERIC_COLUMNS if c in cols]
    text = [c for c in TEXT_COLUMNS if c in cols]
    to_drop = [c for c in DROP_IF_PRESENT if c in cols]
    binary_tags = [
        c for c in cols
        if c not in numeric and c not in text and c not in to_drop
    ]

    # sanity check that we didnt miss or double count a column
    accounted = numeric + text + binary_tags + to_drop
    assert sorted(accounted) == sorted(cols), (
        "Column classification missed or double-counted some columns."
    )

    for col in numeric:
        if not pd.api.types.is_numeric_dtype(X[col]):
            raise TypeError(
                f"Column '{col}' should be numeric but has dtype {X[col].dtype}."
            )

    return numeric, text, binary_tags, to_drop


# Pipeline factories
def _numeric_pipeline():
    """
    median impute + RobustScaler for the nutrition columns. the scaler centers
    on the train median so an imputed value lands exactly at 0.
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]
    )


def build_baseline_preprocessor(numeric_cols, binary_tag_cols):
    """
    the Baseline preprocessor: scaled nutrition + the raw binary tags, no text
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", _numeric_pipeline(), numeric_cols),
            ("tags", "passthrough", binary_tag_cols),
        ],
        remainder="drop",   # drops directions/ingredients and anything else
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def _build_culinary_pipeline():
    """
    extracts the 9 culinary features and scales only the 3 numeric ones.
    the extractor must come from make_culinary_extractor() so the keywords are in.
    """
    return Pipeline(
        steps=[
            ("extract", make_culinary_extractor()),
            (
                "scale_numeric",
                ColumnTransformer(
                    transformers=[
                        ("scaled", RobustScaler(), list(CULINARY_NUMERIC)),
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            ),
        ]
    )


def build_advanced_preprocessor(numeric_cols, text_cols, binary_tag_cols):
    """
    the Advanced preprocessor: same as Baseline + the culinary features
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", _numeric_pipeline(), numeric_cols),
            ("tags", "passthrough", binary_tag_cols),
            ("culinary", _build_culinary_pipeline(), text_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


# Fit / transform
def _assert_no_dead_binary_features(X_train_out):
    """
    fails loud if a has_* column is all zeros on train, that means the
    keywords were never injected to the extractor
    """
    has_cols = [c for c in X_train_out.columns if c.startswith("has_")]
    dead = [c for c in has_cols if int(X_train_out[c].sum()) == 0]
    if dead:
        raise RuntimeError(
            f"Engineered binary features are all-zero on the train split: {dead}. "
            "The CulinaryFeatureExtractor was probably built without its keyword "
            "groups — use make_culinary_extractor() (see src/data_foundation.py)."
        )


def fit_transform_pair(preprocessor, X_train, X_test):
    """
    fit only on train, transform both splits
    """
    X_train_out = preprocessor.fit_transform(X_train)
    X_test_out = preprocessor.transform(X_test)
    return X_train_out, X_test_out


def build_preprocessed_datasets(verbose=True):
    """
    builds both matrices end to end. returns (X_train_baseline, X_test_baseline,
    X_train_advanced, X_test_advanced, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = build_dataset(verbose=verbose)

    numeric, text, binary_tags, to_drop = classify_columns(X_train)

    if verbose:
        print(
            f"[columns] numeric={len(numeric)} | text={len(text)} | "
            f"binary_tags={len(binary_tags)} | dropped={len(to_drop)} "
            f"{to_drop if to_drop else '(none — defensive drop list is a no-op here)'}"
        )

    # Baseline
    baseline = build_baseline_preprocessor(numeric, binary_tags)
    X_train_baseline, X_test_baseline = fit_transform_pair(baseline, X_train, X_test)

    # Advanced
    advanced = build_advanced_preprocessor(numeric, text, binary_tags)
    X_train_advanced, X_test_advanced = fit_transform_pair(advanced, X_train, X_test)

    # the engineered binary features must actually fire on real recipes
    _assert_no_dead_binary_features(X_train_advanced)

    if verbose:
        has_cols = [c for c in X_train_advanced.columns if c.startswith("has_")]
        rates = {c: f"{100 * X_train_advanced[c].mean():.1f}%" for c in has_cols}
        print(f"[culinary] keyword-group activation on train: {rates}")

    return (
        X_train_baseline, X_test_baseline,
        X_train_advanced, X_test_advanced,
        y_train, y_test,
    )


def main():
    """
    builds both matrices and prints a small sanity check
    """
    (
        X_train_baseline, X_test_baseline,
        X_train_advanced, X_test_advanced,
        y_train, y_test,
    ) = build_preprocessed_datasets(verbose=True)

    print("\n" + "=" * 60)
    print("PREPROCESSED MATRICES READY")
    print("=" * 60)

    print("\nBaseline (no engineered text features):")
    print(f"  X_train_baseline shape : {X_train_baseline.shape}")
    print(f"  X_test_baseline  shape : {X_test_baseline.shape}")

    print("\nAdvanced (baseline + 9 culinary features):")
    print(f"  X_train_advanced shape : {X_train_advanced.shape}")
    print(f"  X_test_advanced  shape : {X_test_advanced.shape}")

    delta = X_train_advanced.shape[1] - X_train_baseline.shape[1]
    print(f"\nDelta: Advanced adds {delta} engineered columns over Baseline.")
    print(f"  (expected = 9 from CulinaryFeatureExtractor: "
          f"{len(CULINARY_NUMERIC)} numeric + "
          f"{len(CulinaryFeatureExtractor.KEYWORD_GROUPS)} binary)")

    # on train we expect median ~0 and IQR ~1 (RobustScaler, not StandardScaler)
    def _robust_stats(df):
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        return pd.DataFrame({
            "median": df.median().round(3),
            "iqr": (q3 - q1).round(3),
        })

    nutr_train_baseline = X_train_baseline[list(NUMERIC_COLUMNS)]
    print("\nSanity — RobustScaler on X_train_baseline (median ~0, IQR ~1):")
    print(_robust_stats(nutr_train_baseline).to_string())

    # the test split was scaled with the train stats so its not exactly 0/1
    nutr_test_baseline = X_test_baseline[list(NUMERIC_COLUMNS)]
    print("\nSanity — same columns on X_test_baseline "
          "(medians/IQRs need NOT be 0/1; scaler was fit on train only):")
    print(_robust_stats(nutr_test_baseline).to_string())

    print(f"\ny_train: {y_train.shape}  (hit rate {y_train.mean():.3f})")
    print(f"y_test : {y_test.shape}  (hit rate {y_test.mean():.3f})")


if __name__ == "__main__":
    main()
