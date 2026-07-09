"""
In this module we load the two data files, merge them, build the Hit/Miss target
and do the train/test split. Hit = rating >= 4.0, Miss = rating < 4.0.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "epi_r.csv")
JSON_PATH = os.path.join(DATA_DIR, "full_format_recipes.json")

RATING_THRESHOLD = 4.0  # >= 4.0 = "Hit" (1)
TEST_SIZE = 0.20        # 80 / 20 split
from src._constants import RANDOM_STATE

# the fields that exist in both files, together with the title they make the merge key
SHARED_NUMERIC = ("rating", "calories", "protein", "fat", "sodium")

# the raw text columns that only the json has
JSON_TEXT_COLUMNS = ("directions", "ingredients")

# title is just an id and rating is the target. we keep the csv "date" column
# because its a real tag (the fruit), the json publication date is never pulled in
NON_FEATURE_COLUMNS = ("title", "rating")

_MERGE_KEY = "_merge_key"


# 1. Data loading
def load_binary_matrix(csv_path=CSV_PATH):
    """
    the function loads epi_r.csv, tries utf-8 first and falls back to latin-1
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV at '{csv_path}'.")
    try:
        return pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin-1")


def load_recipe_text(json_path=JSON_PATH):
    """
    the function loads the recipes json into a data frame
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find JSON at '{json_path}'.")
    with open(json_path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    return pd.DataFrame(records)


# 2. Merge
def _build_merge_key(df):
    """
    builds the merge key from the normalized title + the 5 shared numeric fields.
    the numbers are rounded to 3 decimals so both files format the same.
    """
    def _fmt_num(series):
        numeric_series = pd.to_numeric(series, errors="coerce").round(3)
        return numeric_series.map(lambda v: "nan" if pd.isna(v) else f"{v:.3f}")

    merge_key = df["title"].fillna("").astype(str).str.strip().str.lower()
    for col in SHARED_NUMERIC:
        merge_key = merge_key + "|" + _fmt_num(df[col])
    return merge_key


def merge_datasets(binary_df, text_df, verbose=True):
    """
    here we merge the csv and the json. we drop duplicate keys on each side
    and then do a 1:1 inner join, taking only the text columns from the json.
    """
    binary_df = binary_df.copy()
    text_df = text_df.copy()

    binary_df[_MERGE_KEY] = _build_merge_key(binary_df)
    text_df[_MERGE_KEY] = _build_merge_key(text_df)

    num_csv_dupes = int(binary_df[_MERGE_KEY].duplicated().sum())
    num_json_dupes = int(text_df[_MERGE_KEY].duplicated().sum())
    binary_df = binary_df.drop_duplicates(subset=_MERGE_KEY, keep="first")
    text_df = text_df.drop_duplicates(subset=_MERGE_KEY, keep="first")

    text_cols_to_pull = [c for c in JSON_TEXT_COLUMNS if c in text_df.columns]
    missing_text_cols = set(JSON_TEXT_COLUMNS) - set(text_cols_to_pull)
    if missing_text_cols:
        raise KeyError(f"JSON is missing expected text columns: {sorted(missing_text_cols)}")
    text_subset = text_df[[_MERGE_KEY, *text_cols_to_pull]]

    merged = binary_df.merge(text_subset, on=_MERGE_KEY, how="inner", validate="1:1")
    merged = merged.drop(columns=_MERGE_KEY)

    if verbose:
        print(
            f"[merge] CSV rows={len(binary_df) + num_csv_dupes} "
            f"(dropped {num_csv_dupes} dup keys) | "
            f"JSON rows={len(text_df) + num_json_dupes} "
            f"(dropped {num_json_dupes} dup keys) -> merged={len(merged)}"
        )
    return merged


# 3. Target + cleaning + split
def clean_and_binarize(df, verbose=True):
    """
    drops rows with no rating or no directions and builds the target y
    """
    num_rows_before = len(df)
    has_rating = df["rating"].notna()
    has_directions = df["directions"].apply(
        lambda d: isinstance(d, (list, tuple, np.ndarray)) and len(d) > 0
    )
    df = df.loc[has_rating & has_directions].reset_index(drop=True)

    # binarize the rating to get y
    y = (df["rating"] >= RATING_THRESHOLD).astype(int)
    y.name = "is_hit"

    if verbose:
        print(
            f"[clean] dropped {num_rows_before - len(df)} rows missing rating/directions "
            f"-> {len(df)} usable recipes"
        )
        num_hits = int(y.sum())
        print(
            f"[target] Hit(1)={num_hits} ({100 * num_hits / len(y):.1f}%) | "
            f"Miss(0)={len(y) - num_hits} ({100 * (len(y) - num_hits) / len(y):.1f}%)"
        )
    return df, y


def build_feature_frame(df):
    """
    returns X: the nutrition + binary tags + the raw text columns
    """
    cols_to_drop = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def split_data(X, y):
    """
    stratified 80/20 train/test split, done before any feature engineering
    """
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# 4. The custom transformer
class CulinaryFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    This class makes the culinary features from the recipe text: 3 numeric ones
    (num_steps, num_ingredients, avg_words_per_step) and 6 binary has_* columns,
    one for each keyword group. use make_culinary_extractor() to get it populated.
    """

    KEYWORD_GROUPS = (
        "high_heat_techniques",
        "low_and_slow_techniques",
        "technical_execution",
        "prep_and_patience",
        "flavor_development",
        "premium_ingredients",
    )

    BASELINE_FEATURES = (
        "num_steps",
        "num_ingredients",
        "avg_words_per_step",
    )

    def __init__(
        self,
        directions_col="directions",
        ingredients_col="ingredients",
        high_heat_techniques=(),
        low_and_slow_techniques=(),
        technical_execution=(),
        prep_and_patience=(),
        flavor_development=(),
        premium_ingredients=(),
    ):
        self.directions_col = directions_col
        self.ingredients_col = ingredients_col

        # saved as is (no copy or validation) like sklearn wants, so clone() works
        self.high_heat_techniques = high_heat_techniques
        self.low_and_slow_techniques = low_and_slow_techniques
        self.technical_execution = technical_execution
        self.prep_and_patience = prep_and_patience
        self.flavor_development = flavor_development
        self.premium_ingredients = premium_ingredients

    def fit(self, X, y=None):
        """
        stateless fit, just checks the columns exist and saves the output names
        """
        for col in (self.directions_col, self.ingredients_col):
            if col not in X.columns:
                raise KeyError(f"CulinaryFeatureExtractor: missing column '{col}'.")
        self.feature_names_out_ = list(self.BASELINE_FEATURES) + [
            f"has_{group}" for group in self.KEYWORD_GROUPS
        ]
        return self

    def transform(self, X):
        """
        builds the feature matrix (same index as X)
        """
        directions = X[self.directions_col].apply(self._to_token_list)
        ingredients = X[self.ingredients_col].apply(self._to_token_list)

        out = pd.DataFrame(index=X.index)

        # the 3 numeric features
        out["num_steps"] = directions.apply(len)
        out["num_ingredients"] = ingredients.apply(len)
        out["avg_words_per_step"] = directions.apply(self._avg_words_per_step)

        # combined lowercase text for the keyword search
        combined_text = (
            directions.apply(lambda steps: " ".join(steps))
            + " "
            + ingredients.apply(lambda items: " ".join(items))
        ).str.lower()

        # one binary column per keyword group
        for group in self.KEYWORD_GROUPS:
            group_keywords = getattr(self, group)
            column = f"has_{group}"
            if not group_keywords:
                out[column] = 0  # group not populated -> all zeros
            else:
                keyword_pattern = self._build_keyword_pattern(group_keywords)
                out[column] = combined_text.str.contains(
                    keyword_pattern, regex=True, na=False
                ).astype(int)

        return out

    def get_feature_names_out(self, input_features=None):
        """
        returns the output feature names in the same order as transform
        """
        return np.asarray(
            list(self.BASELINE_FEATURES)
            + [f"has_{group}" for group in self.KEYWORD_GROUPS]
        )

    @staticmethod
    def _to_token_list(value):
        """
        turns a cell (list / string / NaN) into a list of strings
        """
        if isinstance(value, (list, tuple, np.ndarray)):
            return [str(v) for v in value]
        if isinstance(value, str):
            return [value]
        return []

    @staticmethod
    def _avg_words_per_step(steps):
        """
        the mean number of words per direction step, 0.0 when there are no steps
        """
        if not steps:
            return 0.0
        return float(np.mean([len(step.split()) for step in steps]))

    @staticmethod
    def _build_keyword_pattern(keywords):
        """
        builds a whole word, case insensitive regex from the keywords
        """
        import re

        escaped_keywords = [re.escape(kw.lower()) for kw in keywords if str(kw).strip()]
        return r"\b(?:" + "|".join(escaped_keywords) + r")\b"


# the keyword groups. some keywords are on purpose in more then one group
# (like "smoke") because the groups describe overlapping ideas, not a partition
CULINARY_KEYWORDS = {
    "high_heat_techniques": (
        "sear", "saute", "sauté", "broil", "stir-fry", "pan-fry", "grill", "deep-fry", "blanch", "char", "blacken",
        "flambe", "flambé", "flash-fry", "scald", "scorch", "blister", "wok-fry", "sear-roast",
    ),
    "low_and_slow_techniques": (
        "sous vide", "confit", "braise", "slow-roast", "simmer", "stew", "poach", "sweat", "render", "coddle",
        "steep", "infuse", "barbecue", "bbq", "smoke", "baste", "slow-cook", "roast",
    ),
    "technical_execution": (
        "temper", "emulsify", "deglaze", "clarify", "monter au beurre", "puree", "purée", "strain", "knead",
        "muddle", "macerate", "score", "dredge", "whip", "skim", "butterfly", "truss", "chiffonade", "supreme",
        "debone", "fillet", "zest", "bind", "thicken", "mount",  "crimp", "pipe", "julienne", "brunoise", "batonnet",
        "oblique", "paysanne", "shave", "mandoline", "mortar and pestle", "ice-cream maker", "spice mill",
        "double boiler", "candy thermometer", "deep-fat thermometer", "steamer insert", "dariole molds",
        "springform pan", "dutch oven", "cleaver", "blowtorch", "piping bag", "immersion circulator","sous vide machine",
        "vacuum sealer", "proof", "punch down", "blind bake", "dock", "flute", "laminate", "bloom", "prove",
    ),
    "prep_and_patience": (
        "marinate", "brine", "ferment", "overnight", "rest", "cure", "soak", "rise", "proof", "age", "pickle", "steep",
        "dry-rub", "bloom", "activate", "temper",
    ),
    "flavor_development": (
        "reduce", "caramelize", "smoke", "jus", "char", "glaze", "infuse", "zest", "baste", "sweat", "render",
        "extract", "curing", "smoke-infuse", "dry-roast", "aioli", "hollandaise", "bearnaise", "bechamel", "veloute",
        "espagnole", "demi-glace", "roux", "slurry", "chutney", "compote", "pesto", "chimichurri", "gastrique",
        "coulis", "gremolata", "mignonette", "compound butter",
    ),
    "premium_ingredients": (
        "truffle", "truffles", "saffron", "bone marrow", "wagyu", "caviar", "dry-aged", "foie gras", "kobe",
        "edible gold", "vanilla bean", "chanterelle", "morel", "porcini", "lobster", "langoustine",
        "oyster", "scallops", "prosciutto", "iberico", "quail", "duck breast", "sweetbreads", "duck confit", "cognac",
        "armagnac", "pancetta", "guanciale", "uni", "bottarga", "matsutake", "beluga", "fish sauce", "oyster sauce",
        "hoisin", "sriracha", "gochujang", "miso", "mirin", "sake", "tahini", "curry paste", "garam masala",
        "harissa", "zaatar", "sumac", "tamarind", "preserved lemon", "chipotle", "ancho", "guajillo", "masa",
        "tomatillo", "dashi", "katsuobushi",
    ),
}

# the dict keys must match the extractor group names exactly
assert set(CULINARY_KEYWORDS) == set(CulinaryFeatureExtractor.KEYWORD_GROUPS), (
    "CULINARY_KEYWORDS keys drifted from CulinaryFeatureExtractor.KEYWORD_GROUPS"
)


def make_culinary_extractor(directions_col="directions", ingredients_col="ingredients"):
    """
    returns a CulinaryFeatureExtractor with the keywords already inside.
    a bare CulinaryFeatureExtractor() has empty groups and gives all zero columns.
    """
    return CulinaryFeatureExtractor(
        directions_col=directions_col,
        ingredients_col=ingredients_col,
        **CULINARY_KEYWORDS,
    )


def build_dataset(verbose=True):
    """
    runs the full data pipeline and returns (X_train, X_test, y_train, y_test)
    """
    binary_df = load_binary_matrix()
    text_df = load_recipe_text()

    merged = merge_datasets(binary_df, text_df, verbose=verbose)
    merged, y = clean_and_binarize(merged, verbose=verbose)
    X = build_feature_frame(merged)

    return split_data(X, y)


def main():
    """
    builds the dataset and prints a small sanity check
    """
    X_train, X_test, y_train, y_test = build_dataset(verbose=True)

    print("\n" + "=" * 60)
    print("DATA FOUNDATION READY")
    print("=" * 60)
    print(f"X_train shape : {X_train.shape}")
    print(f"X_test  shape : {X_test.shape}")
    print(f"y_train shape : {y_train.shape}  (hit rate {y_train.mean():.3f})")
    print(f"y_test  shape : {y_test.shape}  (hit rate {y_test.mean():.3f})")

    extractor = make_culinary_extractor()

    # fit on train only, transform both splits so there is no leakage
    culinary_train = extractor.fit_transform(X_train)
    culinary_test = extractor.transform(X_test)

    print("\nCulinaryFeatureExtractor — engineered feature matrix:")
    print(f"  culinary_train shape : {culinary_train.shape}")
    print(f"  culinary_test  shape : {culinary_test.shape}")

    print("\nKeyword group activation on X_train (positives / total):")
    num_train_rows = len(culinary_train)
    for group in CulinaryFeatureExtractor.KEYWORD_GROUPS:
        col = f"has_{group}"
        num_positives = int(culinary_train[col].sum())
        num_keywords = len(getattr(extractor, group))
        print(
            f"  {col:<32} {num_positives:>5} / {num_train_rows}  "
            f"({100 * num_positives / num_train_rows:5.1f}%)  [{num_keywords} keywords]"
        )

    print("\nBaseline numeric features (X_train) — describe():")
    print(culinary_train[list(CulinaryFeatureExtractor.BASELINE_FEATURES)]
          .describe().round(2).to_string())


if __name__ == "__main__":
    main()
