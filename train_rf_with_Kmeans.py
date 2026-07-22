"""
This script feeds the Kmeans into the RF
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.train_utils import (
    DATASETS,
    PROJECT_ROOT,
    RANDOM_STATE,
    build_metrics_payload,
    fit_and_score,
    load_metrics,
    load_preprocessed,
    print_dataset_block,
    print_delta,
    save_metrics,
    save_predictions,
    save_test_index
)
from train_Kmeans import NUTRITION_COLS, PCA_VARIANCE

MODEL_SLUG= "RF_Kmeans"
MODEL_NAME="RF with Kmeans"
DISPLAY_NAME= "Random Forest with Kmeans features"
NUTRITION_COLS=NUTRITION_COLS
PCA_VARIANCE=PCA_VARIANCE
N_CLUSTERS=3 #Chosen after clustering training
RF_CONFIG = {"n_estimators":200, "n_jobs":-1, "random_state":RANDOM_STATE}

def _add_cluster_columns(X_train, X_test):
    """
    fits PCA + Kmeans on the train data
    """
    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    Z_train = pca.fit_transform(X_train.drop(columns=list(NUTRITION_COLS)))
    Z_test = pca.transform(X_test.drop(columns=list(NUTRITION_COLS)))
    Kmeans=KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    Kmeans.fit(Z_train)
    augmented = []
    for x,z in ((X_train, Z_train), (X_test, Z_test)):
        labels = Kmeans.predict(z)
        distance = Kmeans.transform(z)
        extra= pd.DataFrame(index=x.index)
        for c in range(N_CLUSTERS):
            extra[f"cluster_{c}"] = (labels == c).astype(float)
        for c in range(N_CLUSTERS):
            extra[f"dist_centroid_{c}"] = distance[:,c]
        augmented.append(pd.concat([x,extra], axis=1))
    return augmented

def main():
    print("="*72)
    print(f"  TRAIN - {DISPLAY_NAME}, (Random state={RANDOM_STATE}, K={N_CLUSTERS})")
    print("="*72)

    datasets, y_train, y_test=load_preprocessed()
    reference=load_metrics("random_forest")["datasets"]
    per_ds_results={}
    for ds in DATASETS:
        xtr, xts= datasets[ds]
        X_train, X_test = _add_cluster_columns(xtr, xts)
        results=fit_and_score(RandomForestClassifier(**RF_CONFIG), X_train, y_train, X_test, y_test)
        per_ds_results[ds]=results
        print_dataset_block(f"{ds} +Clusters", X_train.shape, results)
        ref=reference[ds]
        print(f"  Plain RF without clusters: Acc - {ref['accuracy']:.4f} F1 - {ref['f1']:.4f}")
        print(f"  RF with clusters: ACC - {results['accuracy']:.4f} F1 - {results['f1']:.4f}")
        print(f"  Affect of clusters ACC - {results['accuracy'] - ref['accuracy']:+.4f} F1 - {results['f1'] - ref['f1']:+.4f}")
        save_predictions(MODEL_SLUG, ds, results["y_pred"])

    print_delta(per_ds_results)
    save_test_index(MODEL_SLUG, y_test)
    payload=build_metrics_payload(
        model_name=MODEL_NAME,
        display_name=DISPLAY_NAME,
        model_config={"random_forest": RF_CONFIG, "n_clusters": N_CLUSTERS},
        n_train=len(y_train),
        n_test=len(y_test),
        random_state=RANDOM_STATE,
        per_dataset_results=per_ds_results,
        extras={
            "reference_no_clusters":{ds: {"accuracy": reference[ds]["accuracy"], "f1": reference[ds]["f1"]} for ds in DATASETS},
            "cluster_columns_effect":{ ds:{"accuracy": per_ds_results[ds]["accuracy"] - reference[ds]["accuracy"],
                                           "f1":per_ds_results[ds]["f1"]-reference[ds]["f1"] } for ds in DATASETS }
        }
    )
    metrix_path = save_metrics(MODEL_SLUG, payload)
    print(f"  Saved metrics to {metrix_path}")

if __name__ == "__main__":
    main()


