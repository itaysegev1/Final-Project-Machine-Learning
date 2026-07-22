"""
This script splits the recipes data to group with Kmeans (clustering).
We deside to use here the same PCA space we used in KNN (After the improvement).
"""
import json
import numpy as np
import matplotlib, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from train_pca_knn_improved import NUTRITION_COLS
from src.train_utils import (
    PROJECT_ROOT,
    RANDOM_STATE,
    load_preprocessed,
    model_results_dir,
    save_figure
)
matplotlib.use('Agg')
MODEL_SLUG="Kmeans"
NUTRITION_COLS=NUTRITION_COLS
PCA_VARIANCE=0.90
SILHOUETTE_SAMPLE=4000
K_RANGE= range(3,21)

def _select_k(Z_train):
    """
    runs Kmeans for every k in K_RANGE. returns (K rows, Chosen K)
    """
    print("--- Selecting Kmeans ---")
    k_rows = []
    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(Z_train)
        sil=silhouette_score(Z_train, labels, sample_size=SILHOUETTE_SAMPLE, random_state=RANDOM_STATE)
        k_rows.append({"K": k, "inertia": float(kmeans.inertia_), "silhouette": float(sil)})
        print(f" -- K={k:<3} | inertia={kmeans.inertia_:>12.1f} | sil={sil:.4f}")
    chosen_k = max(k_rows, key=lambda r: r["silhouette"])["K"]
    print(f"--- Chosen K={chosen_k} ---")
    return k_rows, chosen_k

def _plot_k_selection(k_rows, chosen_k):
    """
    plot the elbow graph of choosing k
    """
    ks= [r["K"] for r in k_rows]
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(ks, [r["inertia"] for r in k_rows], "o-", color="blue")
    ax1.set_xlabel("number of clusters")
    ax1.set_ylabel("inertia", color="blue")
    ax2 = ax1.twinx()
    ax2.plot(ks,[r["silhouette"] for r in k_rows], "s-", color="red")
    ax2.set_ylabel("silhouette", color="red")
    ax1.axvline(x=chosen_k, color="green", linestyle=":")
    ax1.set_title("Elbow curve")
    fig.tight_layout()
    save_figure(MODEL_SLUG,"K_selection.png" ,fig)

def main():
    print("="*72)
    print(f"  CLUSTERING - Kmeans with PCA (random_state = {RANDOM_STATE})")
    print("="*72)

    datasets, y_train, y_test = load_preprocessed()
    X_train, X_test = datasets["Advanced"]
    #project to pca space
    pca=PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    Z_train = pca.fit_transform(X_train.drop(columns=list(NUTRITION_COLS)))
    Z_test = pca.transform(X_test.drop(columns=list(NUTRITION_COLS)))
    print(f"  PCA kept {int(pca.n_components_)} components for {int(PCA_VARIANCE*100)}% variance")

    #select k
    k_rows, chosen_k = _select_k(Z_train)
    _plot_k_selection(k_rows, chosen_k)

    kmeans = KMeans(n_clusters=chosen_k, random_state=RANDOM_STATE, n_init=10)
    train_labels = kmeans.fit_predict(Z_train)
    test_labels = kmeans.predict(Z_test)

    #save and summery
    out_dir= model_results_dir(MODEL_SLUG)
    np.save(out_dir/"cluster_train.npy", train_labels.astype(np.int8))
    np.save(out_dir/"cluster_test.npy", test_labels.astype(np.int8))
    sizes = [{"cluster":c, "train_size": int((train_labels==c).sum()), "test_size":int((test_labels==c).sum())}
             for c in range(chosen_k)]
    for size in sizes:
        print(f"  cluster {size['cluster']}: train={size['train_size']},  test={size['test_size']}")
    summery = {
        "space": {"dropped_columns": list(NUTRITION_COLS), "pca_variance": PCA_VARIANCE,
                  "pca_components":int(pca.n_components_)},
        "k_selection": k_rows,
        "chosen_k": chosen_k,
        "clusters": sizes,
        "random_state": RANDOM_STATE
    }
    with open(f'{out_dir}/clustering_summery.json', 'w', encoding="utf-8") as f:
        json.dump(summery, f, indent=2)
        f.write("\n")
    print(f"  summery saved to {out_dir}/clustering_summery.json'")

if __name__ == "__main__":
    main()
