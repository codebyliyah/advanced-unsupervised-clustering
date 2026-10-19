import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)

from scipy.cluster.hierarchy import linkage, dendrogram


RANDOM_SEED = 42


# --------------------------------------------------
# DATASET PREPARATION
# --------------------------------------------------

def prepare_dataset(name):

    if name.lower() == "digits":
        data = load_digits()

    elif name.lower() == "wine":
        data = load_wine()

    else:
        raise ValueError("Dataset must be digits or wine")

    return data.data, data.target


# --------------------------------------------------
# KMEANS EXPERIMENT
# --------------------------------------------------

def run_kmeans_experiment(X, preprocessing="raw"):

    results = []

    if preprocessing == "standardized":

        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X)

    elif preprocessing == "pca_95":

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=0.95)
        X_proc = pca.fit_transform(X_scaled)

    else:

        X_proc = X

    seeds = [42, 52, 62, 72, 82]

    for k in range(2, 16):

        inertias = []
        silhouettes = []
        ch_scores = []
        db_scores = []

        for seed in seeds:

            km = KMeans(
                n_clusters=k,
                n_init=20,
                max_iter=300,
                random_state=seed
            )

            labels = km.fit_predict(X_proc)

            inertias.append(km.inertia_)

            silhouettes.append(
                silhouette_score(X_proc, labels)
            )

            ch_scores.append(
                calinski_harabasz_score(X_proc, labels)
            )

            db_scores.append(
                davies_bouldin_score(X_proc, labels)
            )

        results.append(
            {
                "preprocessing": preprocessing,
                "k": k,
                "inertia_mean": np.mean(inertias),
                "inertia_std": np.std(inertias),
                "silhouette_mean": np.mean(silhouettes),
                "silhouette_std": np.std(silhouettes),
                "ch_mean": np.mean(ch_scores),
                "db_mean": np.mean(db_scores),
            }
        )

    return pd.DataFrame(results)


# --------------------------------------------------
# BEST K
# --------------------------------------------------

def select_best_k(df, preprocessing="standardized"):

    subset = df[df["preprocessing"] == preprocessing]

    idx = subset["silhouette_mean"].idxmax()

    return int(subset.loc[idx, "k"])


# --------------------------------------------------
# GMM
# --------------------------------------------------

def run_gmm(X, k):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(
        n_components=k,
        random_state=RANDOM_SEED
    )

    return gmm.fit_predict(X_scaled)


# --------------------------------------------------
# CENTROIDS
# --------------------------------------------------

def plot_centroids(X, k, filename):

    km = KMeans(
        n_clusters=k,
        n_init=20,
        max_iter=300,
        random_state=RANDOM_SEED
    )

    km.fit(X)

    centroids = km.cluster_centers_

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    axes = axes.ravel()

    for i in range(k):

        axes[i].imshow(
            centroids[i].reshape(8, 8),
            cmap="gray"
        )

        axes[i].set_title(f"Cluster {i}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# --------------------------------------------------
# DENDROGRAM
# --------------------------------------------------

def plot_dendrogram(
    X,
    dataset_name,
    linkage_method,
    filename,
    truncate=False
):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    Z = linkage(
        X_scaled,
        method=linkage_method
    )

    plt.figure(figsize=(12, 6))

    if truncate:

        dendrogram(
            Z,
            truncate_mode="lastp",
            p=50
        )

    else:

        dendrogram(Z)

    plt.title(
        f"{dataset_name} Dendrogram ({linkage_method})"
    )

    plt.xlabel("Samples")
    plt.ylabel("Distance")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# --------------------------------------------------
# STABILITY
# --------------------------------------------------

def stability_kmeans(X, k, seeds):

    labels_list = []

    inertia_list = []

    silhouette_list = []

    for seed in seeds:

        km = KMeans(
            n_clusters=k,
            n_init=20,
            max_iter=300,
            random_state=seed
        )

        labels = km.fit_predict(X)

        labels_list.append(labels)

        inertia_list.append(km.inertia_)

        silhouette_list.append(
            silhouette_score(X, labels)
        )

    ari_scores = []

    for i in range(len(labels_list)):

        for j in range(i + 1, len(labels_list)):

            ari_scores.append(
                adjusted_rand_score(
                    labels_list[i],
                    labels_list[j]
                )
            )

    return pd.DataFrame(
        {
            "mean_inertia": [np.mean(inertia_list)],
            "std_inertia": [np.std(inertia_list)],
            "mean_silhouette": [np.mean(silhouette_list)],
            "std_silhouette": [np.std(silhouette_list)],
            "mean_pairwise_ari": [np.mean(ari_scores)],
            "std_pairwise_ari": [np.std(ari_scores)],
        }
    )
