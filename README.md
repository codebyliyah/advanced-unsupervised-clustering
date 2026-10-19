# Advanced Unsupervised Clustering

## Project Overview

This project investigates unsupervised clustering using the Digits and Wine datasets available through scikit-learn. The analysis compares K-Means, hierarchical clustering, and Gaussian Mixture Models (GMM).

The project evaluates cluster quality, dimensionality reduction, hierarchical structure, algorithm performance, and clustering stability.

## Datasets and Methods

**Datasets**
- Digits dataset: handwritten digit images represented by 64 pixel features.
- Wine dataset: chemical measurements represented by 13 numerical features.

**Methods**
- K-Means clustering
- Principal Component Analysis (PCA)
- Hierarchical clustering
- Gaussian Mixture Models
- Internal and external clustering evaluation
- Repeated-run stability analysis

Target labels are reserved for external evaluation and are not used to fit clustering models.

## Repository Structure

- `advanced_clustering.ipynb`: Main analysis notebook.
- `src/clustering_analysis.py`: Reusable clustering functions.
- `figures/`: Generated clustering visualizations and dendrograms.
- `results/`: CSV files containing model-selection, comparison, and stability results.
- `requirements.txt`: Required Python libraries.

## Results

### K-Means Model Selection

K-Means was evaluated for cluster counts from 2 through 15 using raw pixel values, standardized features, and PCA features retaining at least 95% of the variance.

For raw Digits features, the highest silhouette score in the supplied results was approximately 0.189 at k = 9. The lowest Davies-Bouldin score was approximately 1.758 at k = 9.

These metrics provide evidence for comparing cluster counts, but the final choice should consider all metrics, preprocessing configurations, and the resulting cluster structure.

### Algorithm Comparison

#### Digits Dataset

| Algorithm | Silhouette | Davies-Bouldin | ARI | NMI |
|---|---:|---:|---:|---:|
| K-Means | 0.183 | 1.888 | 0.706 | 0.771 |
| Hierarchical Ward | 0.174 | 1.927 | 0.813 | 0.869 |
| Gaussian Mixture Model | 0.160 | 1.843 | 0.592 | 0.704 |

Ward hierarchical clustering had the highest ARI and NMI, indicating the strongest agreement with the known digit labels among these three methods. K-Means had the highest silhouette score.

#### Wine Dataset

| Algorithm | Silhouette | Davies-Bouldin | ARI | NMI |
|---|---:|---:|---:|---:|
| K-Means | 0.516 | 0.497 | 0.138 | 0.325 |
| Hierarchical Ward | 0.496 | 0.501 | 0.141 | 0.326 |
| Gaussian Mixture Model | -0.207 | 9.351 | 0.337 | 0.568 |

K-Means produced the strongest silhouette score for Wine, suggesting better separation according to this internal metric. GMM achieved higher agreement with the known labels but had poor internal clustering scores. This demonstrates that cluster separation and agreement with known classes measure different properties.

### Stability Analysis

Across repeated K-Means runs, the reported results were:

- Mean inertia: 1,099,537
- Standard deviation of inertia: 1,260.30
- Mean silhouette score: 0.1831
- Standard deviation of silhouette score: 0.000465
- Mean pairwise ARI: 0.9529
- Standard deviation of pairwise ARI: 0.0448

The high mean pairwise ARI indicates that the K-Means assignments were highly consistent across the tested runs.

## Required Discussion

### 1. How did standardization affect K-Means?

Standardization changes the scale of the features before distance calculations. Features with larger numerical ranges can dominate K-Means when raw values are used. The standardized results should be compared with raw and PCA-based results to determine how scaling affected cluster quality.

### 2. Did all four cluster-selection metrics recommend the same k?

The metrics measure different aspects of clustering. Inertia decreases as the number of clusters increases, while silhouette, Calinski-Harabasz, and Davies-Bouldin evaluate other aspects of cluster structure. Their preferred values of k must be compared rather than assuming they agree.

### 3. How much variance did the two PCA visualization components preserve?

The variance preserved by the two visualization components is the sum of their individual explained-variance ratios. The exact percentage should be taken from the fitted two-component PCA used to generate the Digits visualization.

### 4. Why can overlapping PCA clusters still be useful?

PCA projects high-dimensional data into two dimensions, which can discard information. Clusters that overlap in a two-dimensional visualization may remain distinguishable using the original feature space or additional principal components.

### 5. Which hierarchical linkage method performed best for each dataset?

The preferred linkage method should be selected using the internal validation metrics and cophenetic correlation coefficient. The complete linkage-comparison results are needed to identify the best method for each dataset.

### 6. Why did single linkage behave differently from Ward linkage?

Single linkage uses the minimum distance between points in two clusters and can form elongated chains of observations. Ward linkage merges clusters to minimize the increase in within-cluster variance and therefore tends to form more compact groups. Ward linkage requires Euclidean distance.

### 7. Did the data-driven number of clusters match the known number of classes?

This must be determined from the selected cluster counts in the experiments. The known class counts are useful for reference but do not establish the correct number of unsupervised clusters.

### 8. Why should known labels not be used to select clustering parameters?

Using target labels to choose the number of clusters or tune the clustering model would introduce supervised information into an unsupervised experiment. Labels should be reserved for external evaluation after clustering.

### 9. What is the difference between internal and external clustering metrics?

Internal metrics, such as silhouette and Davies-Bouldin, evaluate the geometry and separation of clusters without requiring target labels. External metrics, such as ARI and NMI, compare predicted cluster assignments with known labels.

### 10. Which algorithm produced the most stable results?

The repeated K-Means experiments had a mean pairwise ARI of approximately 0.953. This indicates strong consistency across runs. A complete comparison with the stability results for other algorithms is needed before declaring an overall winner.

### 11. Which algorithm would you recommend for each dataset?

For Digits, Ward hierarchical clustering is a promising choice when agreement with known labels is important, while K-Means provides the highest silhouette score among the compared methods.

For Wine, K-Means is a strong choice based on silhouette score and cluster separation. GMM has higher external agreement with the known labels, but its negative silhouette and high Davies-Bouldin score warrant caution.

The final recommendations should also consider execution time, stability, and the purpose of the clustering.

### 12. What limitations prevent these results from proving that the known classes are the only valid clusters?

Clustering results depend on preprocessing, distance measures, algorithms, parameter choices, and the geometry of the data. Different methods can reveal different valid structures. Agreement with known labels does not prove that those labels are the only meaningful grouping, and disagreement does not automatically indicate a failed clustering model.

## Reproducibility

Install the required libraries using:

`pip install -r requirements.txt`

Run the notebook in an environment with the required dependencies installed. The analysis should use fixed random seeds, save generated figures and CSV results, and exclude target labels from all clustering model-fitting stages.

## Files and Outputs

The `figures/` directory contains the PCA cluster plot, K-Means centroid images, and Digits and Wine dendrograms.

The `results/` directory contains the K-Means model-selection metrics, algorithm-comparison results, and stability analysis.
