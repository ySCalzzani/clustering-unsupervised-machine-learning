import pandas as pd
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import hdbscan
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity

# Carregar os dados de um arquivo CSV.
load_data = pd.read_csv('/home/aristotelicfool/Home/coding/IC/dados_normalizados.csv')

# Cria dicionário para utilizar diferentes formas do banco de dados carregados.
data = {
    'all': load_data,
    'ADI-R': load_data[["adi_r_diagnostic_a_total", "adi_r_diagnostic_b_total", "adi_r_diagnostic_c_total",
                        "ssc_core_descriptive_ssc_diagnosis_full_scale_iq"]],
    'without_CBCL': load_data[["vineland_communication_standard", "vineland_dls_standard", "vineland_soc_standard",
                               "adi_r_diagnostic_a_total", "adi_r_diagnostic_b_total",
                               "ssc_core_descriptive_ssc_diagnosis_full_scale_iq"]]
}

metrics = {
    1: 'euclidean',
    2: 'manhattan',
    3: 'chebyshev',
    4: 'minkowski'
}


def umap_reduction(data, n_neighbors, min_dist, n_components, metric):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                        n_components=n_components)
    embedding = reducer.fit_transform(data)
    return embedding


def dbscan_clustering(embedding, eps, min_samples, metric, p=None):
    if metric == 'minkowski' and p is not None:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, p=p)
    else:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    dbscan_labels = dbscan.fit_predict(embedding)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=dbscan_labels, cmap='Spectral', s=5)
    plt.title(f'DBSCAN: eps={eps}, min_samples={min_samples}, metric={metric}')
    plt.show()

    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"Number of clusters found by DBSCAN: {n_clusters}")




for key, metric in metrics.items():
    print(f"Testing UMAP with metric: {metric}")
    default = umap_reduction(data['without_CBCL'], 15, 0.1, 2, metric)
    for key, metric in metrics.items():
        print(f"Testing DBSCAN with metric: {metric}")
        if metric == 'minkowski':
            dbscan_clustering(default, 0.5, 5, metric, p=1)  # Specify p=2 or another value as needed
        else:
            dbscan_clustering(default, 0.5, 5, metric)

