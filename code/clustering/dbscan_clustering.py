from sklearn.cluster import DBSCAN
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import pandas as pd

"""
DBSCAN clustering implementation for argument embeddings
"""

def cluster_dbscan(embeddings, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = dbscan.fit_predict(embeddings)
    return cluster_labels

def plot_k_distance(embeddings, min_samples=5):
    neighbors = NearestNeighbors(n_neighbors=min_samples, metric='cosine')
    neighbors.fit(embeddings)
    distances, _ = neighbors.kneighbors(embeddings)
    k_distances = np.sort(distances[:, -1])
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_distances, marker='o', linestyle='--', color='b')
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{min_samples}-th Nearest Neighbor Distance")
    plt.title("k-Distance Graph for DBSCAN")
    plt.grid(True)
    plt.show()

def evaluate_dbscan(embeddings, eps_values, min_samples_values):
    best_score = -1
    best_params = None
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            labels = dbscan.fit_predict(embeddings)
            if len(set(labels) - {-1}) > 1:
                score = silhouette_score(embeddings, labels)
            else:
                score = -1 
            noise_ratio = np.sum(labels == -1) / len(labels)
            print(f"eps={eps:.2f}, min_samples={min_samples}, clusters={len(set(labels)-{-1})}, " f"silhouette={score:.4f}, noise={noise_ratio:.2%}")
            if score > best_score:
                best_score = score
                best_params = (eps, min_samples)
    print(f"Best Parameters: eps={best_params[0]}, min_samples={best_params[1]}, silhouette={best_score:.4f}")

if __name__ == "__main__":
    df = pd.read_csv("pipeline/final_knowledgebase.csv")
    df['st_embeddings'] = df['st_embeddings'].apply(lambda x: np.array(eval(x)))  
    st_embeddings = np.vstack(df['st_embeddings'].to_numpy()).astype('float32')

    plot_k_distance(st_embeddings, min_samples=5)

    eps_range = np.arange(0.1, 1.0, 0.1) 
    min_samples_range = range(3, 20, 2) 

    evaluate_dbscan(st_embeddings, eps_range, min_samples_range)
