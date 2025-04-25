from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import pandas as pd
from sklearn.metrics import silhouette_score

"""
kmeans clustering implementation for argument embeddings
"""

# performs kmeans, yellowbrick library for finding optimal k as well as visualisations for dissertation
def cluster_kmeans(embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init=20, max_iter=300, random_state=31)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

def optimal_k_using_elbow(embeddings, max_k=15):
    kmeans = KMeans(init='k-means++', n_init=20, max_iter=300, random_state=31)
    visualizer = KElbowVisualizer(kmeans, k=(2, max_k), metric='calinski_harabasz')  
    visualizer.fit(embeddings)
    visualizer.show()

def optimal_k_using_silhouette(embeddings, max_k=15):
    silhouette_scores = []
    k_values = list(range(2, max_k + 1))
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20, max_iter=300, random_state=31)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, silhouette_scores, marker='o', linestyle='--')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("pipeline/final_knowledgebase.csv")
    df['st_embeddings'] = df['st_embeddings'].apply(lambda x: np.array(eval(x)))  
    st_embeddings = np.vstack(df['st_embeddings'].to_numpy()).astype('float32')

    optimal_k_using_elbow(st_embeddings, max_k=20)
    optimal_k_using_silhouette(st_embeddings, max_k=20)
