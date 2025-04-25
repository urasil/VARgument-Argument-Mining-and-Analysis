import pandas as pd
import os
import numpy as np
from sklearn.metrics import silhouette_score, mutual_info_score
from kmeans_clustering import cluster_kmeans
from dbscan_clustering import cluster_dbscan
from hierarchical_clustering import HierarchicalClustering

"""
Comparing the performance of the 3 main clustering methods: kmeans, dbscan, hierarchical
"""

knowledge_base_path = os.path.join("pipeline", "final_knowledgebase.csv")
df = pd.read_csv(knowledge_base_path)
df['st_embeddings'] = df['st_embeddings'].apply(lambda x: np.array(eval(x)))
embeddings = np.vstack(df['st_embeddings'].to_numpy()).astype('float32')

kmeans_labels = cluster_kmeans(embeddings, num_clusters=5)
dbscan_labels = cluster_dbscan(embeddings, eps=0.5, min_samples=5)
hierarchical_clustering = HierarchicalClustering()
hierarchical_labels = hierarchical_clustering.fit(embeddings)

def calculate_metrics(embeddings, labels):
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    silhouette_avg = silhouette_score(embeddings, labels)
    mutual_info = mutual_info_score(labels, labels)
    return num_clusters, silhouette_avg, mutual_info

kmeans_metrics = calculate_metrics(embeddings, kmeans_labels)
dbscan_metrics = calculate_metrics(embeddings, dbscan_labels)
hierarchical_metrics = calculate_metrics(embeddings, hierarchical_labels)

#honestly, why
latex_table = f"""
\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{|c|c|c|c|}}
\\hline
Clustering Method & Number of Clusters & Silhouette Score & Mutual Information \\\\
\\hline
KMeans & {kmeans_metrics[0]} & {kmeans_metrics[1]:.4f} & {kmeans_metrics[2]:.4f} \\\\
DBSCAN & {dbscan_metrics[0]} & {dbscan_metrics[1]:.4f} & {dbscan_metrics[2]:.4f} \\\\
Hierarchical & {hierarchical_metrics[0]} & {hierarchical_metrics[1]:.4f} & {hierarchical_metrics[2]:.4f} \\\\
\\hline
\\end{{tabular}}
\\caption{{Clustering Performance Comparison}}
\\label{{tab:clustering_performance}}
\\end{{table}}
"""

print(latex_table)