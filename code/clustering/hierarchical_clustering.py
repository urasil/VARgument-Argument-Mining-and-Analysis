from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
import matplotlib.pyplot as plt
from embeddings import EmbeddingsGenerator
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

"""
Hierarchical clustering implementation for argument embeddings
"""

class HierarchicalClustering:
    def __init__(self, method='ward', criterion='maxclust'):
        self.method = method
        self.criterion = criterion
        self.linkage_matrix = None
        self.cluster_labels = None

    def fit(self, embeddings, max_clusters=10):
        #max-clusters useless as fcluster determines the number of clusters for hierarchical
        self.linkage_matrix = linkage(embeddings, method=self.method)
        
        max_d = np.median(self.linkage_matrix[:, 2])  # median distance used as threshold
        self.cluster_labels = fcluster(self.linkage_matrix, max_d, criterion=self.criterion)
        return self.cluster_labels
    
    def plot_dendrogram(self):

        plt.figure(figsize=(10, 5))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        dendrogram(self.linkage_matrix)
        plt.show()

if __name__ == "__main__":
    df = pd.read_csv("pipeline/final_knowledgebase.csv")
    df['st_embeddings'] = df['st_embeddings'].apply(lambda x: np.array(eval(x)))  
    st_embeddings = np.vstack(df['st_embeddings'].to_numpy()).astype('float32')
    
    clustering_model = HierarchicalClustering()
    cluster_labels = clustering_model.fit(st_embeddings)
    print("Cluster Labels:", cluster_labels)
    
    clustering_model.plot_dendrogram()
