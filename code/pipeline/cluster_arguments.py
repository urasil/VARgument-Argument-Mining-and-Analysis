import os
import sys
import spacy
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clustering.kmeans_clustering import cluster_kmeans
from clustering.entity_clustering import cluster_sentences_with_entities, extract_specific_entities
from clustering.topic_clustering import cluster_by_topics
from clustering.dbscan_clustering import cluster_dbscan
from clustering.embeddings import EmbeddingsGenerator
from clustering.hierarchical_clustering import HierarchicalClustering

# word embeddings 2d plot
# how to move around the 2d plot

# can cluster with dbscan and kmeans and assign topic with llm


"""
Clustering class for using the separately defined clustering methods
"""

class Clustering:
    def __init__(self, num_clusters=5, eps=0.5, min_samples=10):
        self.num_clusters = num_clusters
        self.embeddings_gen = EmbeddingsGenerator(model_name="ft")
        self.eps = eps
        self.min_samples = min_samples
    
    def generate_embeddings(self, arguments):
        return self.embeddings_gen.get_embeddings(arguments=arguments)

    def cluster(self, method, arguments, embeddings=None):
        if embeddings is None:
            embeddings = self.generate_embeddings(arguments)
        if method == "kmeans":
            return cluster_kmeans(embeddings=embeddings, num_clusters=self.num_clusters)
        elif method == "entity":
            nlp = spacy.load("en_core_web_sm")
            return extract_specific_entities(arguments=arguments, nlp=nlp)
        elif method == "bertopic":
            topics, _ = cluster_by_topics(arguments=arguments)
            return topics
        elif method == "dbscan":
            return cluster_dbscan(embeddings=embeddings, eps=self.eps, min_samples=self.min_samples)
        elif method == "hierarchical":
            hierarchical = HierarchicalClustering()
            return hierarchical.fit(embeddings)
        else:
            raise ValueError(f"Not a clustering method: {method}")
        
    def group_labels(self, labels):
        grouped_labels = defaultdict(list)
        
        for i, label in enumerate(labels):
            grouped_labels[label].append(i)
        
        return grouped_labels



if __name__ == "__main__":
    arguments = [
            "Player X's goal in the final minute proves his incredible composure under pressure. It was the decisive moment of the match.",
            "Team Y's defensive strategy failed because they left gaps in the midfield. This allowed Player Z to exploit the space and score.",
            "The referee made a controversial decision, awarding a penalty to Team A that changed the outcome of the game.",
            "Player B's performance was outstanding as he scored a hat-trick, carrying his team to victory.",
            "If the weather conditions had been better, Team C might have performed more effectively in the match.",
            "Team D dominated possession but failed to convert their chances, highlighting their inefficiency in front of the goal."
        ]

    all_methods = ["bertopic", "dbscan"] # "kmeans", "entity",

    for method in all_methods:
        print(f"\nClustering using {method.upper()}:")
        clustering = Clustering(sentences=arguments, num_clusters=5, eps=0.5, min_samples=5)
        labels = clustering.cluster()
        print(f"Cluster Labels for {method.upper()}: {labels}")