import pandas as pd
import numpy as np
import faiss
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from clustering.embeddings import EmbeddingsGenerator

"""
Building the FAISS index for the argument embeddings.
For fast similarity search.
"""

class ArgumentRetriever:
    def __init__(self, model_name, csv_file, embedding_type):
        self.generator = None
        if model_name == "sentence-transformers":
            self.generator = EmbeddingsGenerator()
        else:
            self.generator = EmbeddingsGenerator(model_name=model_name, pooling_type="cls-token")
        self.df = pd.read_csv(csv_file)
        self.embeddings = self._load_embeddings(embedding_type)
        self.index = self._build_faiss_index()
        self.tokenizer, self.model = self._load_roberta_model(model_name)
        print("Argument Retriever initialized.")

    def _load_embeddings(self, embedding_type):
        start = time.time()
        embeddings = None
        if embedding_type == "ft":
            self.df['ft_embeddings'] = self.df['ft_embeddings'].apply(lambda x: np.array(eval(x)))  
            embeddings = np.vstack(self.df['ft_embeddings'].to_numpy()).astype('float32')
        else:
            self.df['st_embeddings'] = self.df['st_embeddings'].apply(lambda x: np.array(eval(x)))
            embeddings = np.vstack(self.df['st_embeddings'].to_numpy()).astype('float32')
        print(f"Loaded {len(self.df)} embeddings in {time.time()-start:.2f} seconds.")
        return embeddings

    def _build_faiss_index(self):
        start = time.time()
        d = self.embeddings.shape[1]  
        index = faiss.IndexFlatL2(d)
        index.add(self.embeddings)
        print(f"Built FAISS index in {time.time()-start:.2f} seconds.")
        return index
    
    def _load_roberta_model(self, model_name):
        if model_name == "sentence-transformers":
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            return tokenizer, model
        else:
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = RobertaForSequenceClassification.from_pretrained(model_name)
            return tokenizer, model
    
    def retrieve_similar_sentences(self, user_sentence, top_k):
        input_embedding = self.generator.get_embeddings([user_sentence]).cpu().numpy()
        input_embedding = input_embedding.astype('float32')
        distances, indices = self.index.search(input_embedding, top_k)
        
        top_results = self.df.iloc[indices[0]].copy()
        top_results['distance'] = distances[0]
        return top_results, input_embedding

"""
PCA visualisation of the retrieved sentences and the input sentence (2D)
"""

class ArgumentVisualizer:
    @staticmethod
    def visualize(input_sentence, retrieved_sentences, input_embedding, all_embeddings):
        start = time.time()
        
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(all_embeddings)
        print(f"PCA performed in {time.time()-start:.2f} seconds.")
        
        plt.figure(figsize=(10, 6))
        plt.scatter(reduced_embeddings[1:, 0], reduced_embeddings[1:, 1], c='blue', label='Retrieved Sentences', alpha=0.7)
        plt.scatter(reduced_embeddings[0, 0], reduced_embeddings[0, 1], c='red', label='Input Sentence', s=100)
        
        for i, (x, y) in enumerate(reduced_embeddings[1:]):
            plt.text(x, y, f"{retrieved_sentences['argument_sentences'].iloc[i]}", fontsize=9, color='blue', ha='center', va='center')
        
        plt.text(reduced_embeddings[0, 0], reduced_embeddings[0, 1], f"{input_sentence}", fontsize=9, color='red', ha='center', va='center')
        plt.title("Visualization of Sentence Similarities")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

if __name__ == "__main__":
    
    model_name = "../Pure_RoBERTa"
    csv_file = "pipeline/knowledgebase.csv"
    
    df = pd.read_csv(csv_file)


    retriever = ArgumentRetriever(model_name, csv_file, "st")
    user_sentence = "Saka's play was absolutely brilliant because of his understanding of the game."
    
    top_sentences, input_embedding = retriever.retrieve_similar_sentences(user_sentence, top_k=5)
    retrieved_embeddings = retriever.embeddings[top_sentences.index]
    all_embeddings = np.vstack([input_embedding, retrieved_embeddings])
    
    ArgumentVisualizer.visualize(user_sentence, top_sentences, input_embedding, all_embeddings)
    print(top_sentences[['argument_sentences', 'distance']])
