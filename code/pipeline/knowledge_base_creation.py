from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from arg_model_code.helpers.ArgumentDataset import ArgumentDataset
#from article_scraping.single_article_scraping import Article_Scraper
#from clustering.embeddings import EmbeddingsGenerator



"""
Creating the knowledgebase for VARgument. 
Knowledgebase is made up of the argument sentences and their analysis.
"""


class CreateKnowledgeBase:
    def __init__(self, model_path, tokenizer_name, data_file, output_file):
        self.model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.data_file = data_file
        self.output_file = output_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_data(self):
        df = pd.read_csv(self.data_file)
        sentences = df['sentence'].tolist()
        tokenized_sentences = self.tokenizer(
            sentences, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        dataset = ArgumentDataset(encodings=tokenized_sentences, sentences=sentences)
        return DataLoader(dataset, batch_size=16, shuffle=False)

    def extract_arguments(self):
        dataloader = self.load_data()
        argument_sentences = []

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()

                sentences = batch['sentence']
                for i, label in enumerate(predictions):
                    if label == 1:
                        argument_sentences.append(sentences[i])
        
        return argument_sentences

    def save_results(self, argument_sentences):
        output_df = pd.DataFrame({'argument_sentences': argument_sentences})
        output_df.to_csv(self.output_file, index=False)

    def create_knowledge_base(self):
        print("Loading data...")
        argument_sentences = self.extract_arguments()
        print(f"Extracted {len(argument_sentences)} argument sentences.")
        print(f"Saving results to {self.output_file}...")
        self.save_results(argument_sentences)

class ArgumentEmbeddingSaver:
    def __init__(self, arguments_file, output_file):
        self.arguments_file = arguments_file
        self.output_file = output_file
        self.generator_ft = EmbeddingsGenerator(model_name="Pure_RoBERTa", pooling_type="cls-token")
        self.generator_st = EmbeddingsGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2", pooling_type="cls-token")

    def save_embeddings(self):
        df = pd.read_csv(self.arguments_file)
        arguments = df['argument_sentences'].tolist()

        print("Generating embeddings...")
        embeddings_ft = self.generator_ft.get_embeddings(arguments)
        embeddings_st = self.generator_st.get_embeddings(arguments)

        embeddings_list = embeddings_ft.cpu().numpy().tolist()
        embeddings_st_list = embeddings_st.cpu().numpy().tolist()

        df['ft_cls_embeddings'] = embeddings_list
        df['st_cls_embeddings'] = embeddings_st_list

        df.to_csv(self.output_file, index=False)
        print("Embeddings saved successfully!")
    
    def visualize_embeddings(self, sentence_input, method="PCA", embedding_generator="ft", k=3):
        # Visualize the k closest sentences to the input sentence using PCA or t-SNE for dimensionality reduction to 2D -> "PCA" or "t-SNE"

        df = pd.read_csv(self.embeddings_file)

        sentences = df['argument_sentences'].tolist()
        if embedding_generator == "ft":
            embeddings_list = df['ft_embeddings'].tolist()
            embeddings_np = np.array([np.array(embedding) for embedding in embeddings_list])
            input_embedding = self.generate_embedding(sentence_input, embedding_generator)
        
        elif embedding_generator == "st":
            embeddings_list = df['st_embeddings'].tolist()
            embeddings_np = np.array([np.array(embedding) for embedding in embeddings_list])
            input_embedding = self.generate_embedding(sentence_input)

        similarities = cosine_similarity(input_embedding.reshape(1, -1), embeddings_np)
        closest_indices = np.argsort(similarities[0])[-k:][::-1] 

        # get closest sentences embeddings
        closest_sentences = [sentences[i] for i in closest_indices]
        closest_embeddings = embeddings_np[closest_indices]

        if method == "PCA":
            reduced_embeddings = PCA(n_components=2).fit_transform(closest_embeddings)
        elif method == "t-SNE":
            reduced_embeddings = TSNE(n_components=2, random_state=31).fit_transform(closest_embeddings)
        else:
            raise ValueError("Unsupported method. 'PCA' or 't-SNE'.")

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', alpha=0.7)

        for i, sentence in enumerate(closest_sentences):
            plt.annotate(
                sentence,
                (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                fontsize=8,
                alpha=0.75,
                ha='center',  
            )

        plt.title(f"Embedding Space of Closest Sentences ({method})")
        plt.grid(alpha=0.3)
        plt.show()

        print("Closest sentences to your input:")
        for idx in closest_indices:
            print(f"Sentence: {sentences[idx]} | Similarity: {similarities[0][idx]:.4f}")

    def generate_embedding(self, sentence, embedding_generator="ft"):
        generator = self.generator_ft if embedding_generator == "ft" else self.generator_st
        embedding = generator.get_embeddings([sentence])
        return embedding.cpu().numpy().flatten() 


if __name__ == "__main__":
    # model_path = "../Pure_RoBERTa"
    # tokenizer_name = "roberta-base"
    # data_file = "arg_model_code\\fixed_gemini_labelled_articles.csv"
    # output_file = "pipeline\\argument_sentences_output.csv"
    # extractor = CreateKnowledgeBase(model_path, tokenizer_name, data_file, output_file)
    # extractor.run_pipeline()

    """
    Note that some intermediate knowledgebases are deleted as they were no longer required. The use this script, existing csv files must be used.
    """
    arguments_file = "pipeline/new_knowledgebase_with_polarity.csv" # only used for creating the final_knowledgebase
    output_file = "pipeline/final_knowledgebase.csv"
    saver = ArgumentEmbeddingSaver(arguments_file, output_file)
    saver.save_embeddings()