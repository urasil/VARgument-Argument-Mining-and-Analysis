from transformers import AutoModel, AutoTokenizer, RobertaForSequenceClassification, RobertaTokenizer
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import numpy as np

"""
Each hidden layer refines the token representations based on the context learned from the input. These are represented as tensors of shape [batch_size, seq_len, hidden_dim]
batch_size: Number of input sequences in the batch
seq_len: Length of the tokenised input sequence
hidden_dim: Dimensionality of the token representations (768 for RoBERTa base)
The last layer's hidden states refer to the token embeddings from the final layer of the transformer. These embeddings capture the most refined and contextualised information about each token, as learned by the model
"""

"""
Generates embeddings for the arguments using different models (Extended RoBERTa, SentenceBERT) and using different pooling strategies (mean-pooling, cls-token)
"""

class EmbeddingsGenerator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None, pooling_type="mean-pooling"):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.pooling_type = pooling_type

        if "sentence-transformers" in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        else:  
            self.tokenizer = RobertaTokenizer.from_pretrained("../Pure_RoBERTa")
            self.model = RobertaForSequenceClassification.from_pretrained("../Pure_RoBERTa").to(self.device)

    def get_embeddings(self, arguments):
        inputs = self.tokenizer(arguments, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Unsupervised embedding generation using domain-agnostic sentence-transformers
        if "sentence-transformers" in self.model_name:
            last_hidden_state_representation = outputs.hidden_states[-1] # [batch_size, seq_len, hidden_dim]
            attention_mask = inputs["attention_mask"]
            sentence_embeddings = self.mean_pooling(last_hidden_state_representation, attention_mask)
        
        # Embedding generation using fine-tune RoBERTa model 
        else:
            # Pooling Strategy -> mean pooling (average of all token embeddings weighted by the attention mask for ignoring padded tokens)
            if not hasattr(self, "pooling_type") or self.pooling_type == "mean-pooling":
                last_hidden_state_representation = outputs.hidden_states[-1] 
                attention_mask = inputs["attention_mask"]
                sentence_embeddings = self.mean_pooling(last_hidden_state_representation, attention_mask)
            # Poolins Strategy -> cls token
            elif self.pooling_type == "cls-token":
                last_hidden_state_representation = outputs.hidden_states[-1]  
                sentence_embeddings = last_hidden_state_representation[:, 0, :] # cls token
            else:
                raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
        
        return sentence_embeddings

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
       # Performs mean pooling on token embeddings - taking the average of token representations weighted with the attention mask (paddings avoided with weight 0) and returns: Pooled embeddings of shape [batch_size, hidden_dim]

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9) # min defined so we avoid division by 0

def compute_similarity(embeddings1, embeddings2):
    return [cosine_similarity([e1], [e2])[0][0] for e1, e2 in zip(embeddings1, embeddings2)]




if __name__ == "__main__":
    arguments = [
        "Player X's goal in the final minute proves his incredible composure under pressure. It was the decisive moment of the match.",
        "Team Y's defensive strategy failed because they left gaps in the midfield. This allowed Player Z to exploit the space and score.",
        "The referee made a controversial decision, awarding a penalty to Team A that changed the outcome of the game.",
        "Player B's performance was outstanding as he scored a hat-trick, carrying his team to victory.",
        "If the weather conditions had been better, Team C might have performed more effectively in the match.",
        "Team D dominated possession but failed to convert their chances, highlighting their inefficiency in front of the goal."
    ]
    path = "arg_model_code//fixed_gemini_labelled_articles.csv"
    df = pd.read_csv(path)
    


    embedder_st = EmbeddingsGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings_st = embedder_st.get_embeddings(arguments)
    print("Sentence-Transformer Embeddings:", embeddings_st.shape)

    embedder_st.visualize_embeddings(embeddings=embeddings_st)

    embedder_ft = EmbeddingsGenerator(model_name="../Pure_RoBERTa")
    embeddings_ft = embedder_ft.get_embeddings(arguments)
    print("Fine-Tuned RoBERTa Embeddings:", embeddings_ft.shape)

    embedder_ft.visualize_embeddings(embeddings=embeddings_ft)
    
    """
    print("Loading models...")

    roberta_generator = EmbeddingsGenerator(model_name="../Pure_RoBERTa", pooling_type="mean-pooling")
    sbert_generator = EmbeddingsGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Loading dataset...")
    df = pd.read_csv("clustering\\sts-test.csv",sep="\t",header=None,on_bad_lines="skip" )

    if df.shape[1] >= 7:
        df = df.iloc[:, :7] 
    df.columns = ["metadata1", "metadata2", "metadata3", "metadata4", "similarity_score", "sentence1", "sentence2"]

    df["similarity_score"] = pd.to_numeric(df["similarity_score"], errors="coerce")

    df = df.dropna(subset=["similarity_score"])
    df["sentence1"] = df["sentence1"].astype(str)
    df["sentence2"] = df["sentence2"].astype(str)
    sentences1 = df["sentence1"].tolist()
    sentences2 = df["sentence2"].tolist()
    gold_scores = df["similarity_score"].tolist()
    print("Computing embeddings...")

    roberta_embeddings1 = np.array(roberta_generator.get_embeddings(sentences1))
    print("roberta_embeddings1 done")
    roberta_embeddings2 = np.array(roberta_generator.get_embeddings(sentences2))
    print("roberta_embeddings2 done")
    sbert_embeddings1 = np.array(sbert_generator.get_embeddings(sentences1))
    print("sbert_embeddings1 done")
    sbert_embeddings2 = np.array(sbert_generator.get_embeddings(sentences2))
    print("sbert_embeddings2 done")
    
    print("Computing cosine similarities...")
    roberta_similarities = compute_similarity(roberta_embeddings1, roberta_embeddings2)
    sbert_similarities = compute_similarity(sbert_embeddings1, sbert_embeddings2)

    roberta_corr, _ = spearmanr(roberta_similarities, gold_scores)
    sbert_corr, _ = spearmanr(sbert_similarities, gold_scores)

    print(f"Spearman Correlation (Fine-Tuned RoBERTa): {roberta_corr:.4f}")
    print(f"Spearman Correlation (Sentence-BERT): {sbert_corr:.4f}")

    print("Evaluation complete!")
    """