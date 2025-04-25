import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import numpy as np

"""
Sentiment analysis for the extracted arguments. Uses the cardiffnlp model pre-trained on a large Twitter dataset.
Three categories: Positive, Negative, and Neutral.
"""

def sentiment_analysis():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def analyze_sentiment(sentence):
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
        sentiments = ["Negative", "Neutral", "Positive"]
        sentiment = sentiments[np.argmax(scores)]
        return sentiment

    csv_file = "pipeline/knowledgebase.csv"  
    df = pd.read_csv(csv_file)

    positive_col = []
    negative_col = []
    neutral_col = []

    start = time.time()
    print("Analyzing sentiment...")
    for sentence in df["argument_sentences"]:
        sentiment = analyze_sentiment(sentence)
        if sentiment == "Positive":
            positive_col.append(1)
            negative_col.append(0)
            neutral_col.append(0)
        elif sentiment == "Negative":
            positive_col.append(0)
            negative_col.append(1)
            neutral_col.append(0)
        else:
            positive_col.append(0)
            negative_col.append(0)
            neutral_col.append(1)

    print(f"Sentiment analysis completed in {time.time()-start:.2f} seconds.")
    df["Positive"] = positive_col
    df["Negative"] = negative_col
    df["Neutral"] = neutral_col

    output_file = "pipeline/new_knowledgebase.csv"  
    df.to_csv(output_file, index=False)

    print(f"Sentiment analysis done")

if __name__ == "__main__":
    """
    Note that some intermediate knowledgebases are deleted as they were no longer required. To use this script, existing csv files must be used.
    """
    df = pd.read_csv("pipeline/new_knowledgebase.csv")
    print("Negative Sentences:")
    print(df[df["Negative"] == 1]["argument_sentences"].head())
    print("Neutral Sentences:")
    print(df[df["Neutral"] == 1]["argument_sentences"].head())
    print("Positive Sentences:")
    print(df[df["Positive"] == 1]["argument_sentences"].head())
    