import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import numpy as np

"""
Polarity analysis for the extracted argumnets. Scraped in the dissertation.
"""

def polarity_analysis():
    model_name = "roberta-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def map_to_polarity(label):
        if label == "CONTRADICTION":
            return "Attacking"
        elif label == "ENTAILMENT":
            return "Supporting"
        else:
            return "Neutral"

    def classify_polarity(claim, sentence):
        inputs = tokenizer(
            claim,
            sentence,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        scores = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
        label_id = np.argmax(scores)
        label = model.config.id2label[label_id] # fixes it ?
        
        polarity = map_to_polarity(label)
        return polarity

    csv_file = "pipeline/new_knowledgebase.csv"  
    df = pd.read_csv(csv_file)

    supporting_col = []
    attacking_col = []
    neutral_col = []

    claim = "Saka is a world-class football player."

    start = time.time()
    print("Analyzing polarity...")

    for argument in df["argument_sentences"]:
        polarity = classify_polarity(claim, argument)
        if polarity == "Supporting":
            supporting_col.append(1)
            attacking_col.append(0)
            neutral_col.append(0)
        elif polarity == "Attacking":
            supporting_col.append(0)
            attacking_col.append(1)
            neutral_col.append(0)
        else:
            supporting_col.append(0)
            attacking_col.append(0)
            neutral_col.append(1)

    print(f"Polarity analysis completed in {time.time()-start:.2f} seconds.")

    df["Supporting"] = supporting_col
    df["Attacking"] = attacking_col
    df["No Polarity"] = neutral_col

    output_file = "pipeline/new_knowledgebase_with_polarity.csv"
    df.to_csv(output_file, index=False)

    print(f"Polarity analysis done. Updated file saved as {output_file}.")

if __name__ == "__main__":
    """
    Note that some intermediate knowledgebases are deleted as they were no longer required. To use this script, existing csv files must be used.
    """

    df = pd.read_csv("pipeline/new_knowledgebase_with_polarity.csv")
    print("Attacking Sentences:")
    print(df[df["Attacking"] == 1]["argument_sentences"].head())
    print("Neutral Sentences:")
    print(df[df["No Polarity"] == 1]["argument_sentences"].head())
    print("Supporting Sentences:")
    print(df[df["Supporting"] == 1]["argument_sentences"].head())