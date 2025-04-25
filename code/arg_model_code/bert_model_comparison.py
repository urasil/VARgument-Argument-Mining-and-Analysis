import os
import pandas as pd
import numpy as np
from transformers import RobertaForSequenceClassification, RobertaTokenizer, BertForSequenceClassification, DistilBertForSequenceClassification, BertTokenizer, DistilBertTokenizer
from helpers.DataProcessor import DataProcessor, compute_metrics
from helpers.ArgumentDataset import ArgumentDataset
from helpers.no_features_no_freezing_roberta import RoBERTaEnhancedNoFeatures, RoBERTaEnhancedNoFeaturesConfig
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import torch
from torch.utils.data import DataLoader

"""
Comparison of the BERT family models - RoBERTa crowned the best performing model
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_processor = DataProcessor()

train_df, test_df, val_df = data_processor.get_train_test_val()
train_labels, test_labels, val_labels = data_processor.get_labels()

pure_model = RobertaForSequenceClassification.from_pretrained("Pure_RoBERTa", num_labels=2).to(device)
pure_tokenizer = RobertaTokenizer.from_pretrained("Pure_RoBERTa", num_labels=2)

bert_model = BertForSequenceClassification.from_pretrained("bert_model", num_labels=2).to(device)
bert_tokenizer = BertTokenizer.from_pretrained("bert_model", num_labels=2)

dbert_model = DistilBertForSequenceClassification.from_pretrained("distilbert_model", num_labels=2).to(device)
dbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert_model", num_labels=2)

model_dict = {"roberta": pure_model, "bert": bert_model, "distilbert": dbert_model}
tokenizer_dict = {"roberta": pure_tokenizer, "bert": bert_tokenizer, "distilbert": dbert_tokenizer}

class ArgumentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, features=None):
        self.encodings = encodings
        self.labels = labels
        self.features = features
        self.index = np.arange(len(labels))  

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        if self.features is not None:
            item['features'] = torch.tensor(self.features[idx], dtype=torch.float32)
        item['index'] = self.index[idx]  
        return item

    def __len__(self):
        return len(self.labels)

results = {}
for model_name, model in model_dict.items():
    print("Inference with model:", model_name)
    model.eval()
    tokenizer = tokenizer_dict[model_name]
    test_encodings = tokenizer(list(test_df['sentence']), padding=True, truncation=True, max_length=256, return_tensors="pt")

    if model_name in ["freezing", "nonfreezing"]:
        test_dataset = ArgumentDataset(test_encodings, test_df['label'].values, test_features) # add test_features back for this to work
    else:
        test_dataset = ArgumentDataset(test_encodings, test_df['label'].values)

    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    predicted_labels = []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs['logits'], dim=-1).cpu().numpy()
        predicted_labels.extend(predictions)

    results[model_name] = predicted_labels

metrics_dict = {}
for model_name, predictions in results.items():
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average="weighted")
    recall = recall_score(test_labels, predictions, average="weighted")
    f1 = f1_score(test_labels, predictions, average="weighted")

    metrics_dict[model_name] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

for model_name, metrics in metrics_dict.items():
    print(f"Metrics for {model_name}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print("\n")

output_dir = "bert_performance_comparison"
os.makedirs(output_dir, exist_ok=True)

# Bar chart for metrics
metrics = ["accuracy", "precision", "recall", "f1"]
model_names = list(metrics_dict.keys())
x = np.arange(len(model_names)) 
width = 0.2  

fig, ax = plt.subplots(figsize=(12, 6))
for i, metric in enumerate(metrics):
    scores = [metrics_dict[model_name][metric.lower()] for model_name in model_names]
    ax.bar(x + i * width, scores, width, label=metric)

ax.set_xlabel("Models")
ax.set_ylabel("Scores")
ax.set_title("Bert Family Performance Comparison")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names)
ax.legend()
bar_chart_path = os.path.join(output_dir, "model_performance_comparison.png")
plt.savefig(bar_chart_path)
plt.close()
