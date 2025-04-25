import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from helpers.ArgumentDataset import ArgumentDataset
from helpers.Updated_Freezed_Roberta import RoBERTaEnhancedFreezed
from helpers.Updated_Unfreezed_Roberta import RoBERTaEnhancedUnfreezed
from helpers.Nofeatures_Unfreezed import RoBERTaSentence
from helpers.New_data_processor import FeatureEngineering
from helpers.DataProcessor import DataProcessor

"""
Performance visualisations for the various RoBERTa architectures
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading and preprocessing data...")
processor = DataProcessor()
train_df, test_df, val_df = processor.get_train_test_val()
train_encodings, test_encodings, val_encodings = processor.get_encodings()
train_labels, test_labels, val_labels = processor.get_labels()

print("Performing feature engineering on test set...")
feature_engineer = FeatureEngineering()
test_df = feature_engineer.get_features(test_df)

def convert_features(df):
    feature_cols = [col for col in df.columns if col not in ['sentence', 'label', 'article_topic']]
    return torch.tensor(df[feature_cols].values, dtype=torch.float32)

test_features = convert_features(test_df).to(device)

print("Loading models...")
pure_model = RobertaForSequenceClassification.from_pretrained("Pure_RoBERTa", num_labels=2).to(device)
pure_tokenizer = RobertaTokenizer.from_pretrained("Pure_RoBERTa")
freezing_model = RoBERTaEnhancedFreezed.from_pretrained("new_best_freezed", feature_dim=test_features.shape[1], num_labels=2).to(device)
freezing_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
nonfreezing_model = RoBERTaEnhancedUnfreezed.from_pretrained("new_best_unfreezed", feature_dim=test_features.shape[1], num_labels=2).to(device)
nonfreezing_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
nofeatures_model = RoBERTaSentence.from_pretrained("nofeatures_best", num_labels=2).to(device)
nofeatures_tokenizer = RobertaTokenizer.from_pretrained("roberta-base", num_labels=2)


model_dict = {"pure": pure_model, "freezing": freezing_model, "nonfreezing": nonfreezing_model, "nofeatures":nofeatures_model}
tokenizer_dict = {"pure": pure_tokenizer, "freezing": freezing_tokenizer, "nonfreezing": nonfreezing_tokenizer, "nofeatures":nofeatures_tokenizer}

print("Running inference on test set...")
results = {}
for model_name, model in model_dict.items():
    print(f"Inference with model: {model_name}")
    model.eval()
    tokenizer = tokenizer_dict[model_name]
    test_encodings = tokenizer(list(test_df['sentence']), padding=True, truncation=True, max_length=256, return_tensors="pt")
    
    test_encodings = {k: v.to(device) for k, v in test_encodings.items()}
    test_labels_tensor = torch.tensor(test_df['label'].values).to(device)

    if model_name in ["freezing", "nonfreezing"]:
        test_dataset = ArgumentDataset(test_encodings, test_labels_tensor, test_features)
    else:
        test_dataset = ArgumentDataset(test_encodings, test_labels_tensor)

    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    predicted_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            if model_name in ["freezing", "nonfreezing"]:
                batch_features = batch['features'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, features=batch_features)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            predictions = torch.argmax(outputs['logits'], dim=-1).cpu().numpy()
            predicted_labels.extend(predictions)

    results[model_name] = predicted_labels

print("Computing performance metrics...")
metrics_dict = {}
for model_name, predictions in results.items():
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average="weighted")
    recall = recall_score(test_labels, predictions, average="weighted")
    f1 = f1_score(test_labels, predictions, average="weighted")

    metrics_dict[model_name] = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

print("\nModel Performance Metrics:")
for model_name, metrics in metrics_dict.items():
    print(f"Metrics for {model_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print("-" * 40)

output_dir = "new_performance_visualizations"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving visualizations in {output_dir}...")
metrics = ["accuracy", "precision", "recall", "f1"]
model_names = list(metrics_dict.keys())
x = np.arange(len(model_names))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
for i, metric in enumerate(metrics):
    scores = [metrics_dict[model_name][metric] for model_name in model_names]
    ax.bar(x + i * width, scores, width, label=metric)

ax.set_xlabel("Models")
ax.set_ylabel("Scores")
ax.set_title("Model Performance Comparison")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names)
ax.legend()
plt.savefig(os.path.join(output_dir, "model_performance_comparison.png"))
plt.close()

#Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
for i, model_name in enumerate(model_names):
    cm = confusion_matrix(test_labels, results[model_name])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i], cbar=False)
    axes[i].set_title(f"Confusion Matrix - {model_name}")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("True")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrices.png"))
plt.close()

# ROC Curve
plt.figure(figsize=(10, 8))
for model_name, predictions in results.items():
    fpr, tpr, _ = roc_curve(test_labels, predictions)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig(os.path.join(output_dir, "roc_curve.png"))
plt.close()

# Precision-Recall Curve
plt.figure(figsize=(10, 8))
for model_name, predictions in results.items():
    precision, recall, _ = precision_recall_curve(test_labels, predictions)
    plt.plot(recall, precision, label=f"{model_name}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
plt.close()

print(f"All visualizations saved in {output_dir}.")

