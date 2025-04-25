import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from helpers.Updated_Freezed_Roberta import RoBERTaEnhancedFreezed
from helpers.DataProcessor import DataProcessor
from helpers.New_data_processor import FeatureEngineering

"""
Fine-tuning script for the RoBERTa model with freezing layers - updated to use the new feature engineering (projection of featuers using a linear layer) and data processing methods
"""

processor = DataProcessor()
train_df, test_df, val_df = processor.get_train_test_val()
train_encodings, test_encodings, val_encodings = processor.get_encodings()
train_labels, test_labels, val_labels = processor.get_labels()

feature_engineer = FeatureEngineering()
train_df = feature_engineer.get_features(train_df)
test_df = feature_engineer.get_features(test_df)
val_df = feature_engineer.get_features(val_df)

#convert features to tensors
def convert_features(df):
    feature_cols = [col for col in df.columns if col not in ['sentence', 'label', 'article_topic']]
    return torch.tensor(df[feature_cols].values, dtype=torch.float32)

train_features = convert_features(train_df)
test_features = convert_features(test_df)
val_features = convert_features(val_df)

class ArgumentDataset(Dataset):
    def __init__(self, encodings, labels, features):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.features = features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        item["features"] = self.features[idx]
        return item

train_dataset = ArgumentDataset(train_encodings, train_labels, train_features)
test_dataset = ArgumentDataset(test_encodings, test_labels, test_features)
val_dataset = ArgumentDataset(val_encodings, val_labels, val_features)

model = RoBERTaEnhancedFreezed.from_pretrained("roberta-base", feature_dim=train_features.shape[1], num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

eval_results = trainer.evaluate(test_dataset)
print("Test Results:", eval_results)

