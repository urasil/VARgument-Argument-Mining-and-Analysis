import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from helpers.Nofeatures_Unfreezed import RoBERTaSentence
from helpers.DataProcessor import DataProcessor
from helpers.New_data_processor import FeatureEngineering

"""
No-features Extended RoBERTa model fine-tuning. This model is created as a benchmark to compare the effectiveness of the feature engineering methods
"""

processor = DataProcessor()
train_df, test_df, val_df = processor.get_train_test_val()
train_encodings, test_encodings, val_encodings = processor.get_encodings()
train_labels, test_labels, val_labels = processor.get_labels()

class ArgumentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_dataset = ArgumentDataset(train_encodings, train_labels)
test_dataset = ArgumentDataset(test_encodings, test_labels)
val_dataset = ArgumentDataset(val_encodings, val_labels)

model = RoBERTaSentence.from_pretrained("roberta-base", num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir="./no_features_result",
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
