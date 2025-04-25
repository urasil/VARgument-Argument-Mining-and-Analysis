from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, BertTokenizer, DistilBertTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
import torch
from helpers.DataProcessor import DataProcessor, compute_metrics
from helpers.ArgumentDataset import ArgumentDataset

"""
Fine-tuning script for the BERT and DistilBERT models
"""

data_processor = DataProcessor()
train_df, test_df, val_df = data_processor.get_train_test_val()

best_hyperparameters = {'learning_rate': 1.8e-05, 'per_device_train_batch_size': 16, 'weight_decay': 0.01, 'warmup_steps': 270}
def train_model(model_name, model_class, tokenizer_class, save_path):
    tokenizer = tokenizer_class.from_pretrained(model_name, num_labels=2)
    model = model_class.from_pretrained(model_name, num_labels=2)
    train_encodings = tokenizer(list(train_df['sentence']), padding=True, truncation=True, max_length=256, return_tensors="pt")
    test_encodings = tokenizer(list(test_df['sentence']), padding=True, truncation=True, max_length=256, return_tensors="pt")
    val_encodings = tokenizer(list(val_df['sentence']), padding=True, truncation=True, max_length=256, return_tensors="pt")

    train_dataset = ArgumentDataset(train_encodings, train_df['label'].values)
    test_dataset = ArgumentDataset(test_encodings, test_df['label'].values)
    val_dataset = ArgumentDataset(val_encodings, val_df['label'].values)
    
    training_args = TrainingArguments(
        output_dir=save_path,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=best_hyperparameters['learning_rate'],
        per_device_train_batch_size=best_hyperparameters['per_device_train_batch_size'],
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=best_hyperparameters['weight_decay'],
        warmup_steps=best_hyperparameters['warmup_steps'],
        load_best_model_at_end=True,
        save_total_limit=2
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    trainer.train()
    
    print(f"Evaluating {model_name} on test set...")
    metrics = trainer.evaluate(test_dataset)
    print(metrics)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

train_model("distilbert-base-uncased", DistilBertForSequenceClassification, DistilBertTokenizer, "./distilbert_model")
train_model("bert-base-uncased", BertForSequenceClassification, BertTokenizer, "./bert_model")

