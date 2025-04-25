from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
from helpers.DataProcessor import DataProcessor, compute_metrics
from helpers.ArgumentDataset import ArgumentDataset
import torch
import optuna
import json

"""
Fine-tuning script for the original RoBERTa model 
"""

data_processor = DataProcessor()
train_encodings, test_encodings, val_encodings = data_processor.get_encodings()
train_df, test_df, val_df = data_processor.get_train_test_val()

train_dataset = ArgumentDataset(train_encodings, train_df['label'].values)
test_dataset = ArgumentDataset(test_encodings, test_df['label'].values)
val_dataset = ArgumentDataset(val_encodings, val_df['label'].values)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base", num_labels=2)
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

"""
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
    warmup_steps = trial.suggest_int("warmup_steps", 0, 500)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        load_best_model_at_end=True,
        save_total_limit=2,
        disable_tqdm=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics["eval_f1"]

#hyperparameter tuning with Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)  # Adjust n_trials based on resources

best_hyperparameters = study.best_params
with open("best_hyperparameters.json", "w") as f:
    json.dump(best_hyperparameters, f)
"""
best_hyperparameters = {'learning_rate': 1.8e-05, 'per_device_train_batch_size': 16, 'weight_decay': 0.01, 'warmup_steps': 270}

training_args = TrainingArguments(
    output_dir="./final_model",
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

final_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

final_trainer.train()

print("Evaluating on test set...")
metrics = final_trainer.evaluate(test_dataset)
print(metrics)