from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from helpers.DataProcessor import DataProcessor, compute_metrics
from helpers.ArgumentDataset import ArgumentDataset
from helpers.FreezedRoberta import RoBERTaEnhancedFreezed
import optuna
import os

"""
Fine-tuning script for the RoBERTa model with freezing layers
"""

print("Data Processing...")

data_processor = DataProcessor()
train_encodings, test_encodings, val_encodings = data_processor.get_encodings()
train_df, test_df, val_df = data_processor.get_train_test_val()

train_dataset = ArgumentDataset(train_encodings, train_df['label'].values)
test_dataset = ArgumentDataset(test_encodings, test_df['label'].values)
val_dataset = ArgumentDataset(val_encodings, val_df['label'].values)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base", num_labels=2)
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.02, 0.1)
    warmup_steps = trial.suggest_int("warmup_steps", 0, 500)

    trial_output_dir = f"./results/trial_{trial.number}"

    training_args = TrainingArguments(
        output_dir=trial_output_dir,
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
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    metrics = trainer.evaluate()
    return metrics["eval_f1"]

print("Hyperparameter Tuning...")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)


def evaluate_best_model():
    best_trial = study.best_trial

    best_output_dir = f"./results/trial_{best_trial.number}"
    training_args = TrainingArguments(
        output_dir=best_output_dir,
        per_device_eval_batch_size=16,
        disable_tqdm=False,
        report_to="none"
    )

    best_model_path = os.path.join(best_output_dir, "checkpoint-best")
    model = RobertaForSequenceClassification.from_pretrained(best_model_path, num_labels=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    final_metrics = trainer.evaluate()
    print(f"Best model's performance: {final_metrics}")
    return final_metrics

print("Evaluating the best-performing model...")

best_metrics = evaluate_best_model()
