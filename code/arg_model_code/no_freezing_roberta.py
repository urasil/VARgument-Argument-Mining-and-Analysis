from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from helpers.DataProcessor import DataProcessor, compute_metrics
from helpers.ArgumentDataset import ArgumentDataset
from helpers.UnfreezedRoberta import RoBERTaEnhanced
from helpers.FeatureEngineering import FeatureEngineering
import os

"""
Old script for fine-tuning the RoBERTa Extended model
"""

print("Data Processing...")

data_processor = DataProcessor()
engineering = FeatureEngineering()
column_list = ['best', 'better', 'big', 'did', 'going', 'good', 'just', 'know', 'like', 'lot', 'make', 'need', 'new', 'play', 'really', 'right', 'said', 'start', 'think', 've', 'want', 'way', 'win', 'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'WDT', 'PREP', 'MODAL', 'NUM']

train_encodings, test_encodings, val_encodings = data_processor.get_encodings()
train_df, test_df, val_df = data_processor.get_train_test_val()
train_labels, test_labels, val_labels = data_processor.get_labels()

train_features = engineering.get_features(train_df)[column_list].values
test_features = engineering.get_features(test_df)[column_list].values
val_features = engineering.get_features(val_df)[column_list].values

train_dataset = ArgumentDataset(train_encodings, train_df['label'].values, train_features)
test_dataset = ArgumentDataset(test_encodings, test_df['label'].values, test_features)
val_dataset = ArgumentDataset(val_encodings, val_df['label'].values, val_features)


tokenizer = RobertaTokenizer.from_pretrained("roberta-base", num_labels=2)
model = RoBERTaEnhanced(
    feature_dim=train_features.shape[1], 
    num_labels=2,
    num_rnn_layers=1,
    num_fine_tuning_layers=3
)

best_hyper = {'learning_rate': 1.8e-05, 'per_device_train_batch_size': 16, 'weight_decay': 0.02, 'warmup_steps': 200}

training_args = TrainingArguments(
    output_dir="./non_freezing_final",               
    evaluation_strategy="steps",                 
    eval_steps=100,                              
    save_strategy="steps",                        
    save_steps=100,                               
    learning_rate=best_hyper["learning_rate"],   
    per_device_train_batch_size=best_hyper["per_device_train_batch_size"],
    per_device_eval_batch_size=16,
    num_train_epochs=2,                            
    weight_decay=best_hyper["weight_decay"],
    warmup_steps=best_hyper["warmup_steps"],
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] 
)

trainer.train()

test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test Results:", test_results)
