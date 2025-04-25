from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from helpers.DataProcessor import DataProcessor, compute_metrics
from helpers.ArgumentDataset import ArgumentDataset
from helpers.FeatureEngineering import FeatureEngineering
import os
import argparse
import torch
from torch.utils.data import DataLoader
from helpers.FreezedRoberta import RoBERTaEnhancedFreezed, RoBERTaEnhancedFreezedConfig
from helpers.UnfreezedRoberta import RoBERTaEnhanced, RoBERTaEnhancedConfig

print("Data Processing...")

"""
This is a manual check for performance. Running this file will classify sentences as arguments or non-arguments.
Result is saved in a text file to manually check the classifications. Change model and tokenisers initialisation to use different models.
"""


data_processor = DataProcessor()
engineering = FeatureEngineering()
column_list = ['best', 'better', 'big', 'did', 'going', 'good', 'just', 'know', 'like', 'lot', 'make', 'need', 'new', 'play', 'really', 'right', 'said', 'start', 'think', 've', 'want', 'way', 'win', 'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'WDT', 'PREP', 'MODAL', 'NUM']

train_encodings, test_encodings, val_encodings = data_processor.get_encodings()
train_df, test_df, val_df = data_processor.get_train_test_val()

train_labels, test_labels, val_labels = data_processor.get_labels()

test_features = engineering.get_features(test_df)[column_list].values

test_dataset = ArgumentDataset(test_encodings, test_df['label'].values, test_features, test_df["sentence"].values)

parser = argparse.ArgumentParser(description="Evaluate model on argument classification.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
parser.add_argument("--output_file", type=str, default="classified_sentences.txt", help="Output file for classified sentences.")
args = parser.parse_args()

model = None

if args.checkpoint == "Pure_RoBERTa" or args.checkpoint == "curr-pureroberta-best":
    model = RobertaForSequenceClassification.from_pretrained(args.checkpoint, num_labels=2)
else:
    if args.checkpoint == "curr-freezing-best":
        config = RoBERTaEnhancedFreezedConfig(feature_dim=test_features.shape[1], num_labels=2)
        model = RoBERTaEnhancedFreezed(config)
    else:
        config = RoBERTaEnhancedConfig(feature_dim=test_features.shape[1], num_labels=2)
        model = RoBERTaEnhanced(config)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


test_dataloader = DataLoader(test_dataset, batch_size=16)

model.eval()
argument_sentences = []
non_argument_sentences = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        features = batch['features']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, features=features)
        predictions = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
        
        sentences = batch['sentence']  
        for i, label in enumerate(predictions):
            if label == 1:
                argument_sentences.append(sentences[i])
            else:
                non_argument_sentences.append(sentences[i])

with open(args.output_file, "w") as f:
    f.write("Arguments:\n")
    f.write("\n".join(argument_sentences))
    f.write("\n\nNon-Arguments:\n")
    f.write("\n".join(non_argument_sentences))

print(f"Classification results saved to {args.output_file}")

# python classify_sentences.py --checkpoint ./non_freezing_final/checkpoint-200 --output_file results.txt
