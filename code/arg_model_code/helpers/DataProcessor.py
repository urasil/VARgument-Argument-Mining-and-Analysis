import pandas as pd
from transformers import RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit

class DataProcessor():

    def __init__(self):
        initial = pd.read_csv('fixed_gemini_labelled_articles.csv')
        initial['sentence'] = initial['sentence'].str.lower()
        initial['label'] = initial['label'].apply(lambda x: str(x).replace(",", ""))
        initial['label'] = initial['label'].apply(lambda x: int(x))

        args_df = initial[initial['label'] == 1]
        nonargs_df = initial[initial['label'] == 0]

        min_count = min(len(args_df), len(nonargs_df))
        sampled_args_df = args_df.sample(n=min_count, random_state=31)
        sampled_nonargs_df = nonargs_df.sample(n=min_count, random_state=31)
        self.train_df = pd.concat([sampled_args_df, sampled_nonargs_df]).sample(frac=1, random_state=31).reset_index(drop=True)
        test_df = pd.read_csv('validation_sentences.csv')
        test_df['sentence'] = test_df['sentence'].str.lower()
        test_df['label'] = test_df['label'].apply(lambda x: str(x).replace(",", ""))
        test_df['label'] = test_df['label'].apply(lambda x: int(x))

        splitter = GroupShuffleSplit(test_size=0.5, random_state=31)
        test_idx, val_idx = next(splitter.split(test_df, groups=test_df['article_topic']))

        self.val_df = test_df.iloc[val_idx].reset_index(drop=True)
        self.test_df = test_df.iloc[test_idx].reset_index(drop=True)


    def get_train_test_val(self):
        return self.train_df, self.test_df, self.val_df
    

    def get_encodings(self):
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        train_encodings = tokenizer(list(self.train_df['sentence']), padding=True, truncation=True, max_length=256, return_tensors="pt")
        test_encodings = tokenizer(list(self.test_df['sentence']), padding=True, truncation=True, max_length=256, return_tensors="pt")
        val_encodings = tokenizer(list(self.val_df['sentence']), padding=True, truncation=True, max_length=256, return_tensors="pt")
        return train_encodings, test_encodings, val_encodings

    
    def get_labels(self):
        return self.train_df["label"].values, self.test_df["label"].values, self.val_df["label"].values


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

if __name__ == "__main__":
    processor = DataProcessor()
    train, test, val = processor.get_train_test_val()
    print(test.head())
    print(val.head())
    print('test len: ', len(test))
    print('val len: ', len(val))
