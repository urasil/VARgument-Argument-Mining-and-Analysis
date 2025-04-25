import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import nltk
nltk.download('punkt')
from nltk import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')

#New_data_processor.py

"""
This file contains the new FeatureEngineering class. The name of the file is not a good representation for the file as it was used in the UCL servers for training before it was moved to local.
Due to VSC automatic reference updating breaking everything, the name of the file was not changed.
"""

class FeatureEngineering():
    def __init__(self):
        self.selected_words = [
            'best', 'better', 'big', 'did', 'going', 'good', 'just', 'know', 'like',
            'lot', 'make', 'need', 'new', 'play', 'really', 'right', 'said', 'start',
            'think', 've', 'want', 'way', 'win'
        ]
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', vocabulary=self.selected_words)
        self.scaler = MinMaxScaler()

        self.grouped_tags = {
            'NOUN': {'NN', 'NNS', 'NNP', 'NNPS'},
            'VERB': {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'},
            'ADJ': {'JJ', 'JJR', 'JJS'},
            'ADV': {'RB', 'RBR', 'RBS'},
            'PRON': {'PRP', 'PRP$', 'WP', 'WP$'},
            'WDT': {'WDT'},
            'PREP': {'IN', 'TO'},
            'MODAL': {'MD'},
            'NUM': {'CD'},
        }

    def pos_tag_distribution(self, text):
        tokens = word_tokenize(text)
        tags = [tag for _, tag in pos_tag(tokens)]
        grouped_counts = {group: 0 for group in self.grouped_tags.keys()}

        for tag in tags:
            for group, members in self.grouped_tags.items():
                if tag in members:
                    grouped_counts[group] += 1
                    break

        total_tokens = len(tokens) if len(tokens) > 0 else 1
        return {group: count / total_tokens for group, count in grouped_counts.items()}

    def get_features(self, df):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['sentence'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())
        pos_tags_df = df['sentence'].apply(self.pos_tag_distribution).apply(pd.Series).fillna(0)
        features = pd.concat([tfidf_df, pos_tags_df], axis=1)
        features = pd.DataFrame(self.scaler.fit_transform(features), columns=features.columns)
        df = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
        return df

