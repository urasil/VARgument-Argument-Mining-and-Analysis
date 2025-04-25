import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
from nltk import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger_eng')
nltk.download("punkt")

class FeatureEngineering():
    def __init__(self):

        selected_words = [
            'best', 'better', 'big', 'did', 'going', 'good', 'just', 'know', 'like',
            'lot', 'make', 'need', 'new', 'play', 'really', 'right', 'said', 'start',
            'think', 've', 'want', 'way', 'win'
        ]

        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', vocabulary=selected_words)
        

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
        tags = [tag for word, tag in pos_tag(tokens)]
        grouped_counts = {group: 0 for group in self.grouped_tags.keys()}
        
        for tag in tags:
            for group, members in self.grouped_tags.items():
                if tag in members:
                    grouped_counts[group] += 1
                    break
        
            
        tag_counts = pd.Series(grouped_counts).div(len(tags))
        return tag_counts


    def get_features(self, df):
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['sentence'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())    
        df = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        pos_tags_df = df['sentence'].apply(self.pos_tag_distribution).fillna(0)
        df = pd.concat([df.reset_index(drop=True), pos_tags_df.reset_index(drop=True)], axis=1)
        return df
    
if __name__ == "__main__":
    engineer = FeatureEngineering()
    df = engineer.get_features(pd.read_csv("../fixed_gemini_labelled_articles.csv"))
    print(df["best"].value_counts())
