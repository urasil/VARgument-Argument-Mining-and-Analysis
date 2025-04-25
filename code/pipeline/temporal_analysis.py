import pandas as pd
import numpy as np
import nltk
import spacy
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from empath import Empath

nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
lexicon = Empath()

"""
Temporal linguistic analysis for the extracted arguments.
"""

class KnowledgeBaseTemporalAnalysisFeatures:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.knowledge_base = pd.read_csv(csv_path)
        self.stopwords = set(stopwords.words("english"))

    def add_linguistic_features(self):
        if "argument_sentences" not in self.knowledge_base.columns:
            raise ValueError("The knowledge base must have a column named 'argument_sentences'.")

        feature_columns = [
            "lexical_richness", "negation_frequency", "syntax_depth", 
            "use_of_intensifiers", "question_frequency", "num_quotes", 
            "adj_frequency", "verb_frequency", "noun_frequency", "argument_length",
            "argument_complexity", "emotion_tone_shift", "formality", "domain_specific_term_percentage"
        ]

        for col in feature_columns:
            self.knowledge_base[col] = np.nan

        for index, row in tqdm(self.knowledge_base.iterrows(), total=len(self.knowledge_base)):
            sentence = row["argument_sentences"]
            doc = nlp(sentence)

            self.knowledge_base.at[index, "lexical_richness"] = self.calculate_lexical_richness(sentence)
            self.knowledge_base.at[index, "negation_frequency"] = self.calculate_negation_frequency(doc)
            self.knowledge_base.at[index, "syntax_depth"] = self.calculate_syntax_depth(doc)
            self.knowledge_base.at[index, "use_of_intensifiers"] = self.calculate_intensifiers(sentence)
            self.knowledge_base.at[index, "question_frequency"] = self.calculate_question_frequency(sentence)
            self.knowledge_base.at[index, "num_quotes"] = self.calculate_quote_frequency(sentence)
            self.knowledge_base.at[index, "argument_complexity"] = self.calculate_argument_complexity(doc)
            self.knowledge_base.at[index, "emotion_tone_shift"] = self.calculate_emotion_tone_shift(sentence)
            self.knowledge_base.at[index, "formality"] = self.calculate_formality(sentence)
            self.knowledge_base.at[index, "domain_specific_term_percentage"] = self.calculate_domain_specific_term_percentage(sentence)

            pos_freq = self.calculate_pos_frequency(doc)
            self.knowledge_base.at[index, "adj_frequency"] = pos_freq.get("ADJ", 0)
            self.knowledge_base.at[index, "verb_frequency"] = pos_freq.get("VERB", 0)
            self.knowledge_base.at[index, "noun_frequency"] = pos_freq.get("NOUN", 0)
            self.knowledge_base.at[index, "argument_length"] = len(sentence.split())

        self.knowledge_base.to_csv(self.csv_path, index=False)
        print(f"Linguistic features added and saved to {self.csv_path}")

    def calculate_lexical_richness(self, sentence):
        words = word_tokenize(sentence)
        unique_words = set(words) - self.stopwords
        return len(unique_words) / len(words) if words else 0

    def calculate_negation_frequency(self, doc):
        return sum(1 for token in doc if token.dep_ == "neg") / len(doc) if len(doc) > 0 else 0

    def calculate_syntax_depth(self, doc):
        return max([token.head.i - token.i for token in doc if token.head.i != token.i], default=0)

    def calculate_intensifiers(self, sentence):
        intensifiers = {"very", "extremely", "highly", "totally", "utterly"}
        words = word_tokenize(sentence)
        return sum(1 for word in words if word.lower() in intensifiers) / len(words) if words else 0

    def calculate_question_frequency(self, sentence):
        return 1 if "?" in sentence else 0

    def calculate_quote_frequency(self, sentence):
        return sentence.count('"') + sentence.count("'")

    def calculate_pos_frequency(self, doc):
        pos_counts = Counter(token.pos_ for token in doc)
        total_tokens = len(doc)
        return {pos: pos_counts.get(pos, 0) / total_tokens if total_tokens > 0 else 0 for pos in ["ADJ", "VERB", "NOUN"]}

    def calculate_argument_complexity(self, doc):
        return sum(1 for token in doc if token.dep_ in {"advcl", "ccomp", "xcomp", "acl"}) / len(doc) if len(doc) > 0 else 0

    def calculate_emotion_tone_shift(self, sentence):
        emotions = lexicon.analyze(sentence, categories=["joy", "anger", "fear", "sadness", "trust"], normalize=True)
        return max(emotions.values()) - min(emotions.values())

    def calculate_formality(self, sentence):
        words = word_tokenize(sentence)
        formal_words = {"thus", "therefore", "moreover", "hence", "furthermore", "whereas"}
        return sum(1 for word in words if word.lower() in formal_words) / len(words) if words else 0

    def calculate_domain_specific_term_percentage(self, sentence):
        domain_terms = {"goal", "penalty", "offside", "referee", "striker", "midfielder", "defender", "VAR", "free kick", "corner", "third", "half", "injury time", "extra time", "substitute", "tackle", "foul", "yellow card", "red card", "player", "coach", "title", "league", "championship", "club", "manager", "assist", "kick", "header", "overhead", "shot", "save", "goalkeeper", "defence", "attack", "possession", "counter attack", "cross", "pass", "dribble", "tactic", "formation", "press", "mark"}
        words = word_tokenize(sentence)
        return sum(1 for word in words if word.lower() in domain_terms) / len(words) if words else 0

if __name__ == "__main__":
    """
    Note that some intermediate knowledgebases are deleted as they were no longer required. To use this script, existing csv files must be used.
    """
    csv_path = "pipeline/new_knowledgebase_with_polarity.csv"
    updater = KnowledgeBaseTemporalAnalysisFeatures(csv_path)
    updater.add_linguistic_features()
    