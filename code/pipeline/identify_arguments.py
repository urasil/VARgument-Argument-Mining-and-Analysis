from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import DataLoader
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from arg_model_code.helpers.ArgumentDataset import ArgumentDataset
from article_scraping.single_article_scraping import Article_Scraper


"""
Argument Identification with the Extended RoBERTa model - unfreezed
Identifies the arguments in raw text
"""

class ArgumentIdentification():

    def __init__(self, model_name):
        self.model_name = model_name
        self.scraper = Article_Scraper()

    def get_sentences(self, url):
        return self.scraper.get_article_sentences(url)
    
    def preprocess_sentence(self, sentences):
        res = []
        for sentence in sentences:
            sentence = sentence.lower().replace('\"', '')
            sentence = '"' + sentence + '"'
            res.append(sentence)
        return res

    def identify_arguments(self, url):
        model = RobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        sentences = self.get_sentences(url)
        processed_sentences = self.preprocess_sentence(sentences)
        encodings = tokenizer(processed_sentences, padding=True, truncation=True, return_tensors='pt')
        dataset = ArgumentDataset(encodings=encodings, sentences=sentences)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        model.eval()
        argument_sentences = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
                batch_sentences = batch['sentence']  
                
                # Clasification 
                for i, label in enumerate(predictions):
                    if label == 1: 
                        sentence = batch_sentences[i].replace('\\', '')
                        sentence = batch_sentences[i].replace('/', '')
                        argument_sentences.append(sentence)
        
        return argument_sentences

if __name__ == "__main__":
    model_name = "../Pure_RoBERTa"
    identifier = ArgumentIdentification(model_name)
    identified_arguments = identifier.identify_arguments("https://www.skysports.com/football/newcastle-united-vs-west-ham-united/report/505925")
    print("---IDENTIFIED ARGUMENTS---")
    for arg in identified_arguments:
        print(arg)
