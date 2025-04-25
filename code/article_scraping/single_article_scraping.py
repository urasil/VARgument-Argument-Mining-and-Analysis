from bs4 import BeautifulSoup
import requests
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

class Article_Scraper:

    def __init__(self):
        self.article_topic = None
        self.article_content = []
    
    def fetch_article(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            return BeautifulSoup(response.content, 'html.parser')
        else:
            print(f"Failed to fetch the article. {response.status_code}")
            return None

    def extract_paragraphs(self, soup):
        skip_first = True
        for p in soup.find_all('p'):
            if p.has_attr('class') or p.find('strong') or p.find('em'):
                continue  
            text = p.get_text(strip=True)
            if skip_first:
                self.article_topic = text
                skip_first = False
            elif len(text) > 3 and text[0] != '\n':
                self.article_content.append(text)

    def tokenize_sentences(self):
        split_sentences = []
        for text in self.article_content:
            new_text = str(text).replace('/', '')
            new_text = new_text.replace('\\', '')
            sentences = sent_tokenize(new_text)
            split_sentences.extend(sentences)
        return split_sentences

    def get_article_sentences(self, url):
        soup = self.fetch_article(url)
        if soup:
            self.extract_paragraphs(soup)
            return self.tokenize_sentences()
        return []

if __name__ == '__main__':

    scraper = Article_Scraper()
    sentences = scraper.get_article_sentences("https://www.skysports.com/football/news/11661/13228680/bukayo-saka-arsenals-mikel-arteta-hails-unbelievable-player-for-taking-another-step-up-as-arsenal-beat-southampton")

    for sentence in sentences:
        print(sentence)
