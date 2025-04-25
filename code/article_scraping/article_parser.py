import pandas as pd
from single_article_scraping import Article_Scraper

class Article_Parser(Article_Scraper):
    def __init__(self, csv_file_name):
        self.csv_file_name = csv_file_name
        super().__init__()
        self.create_dataset_csv(self.read_article_data('football_articles.txt'))

    def read_article_data(self, file_path):
        article_data = []
        with open(file_path, 'r') as file:
            # do scrape article links independently and change here
            for line in file.readlines()[:]:
                print(line)
                title, url = line.strip().split(',http')
                article_data.append({'title': title, 'url': url})
        return article_data

    def create_dataset_csv(self, article_data):
        dataset = []
        for article in article_data:
            self.article_content = [] # needs to be reset for each new article
            article_topic = article['title']
            sentences = self.get_article_sentences("http" + article['url'])
            
            for sentence in sentences:
                #print(sentence)
                dataset.append({
                    'article_topic': article_topic,
                    'sentence': sentence
                })
        df = pd.DataFrame(dataset)
        df.to_csv(self.csv_file_name, index=False)

if __name__ == '__main__':
    Article_Parser("football_articles.csv")
