import json
import pandas as pd

if __name__ == "__main__":
    all_labels = {}

    json_files = ["article_labels2.json","article_labels3.json","article_labels4.json","article_labels5.json","article_labels6.json"]
    df = pd.read_csv("football_articles.csv")
    df2 = pd.read_csv("football_articles2.csv")
    combined_df = pd.concat([df, df2], ignore_index=True)

    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            for article_topic, labels in json_data.items():
                if article_topic not in all_labels:
                    cleaned_labels = [int(label.strip().split(")")[1]) for label in labels]
                    all_labels[article_topic] = cleaned_labels

    articles_missing_labels = []

    def assign_labels(row):
        article_topic = row['article_topic']
        sentence_idx = combined_df[combined_df['article_topic'] == article_topic].index.get_loc(row.name)
        
        if article_topic in all_labels and sentence_idx < len(all_labels[article_topic]):
            return all_labels[article_topic][sentence_idx]
        else:
            print(article_topic)
            articles_missing_labels.append(article_topic)
            return None
    
    missing_labels = set(articles_missing_labels)
    combined_df['label'] = combined_df.apply(assign_labels, axis=1)
    combined_df.to_csv("labeled_combined_football_articles.csv", index=False)