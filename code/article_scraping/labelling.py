# GPT GPT GPT GPT
import pandas as pd
import os
from dotenv import load_dotenv
import openai
import json
import google.generativeai as genai
import time

class Labelling:
    
    def __init__(self):
        
        self.df = pd.read_csv("football_articles.csv")
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(
            model_name = "gemini-1.5-flash",
            system_instruction = """
                        You are an argument identification model that decides whether a sentence is an argument, or not an argument. You will label each argument with 1 and each non-argument with 0.
                        Note: purely factual sentences with no claim or premise are not arguments. Remember for a sentence to be considered an argument, it MUST have both a claim and a premise. Do not mark sentences as arguments liberally,
                        ensure that they have a claim and a premise! Consider each sentence independently and decide carefully!

                        The following is the format your response MUST follow: Index)) Sentence * Label^@
                            
                        Example Labelling with explanations - do not include explanations in your response:
                        '
                        1)) Saka has scored 2 goals in today's match against Manchester City. * 0^@ (Non-argument: purely factual)
                        2)) Saka was clearly the best player on the pitch today because of his footwork. * 1^@ (Argument: claim: saka was the best player, premise: his footwork)
                        3)) He said "Manchester City did not perform the best in yesterday's game largely due to weather." * 1^@ (Argument: claim: Manchester City did not perform, premise: weather)
                        4)) Manchester United confirmed Anthony will play for them this season. * 0^@ (Non-argument: purely factual)
                        5)) It was a terrible game * 0^@ (Non-argument: just a claim, no premise)
                        '
                        """
        )

    def test_gemini(self):
        response = self.model.generate_content("Write a story about a magic backpack.")
        print(response.text)

    def parse_sentences(self):
        print("Parsing sentences...")
        for _, row in self.df.iterrows():
            row["sentence"] = row["sentence"].strip("\"")
    
    def get_all_sentences(self):
        print("Getting sentences...")
        sentences = [] # 2d array of sentences, each inner array represents all the sentences in one article
        article_topics = []
        article_topic = ""
        for _, row in self.df.iterrows():
            if article_topic != row["article_topic"]:
                sentences.append([])
                article_topics.append(row["article_topic"])
            article_topic = row["article_topic"]
            sentence = row["sentence"]  
            sentences[-1].append(sentence)
        return sentences, article_topics
    
    
    def label_the_data_hopefully_improved_opeanai(self, sentences, article_topics):

        for i in range(0, len(sentences)):
            article_topic = article_topics[i]
            article_sentences = sentences[i]
            chunk_size = 25
            total_output = []
            
            # process article
            for j in range(0, len(article_sentences), chunk_size):
                chunk = article_sentences[j:j + chunk_size]  # get a chunk of up to 25 sentences

                joined_sentences = ""
                for idx, sentence in enumerate(chunk, start=j + 1):
                    joined_sentences += str(idx) + ") " + sentence + "\n"
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": f"""
                             Your response must be in the form: Index)) Sentence * Label^@
                            
                            Example:
                            '
                            1)) Saka has scored 2 goals in today's match against Manchester City * 0^@
                            2)) Saka was clearly the best player on the pitch today * 1^@
                            3)) He said "Manchester City did not perform the best in yesterday's game" * 1^@
                            4)) Manchester United confirmed Anthony will play for them this season * 0^@
                            5)) It was a terrible game * 1^@
                            '
                        """
                        },
                        {
                            "role": "user",
                            "content": joined_sentences
                        }
                    ]
                )

                output = response.choices[0].message.content
                output_split = output.strip().split("^@")
                output_split.pop()

                if len(output_split) != len(chunk):
                    print(f"At index {i}, the article {article_topics[i]}, chunk {j // chunk_size + 1}, the number of labels is {len(output_split)}, while the number of sentences in the chunk is {len(chunk)}")
                    break

                total_output.extend(output_split)

            if len(total_output) != len(article_sentences):
                print(f"At index {i}, the article {article_topics[i]}, the number of labels is {len(total_output)}, while the number of sentences in the article is {len(article_sentences)}")
                break
            
            processed_article_sentences = []
            processed_labels = []

            for out in total_output:
                try:
                    sentence, label = out.strip().split(' * ')
                    processed_article_sentences.append(sentence.split("))")[-1])
                    processed_labels.append(label)
                except:
                    print(out)

            with open("processed_football_articles.csv", mode='a', newline='', encoding='utf-8') as f:
                for k in range(0, len(processed_labels)):
                    f.write(f'"{article_topic}","{processed_article_sentences[k].strip()}","{processed_labels[k].strip()}"\n')

            print(f"Processed article {i+1}/{len(article_topics)}")
        
    def gemini_label(self, sentences, article_topics):
        print("Labelling starting...")
        for i in range(0, len(sentences)):
            article_topic = article_topics[i]
            article_sentences = sentences[i]
            chunk_size = 25
            total_output = []
            
            # process article
            for j in range(0, len(article_sentences), chunk_size):
                chunk = article_sentences[j:j + chunk_size]  # get a chunk of up to 25 sentences
                joined_sentences = ""
                for idx, sentence in enumerate(chunk, start=j + 1):
                    joined_sentences += str(idx) + ") " + sentence + "\n"
                response = self.model.generate_content(
                    f"""     
                        Label the following {len(chunk)} sentences from an article on {article_topic} using the format: Index)) Sentence * Label^@.

                        {chunk} 
                    """

                )
                output = response.text
                output_split = output.strip().split("^@")
                output_split.pop()

                if len(output_split) != len(chunk):
                    print(f"At index {i}, the article {article_topics[i]}, chunk {j // chunk_size + 1}, the number of labels is {len(output_split)}, while the number of sentences in the chunk is {len(chunk)}")
                    break

                total_output.extend(output_split)

            if len(total_output) != len(article_sentences):
                print(f"At index {i}, the article {article_topics[i]}, the number of labels is {len(total_output)}, while the number of sentences in the article is {len(article_sentences)}")
                break
            
            processed_article_sentences = []
            processed_labels = []

            for out in total_output:
                try:
                    sentence, label = out.strip().split(' * ')
                    processed_article_sentences.append(sentence.split("))")[-1])
                    processed_labels.append(label)
                except:
                    print(out)

            with open("gemini_labelled_football_articles.csv", mode='a', newline='', encoding='utf-8') as f:
                for k in range(0, len(processed_labels)):
                    f.write(f'"{article_topic}","{processed_article_sentences[k].strip()}","{processed_labels[k].strip()}"\n')

            print(f"Processed article {i}/{len(article_topics)}")
            time.sleep(5)

    def generate_arguments(self, sentences):
        
        generated_sentences = {}

        model = genai.GenerativeModel(
            model_name = "gemini-1.5-flash",
            system_instruction = """
                        TASK: Generate linguistic variations of the given football-related arguments consisting of both a claim and a premise.
                        
                        Guidelines: Maintain the argumentative structure consisting of both a claim and a sentence.
                        Use synonyms, paraphrasing, and different sentence structures to generate new arguments.
                        The generated arguments can be about different teams, players, technical directors (from real-world football) but all arguments must be about football.
                        You can experiment with different argument lengths and argument structures.

                        Example Input: Manchester United's recent performance under Erik ten Hag suggests they are a strong contender for the Premier League title.
                        Example Output: 1)) Manchester United's recent performances suggest they have the quality and depth to challenge for the Premier League title.@@
                        2)) The Red Devils' resurgence under Erik ten Hag has reignited their hopes of reclaiming the Premier League crown.@@
                        3)) Erik ten Hag has breathed new life into Manchester United, transforming them from a sleeping giant into a waking powerhouse.@@
                        """
        )

        for sentence in sentences:
            response = model.generate_content(sentence).text
            output = response.strip().split("@@")
            output.pop()
            generated_sentences[sentence] = output

        return generated_sentences 



    def save_labels_backup(self, labels, file_name="labels_backup.txt"):
        with open(file_name, 'w') as f:
            for label in labels:
                f.write(f"{label}\n")

    def articles_with_missing_labels(self, sentences, article_topics):

        articles_missing_labels = []
        with open('article_labels.json', 'r') as file:
            data = json.load(file)

        for i in range(len(article_topics)):
            article_sentences = sentences[i]
            article_topic = article_topics[i]
            labels_for_article = data[article_topic]

            if(len(labels_for_article) != len(article_sentences)):
                articles_missing_labels.append(article_topic)

        return articles_missing_labels, len(articles_missing_labels)





if __name__ == "__main__":
    label = Labelling()
    label.parse_sentences()
    sentences, article_topics = label.get_all_sentences()
    label.gemini_label(sentences, article_topics)