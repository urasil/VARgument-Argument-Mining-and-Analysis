import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai
from ast import literal_eval
from identify_arguments import ArgumentIdentification
from cluster_arguments import Clustering
from topic_extraction import ExtractTopic
from collections import defaultdict
import time
import re

class Main:
    def __init__(self):
        self.identifier = ArgumentIdentification(model_name="../Pure_RoBERTa")
        self.cluster = Clustering(num_clusters=100, eps=0.3, min_samples=5)
        self.extract = ExtractTopic()
        self.topic_to_args = defaultdict(list)
    
    def combine_entity_information_with_arguments(self, csv_file, txt_file, output_csv):
        df = pd.read_csv(csv_file)
        entity_data = {"Player": [], "Team": [], "Manager": []}

        with open(txt_file, 'r') as f:
            txt_lines = f.readlines()

        for line in txt_lines:
            line = line.strip() 
            if len(line) < 2:
                continue            
            line = line.rstrip('@')
            line = line[2:]
            
            if "Player:" not in line:
                line += "@Player:None"
            if "Team:" not in line:
                line += "@Team:None"
            if "Manager:" not in line:
                line += "@Manager:None"

            # Extract Player, Team, and Manager values
            match = re.search(r"Player:([^@]*)@Team:([^@]*)@Manager:([^@]*)", line)
            if match:
                player = match.group(1).strip()
                team = match.group(2).strip()
                manager = match.group(3).strip()
            else:
                player, team, manager = "None", "None", "None"

            entity_data["Player"].append(player)
            entity_data["Team"].append(team)
            entity_data["Manager"].append(manager)

        if len(entity_data["Player"]) != len(df):
            raise ValueError("The number of valid lines in the text file does not match the number of rows in the CSV.")

        df["Player"] = entity_data["Player"]
        df["Team"] = entity_data["Team"]
        df["Manager"] = entity_data["Manager"]
        df.to_csv(output_csv, index=False)

        print(f"Updated CSV saved to '{output_csv}'.")



    def entity_type(self, csv_file, output_file):
        # determine the entity type for each label-entity pair in a given file and append the results to an output file

        df = pd.read_csv(csv_file)
        arguments = df['argument_sentences'].tolist()
        print(len(arguments))
        load_dotenv()
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(
            model_name = "gemini-1.5-flash",
            system_instruction = """You are a model that determines whether an entity belongs to one of the following categories: Player, Team, Team Manager or None. Your goal is to identify the entity type for each argument. Your response should be the entity type for each argument and the full name of that entity. 
            Example Input:
            1)But his value to Fulham, following his arrival from Everton last year, has been abundantly clear this term.
            2)Iwobi has dovetailed effectively with Emile Smith Rowe and Antonee Robinson on Fulham's left and, with licence to drift infield, the quality of his passing has stood out more than ever.
            3)Niclas Fullkrug's continued absence with a calf injury means Michail Antonio is likely to keep his place up front for West Ham against Tottenham on Saturday.
            Example Output:
            1)Team:Fulham,Everton@Player:None@Manager:None
            2)Team:Fulham@Player:Emile Smith Rowe,Antonee Robinson,Alex Iwobi@Manager:None  
            3)Player:Michail Antonio, Niclas Fullkrug@Team:West Ham, Tottenham@Manager:None  
            
            The number of input sentences must match the number of response sentences.
            Never put anything else in your response and never deviate from the format.
            """
        )
        
        with open(output_file, "a") as outfile:
            batch_size = 5
            for i in range(217*5, len(arguments), batch_size):
                print(f"Processing batch {i//batch_size} / {len(arguments)//batch_size}...")
                batch = arguments[i:i+batch_size]
                batch_input = [f"{i+1}){sen}\n" for i, sen in enumerate(batch)]
                response = self.model.generate_content(batch_input)
                for idx, sentence_output in enumerate(response.text.split("\n")):
                    outfile.write(f"{sentence_output}\n")
                time.sleep(5)

        print(f"Entity types appended to '{output_file}'.")


    def entity_clustering(self, csv_file):
        # Cluster arguments based on sentence embeddings and extracted entities
        df = pd.read_csv(csv_file)
        arguments = df['argument_sentences'].tolist()
        self.cluster.num_cluster = 100
        labels, entities = self.cluster.cluster("entity", arguments)
        with open('pipeline/a.txt', 'w') as f:
            for i, (arg, label, ents) in enumerate(zip(arguments, labels, entities)):
                f.write(f"{i},{arg},{ents}\n")

    def cluster_and_label_args(self, csv_file, method="kmeans", embedding_type="ft"):
        df = pd.read_csv(csv_file)

        df[f'{embedding_type}_embeddings'] = df[f'{embedding_type}_embeddings'].apply(literal_eval)
        embeddings = np.array(df[f'{embedding_type}_embeddings'].tolist())
        arguments = df['argument_sentences'].tolist()

        if method == 'kmeans':
            cluster_labels = self.cluster.cluster("kmeans", arguments, embeddings)
        else:
            cluster_labels = self.cluster.cluster("hierarchical", arguments, embeddings)

        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(arguments[idx])

        total_clusters = len(clusters)
        for i, (label, args) in enumerate(clusters.items(), 1):
            print(f"Processing cluster {i}/{total_clusters} with {len(args)} arguments...")
            topic = self.extract.determine_topic(args)  # Call LLM once per cluster
            self.topic_to_args[topic] = args
        return self.topic_to_args

    def cluster_and_save(self, csv_file):
        embedding_types = ['st']
        clustering_methods = ['hierarchical']

        for embedding_type in embedding_types:
            for method in clustering_methods:
                results = self.cluster_and_label_args(csv_file, method, embedding_type)
                output_file = f'results_{embedding_type}_{method}.txt'
                with open(output_file, 'w') as f:
                    for topic in results.keys():
                        f.write(f'Topic: {topic}\n')
                    for topic, args in results.items():
                        f.write(f'Topic: {topic}\nArguments:\n')
                        for arg in args:
                            f.write(f'- {arg}\n')
                        f.write('\n')
        return "Clustering and saving complete."

if __name__ == "__main__":
    main = Main()
    print("a")
    """
    argument_embeddings_output is deleted as entity clustering is scraped and is no longer needed
    """
    print(main.cluster_and_save('pipeline/argument_embeddings_output.csv')) 
    #main.entity_type('pipeline/argument_embeddings_output.csv', 'pipeline/a.txt')
    #main.combine_entity_information_with_arguments('pipeline/argument_embeddings_output.csv', 'pipeline/a.txt', 'pipeline/deneme.csv')