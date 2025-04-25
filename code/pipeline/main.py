import tkinter as tk
from tkinter import scrolledtext, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from similarity_search_retrival import ArgumentRetriever, ArgumentVisualizer


"""
The VARgument tool GUI. Very basic, can be improved.
"""

class Tool:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentence Similarity Finder")
        self.root.geometry("900x700")
        """
        IF there is a problem with the knowledge base, the error most likely because of the EOL Sequence (CRLF or LF) in the CSV file.
        Should work correctly as is in a Windows machine, but if you are using a different OS, make sure to check the EOL sequence.
        """
        self.knowledge_base = pd.read_csv("pipeline/final_knowledgebase.csv") # This file is absolutely necessary, make sure the naming is correct, make sure there are no errors with EOL sequence (CRLF or LF)
        self.label = tk.Label(root, text="Enter a sentence:")
        self.label.pack(pady=5)
        self.entry = tk.Entry(root, width=80)
        self.entry.pack(pady=5)
        
        # search 
        self.button = tk.Button(root, text="Find Similar Sentences", command=self.find_similar)
        self.button.pack(pady=10)

        """
        Should be uncommented to toggle to Extended_RoBERTa embeddings.
        """

        # toggle to change embeddings
        #self.embedding_type = tk.StringVar(value="st")
        #self.toggle_button = tk.Button(root, text="Using ST Embeddings", command=self.toggle_embeddings)
        #self.toggle_button.pack(pady=5)
        
        # number of similar sentences
        self.num_sentences_label = tk.Label(root, text="Number of similar sentences:")
        self.num_sentences_label.pack(pady=5)
        self.num_sentences = tk.IntVar(value=5)
        self.num_sentences_dropdown = ttk.Combobox(root, textvariable=self.num_sentences, values=[str(1), str(3), str(5), str(10), str(15)], state="readonly")
        self.num_sentences_dropdown.pack(pady=5)
        
        self.result_text = scrolledtext.ScrolledText(root, height=20, width=120)
        self.result_text.pack(pady=10)
        
        # perform Temporal Analysis Button (Initially Disabled)
        self.temporal_button = tk.Button(root, text="Perform Temporal Analysis", state=tk.DISABLED, command=self.analyze_temporal)
        self.temporal_button.pack(pady=5)
        
        # FAISS retrievers
        self.retriever_st = ArgumentRetriever("sentence-transformers", "pipeline/final_knowledgebase.csv", "st")
        """
        Should be uncommented to toggle to Extended_RoBERTa embeddings.
        """
        #self.retriever_ft = ArgumentRetriever("../Pure_RoBERTa", "pipeline/final_knowledgebase.csv", "ft")
        self.retriever = self.retriever_st#
        """
            Should be uncommented to toggle to Extended_RoBERTa embeddings.
        """
    """
    def toggle_embeddings(self):
        if self.embedding_type.get() == "st":
            self.embedding_type.set("ft")
            self.retriever = self.retriever_ft
            self.toggle_button.config(text="Using FT Embeddings")
        else:
            self.embedding_type.set("st")
            self.retriever = self.retriever_st
            self.toggle_button.config(text="Using ST Embeddings")
    """
    
    def old_find_similar(self):
        user_sentence = self.entry.get()
        if not user_sentence:
            return
        
        top_k = self.num_sentences.get()
        top_sentences, input_embedding = self.retriever.retrieve_similar_sentences(user_sentence, top_k=top_k)
        
        self.result_text.delete('1.0', tk.END)
        for idx, row in top_sentences.iterrows():
            sentiment_info = f"Positive: {row['Positive']}, Negative: {row['Negative']}, Neutral: {row['Neutral']}"
            polarity_info = f"Polarity: {row['Supporting']} (Supporting) | {row['Attacking']} (Attacking) | {row['No Polarity']} (No Polarity)"
            
            self.result_text.insert(tk.END, f"{row['argument_sentences']}\n")
            self.result_text.insert(tk.END, f"Distance: {row['distance']:.4f}\n")
            self.result_text.insert(tk.END, f"{sentiment_info}\n")
            #self.result_text.insert(tk.END, f"{polarity_info}\n") polarity info omitted in the report
            self.result_text.insert(tk.END, "-" * 100 + "\n")
        
        self.temporal_button.config(state=tk.NORMAL)
        #self.selected_index = top_sentences.index[0]
        self.lowest_index = top_sentences.index.min()
        self.highest_index = top_sentences.index.max()
        #print(f"Indexes: {top_sentences.index}")
        
        ArgumentVisualizer.visualize(user_sentence, top_sentences, input_embedding, np.vstack([input_embedding, self.retriever.embeddings[top_sentences.index]]))
    
    def find_similar(self):
        user_sentence = self.entry.get()
        if not user_sentence:
            return
        
        top_k = self.num_sentences.get()
        top_sentences, input_embedding = self.retriever.retrieve_similar_sentences(user_sentence, top_k=top_k)
        
        self.result_text.delete('1.0', tk.END)
        for idx, row in top_sentences.iterrows():
            sentiment_info = f"Positive: {row['Positive']}, Negative: {row['Negative']}, Neutral: {row['Neutral']}"
            polarity_info = f"Polarity: {row['Supporting']} (Supporting) | {row['Attacking']} (Attacking) | {row['No Polarity']} (No Polarity)"

            
            self.result_text.insert(tk.END, f"{row['argument_sentences']}\n")
            self.result_text.insert(tk.END, f"Distance: {row['distance']:.4f}\n")              
            self.result_text.insert(tk.END, f"{sentiment_info}\n")
            self.result_text.insert(tk.END, f"{polarity_info}\n")
            self.result_text.insert(tk.END, "-" * 100 + "\n")
        
        self.temporal_button.config(state=tk.NORMAL)
        self.lowest_index = top_sentences.index.min()
        self.highest_index = top_sentences.index.max()
        
        ArgumentVisualizer.visualize(user_sentence, top_sentences, input_embedding, np.vstack([input_embedding, self.retriever.embeddings[top_sentences.index]]))
    

    def analyze_temporal(self):
        self.perform_temporal_analysis(self.lowest_index, self.highest_index)
    
    def perform_temporal_analysis_old(self, lowest, highest, window=50):
        older_sentences = self.knowledge_base.iloc[max(0, lowest - window):lowest+1]
        newer_sentences = self.knowledge_base.iloc[highest:highest + 1 + window]

        if older_sentences.empty or newer_sentences.empty:
            print("Not enough data")
            return

        features = [
            "lexical_richness", "negation_frequency", "syntax_depth", "use_of_intensifiers",
            "question_frequency", "num_quotes", "adj_frequency", "verb_frequency",
            "noun_frequency", "argument_length", "argument_complexity", "emotion_tone_shift",
            "formality", "domain_specific_term_percentage"
        ]

        fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(10, len(features) * 2))

        for i, feature in enumerate(features):
            if feature in self.knowledge_base.columns:
                x_older = range(len(older_sentences)) 
                x_newer = range(len(older_sentences), len(older_sentences) + len(newer_sentences))  

                axes[i].plot(older_sentences.index, older_sentences[feature], label="Older Arguments", color="blue", marker="o")
                axes[i].plot(newer_sentences.index, newer_sentences[feature], label="Newer Arguments", color="red", marker="x")
                
                axes[i].set_title(feature, fontsize=10)
                axes[i].legend(fontsize=8)
                axes[i].tick_params(axis='both', labelsize=8)

        plt.tight_layout()
        plt.show()

        older_summary = older_sentences[features].mean().to_frame(name="Older Arguments")
        newer_summary = newer_sentences[features].mean().to_frame(name="Newer Arguments")
        comparison_table = older_summary.join(newer_summary)
        
        #print("Feature Comparison Between Older and Newer Arguments:")
        #print(comparison_table)
        # Code for the radar chart -> mostly taken and adapted from matplotlib radar chart website and stackoverflow
    """
    def perform_temporal_analysis(self, lowest, highest, window=50):
        older_sentences = self.knowledge_base.iloc[max(0, lowest - window):lowest+1]
        newer_sentences = self.knowledge_base.iloc[highest:highest + 1 + window]

        if older_sentences.empty or newer_sentences.empty:
            print("Not enough data")
            return

        features = [
            "lexical_richness", "negation_frequency", "syntax_depth", "use_of_intensifiers",
            "question_frequency", "num_quotes", "adj_frequency", "verb_frequency",
            "noun_frequency", "argument_length", "argument_complexity", "emotion_tone_shift",
            "formality", "domain_specific_term_percentage"
        ]

        # Instead of plotting raw indices, create relative positions for better visualization
        # and calculate averages for cleaner comparison
        older_avg = older_sentences[features].mean()
        newer_avg = newer_sentences[features].mean()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        width = 0.35
        
        # plot means rather than individual points
        rects1 = ax.bar(x - width/2, older_avg, width, label='Older Arguments', color='blue')
        rects2 = ax.bar(x + width/2, newer_avg, width, label='Newer Arguments', color='red')
        
        ax.set_title('Comparison of Argument Features Between Time Periods', fontsize=15)
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Average Value', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=12)
        
        percent_changes = (newer_avg - older_avg) / older_avg * 100
        
        table_data = []
        for i, feature in enumerate(features):
            table_data.append([
                feature, 
                f"{older_avg[feature]:.3f}", 
                f"{newer_avg[feature]:.3f}", 
                f"{percent_changes[feature]:+.1f}%"
            ])
        
        fig_table, ax_table = plt.subplots(figsize=(12, 6))
        ax_table.axis('off')
        table = ax_table.table(
            cellText=table_data,
            colLabels=['Feature', 'Older Arguments', 'Newer Arguments', 'Change (%)'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.title('Detailed Comparison of Argument Features', fontsize=15)
        plt.tight_layout()
        plt.show()
        
        # create a radar chart for multi-dimensional comparison
        # Normalis the data for radar chart
        max_values = np.maximum(older_avg, newer_avg)
        min_values = np.minimum(older_avg, newer_avg)
        data_range = max_values - min_values
        data_range = np.where(data_range == 0, 1, data_range)
        older_normalized = (older_avg - min_values) / data_range
        newer_normalized = (newer_avg - min_values) / data_range
        
        fig_radar = plt.figure(figsize=(10, 8))
        ax_radar = fig_radar.add_subplot(111, polar=True)
        angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        older_values = older_normalized.tolist()
        older_values += older_values[:1]
        newer_values = newer_normalized.tolist()
        newer_values += newer_values[:1]
        ax_radar.plot(angles, older_values, 'o-', linewidth=2, label='Older Arguments', color='blue')
        ax_radar.plot(angles, newer_values, 'o-', linewidth=2, label='Newer Arguments', color='red')
        ax_radar.fill(angles, older_values, alpha=0.1, color='blue')
        ax_radar.fill(angles, newer_values, alpha=0.1, color='red')
        feature_labels = features.copy()
        feature_labels += feature_labels[:1]  # Close the loop
        ax_radar.set_xticks(angles)
        ax_radar.set_xticklabels(feature_labels, size=10)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Radar Chart Comparison of Argument Features', size=15)
        plt.tight_layout()
        plt.show()
"""
    def perform_temporal_analysis(self, lowest, highest, window=50):
        older_sentences = self.knowledge_base.iloc[max(0, lowest - window):lowest+1]
        newer_sentences = self.knowledge_base.iloc[highest:highest + 1 + window]

        if older_sentences.empty or newer_sentences.empty:
            print("Not enough data")
            return

        features = [
            "lexical_richness", "negation_frequency", "syntax_depth", "use_of_intensifiers",
            "question_frequency", "num_quotes", "adj_frequency", "verb_frequency",
            "noun_frequency", "argument_length", "argument_complexity", "emotion_tone_shift",
            "formality", "domain_specific_term_percentage"
        ]

        # averages for each time period
        older_avg = older_sentences[features].mean()
        newer_avg = newer_sentences[features].mean()
        
        # ! change to bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(features))
        width = 0.35
        rects1 = ax.bar(x - width/2, older_avg, width, label='Older Arguments', color='blue')
        rects2 = ax.bar(x + width/2, newer_avg, width, label='Newer Arguments', color='red')       
        ax.set_title('Comparison of Argument Features Between Time Periods', fontsize=15)
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Average Value', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=12)
        percent_changes = (newer_avg - older_avg) / older_avg * 100
        
        table_data = []
        for i, feature in enumerate(features):
            table_data.append([
                feature, 
                f"{older_avg[feature]:.3f}", 
                f"{newer_avg[feature]:.3f}", 
                f"{percent_changes[feature]:+.1f}%"
            ])
        
        fig_table, ax_table = plt.subplots(figsize=(12, 6))
        ax_table.axis('off')
        table = ax_table.table(
            cellText=table_data,
            colLabels=['Feature', 'Older Arguments', 'Newer Arguments', 'Change (%)'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.title('Detailed Comparison of Argument Features', fontsize=15)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = Tool(root)
    root.mainloop()
