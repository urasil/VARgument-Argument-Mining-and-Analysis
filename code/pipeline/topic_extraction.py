import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai
import time

"""
Due to clustering methods underperforming, and eventually being scraped from the dissertation, this class is not necessary.
"""

class ExtractTopic:
    def __init__(self):
        
        load_dotenv()
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel(
            model_name = "gemini-1.5-flash",
            system_instruction = """
            You are a topic identification model tasked with determining the overarching topic for a group of arguments. Your goal is to identify an appropriate topic for the cluster of sentences you are given. Your response should be concise but make sure it's not extremely general, such as "Football Analysis".

            Example:  

            Arguments:
            1. Everton's lack of consistency in performance, especially in attack and defense, shows that they struggle to maintain competitive results, leading to their underperformance.  
            2. Southampton's ineffective transfer policy, failing to reinvest properly in the squad, has resulted in an unbalanced team that cannot compete consistently in the Premier League.  
            3. Watford's managerial instability, with frequent changes in leadership, has prevented the team from developing a cohesive tactical approach, contributing to their underachievement.  
            4. Leeds United's defensive vulnerabilities, arising from their high-pressing style, have left them exposed and contributed to their underperformance in the league.  
            5. Newcastle United's persistent injuries to key players undermine their ability to compete effectively, resulting in a disappointing season.  
            6. Aston Villa's poor squad depth leaves them vulnerable when key players are unavailable, directly contributing to their inconsistent performances and underachievement.  
            7. Crystal Palace's lack of effective leadership, particularly when Wilfried Zaha is not performing, causes the team to struggle in crucial moments and underperform in the Premier League.  

            Determined Topic: Underperforming Teams  
            Explanation: Each argument describes factors contributing to underperformance in competitive environments, such as lack of consistency, poor depth, or ineffective leadership. The shared theme is the inability to achieve consistent or competitive results.

            Ensure your response follows this format:  
            Arguments: [List of arguments]
            Determined Topic: [Your topic]  
            Explanation: [Your explanation of the identified topic]  

            Your task is to analyze the provided arguments, identify their shared theme, and respond in the specified format.
            """
        )

    def test_gemini(self):
        response = self.model.generate_content("Write a story about a magic backpack.")
        print(response.text)

    def determine_topic(self, group):  
        group_text = "\n".join(group) 
        response = self.model.generate_content(f"""Determine the topic of the following group of sentences:
                                                {group_text}""")
        try:
            topic = response.text.split("Determined Topic:")[1].split(",")[0].strip() 
        except IndexError:
            topic = "Unknown"
        time.sleep(2)
        return topic