import spacy
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
# Use an LLM-based post-processing step to refine incorrect entity assignments.

"""
Entity clustering implementation for argument embeddings
Uses NER to extract entities from the arguments
Combines the argument embedding with the entity embedding and clusters them using kmeans
"""

def extract_specific_entities(arguments, nlp):
    #Extract specific entities (Player, Team, Manager) from arguments
    players_teams_managers = []
    raw_entities_list = []

    relevant_labels = {"PERSON", "ORG"}  
    for i, arg in enumerate(arguments):
        doc = nlp(arg)
        entities = []
        raw_entities = []
        for ent in doc.ents:
            if ent.label_ in relevant_labels:
                entities.append(ent.text.lower())  
                raw_entities.append(ent.text) 
        players_teams_managers.append((i, " ".join(entities)))
        raw_entities_list.append((i, raw_entities))

    return players_teams_managers, raw_entities_list

def extract_and_normalize_entities(arguments, nlp):
    # extract and normalize entities from arguments using spaCy

    entities_list = []
    for arg in arguments:
        doc = nlp(arg)
        entities = [ent.text.lower() for ent in doc.ents]  
        entities_list.append(" ".join(entities))
    return entities_list


def cluster_sentences_with_entities(arguments, num_clusters):
    #Cluster arguments based on sentence embeddings and extracted entities
    nlp = spacy.load("en_core_web_sm")
    
    normalized_entities, raw_entities = extract_specific_entities(arguments, nlp)
    print("Entities extracted")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_embeddings = model.encode(arguments)
    entity_embeddings = model.encode(normalized_entities)
    print("Embeddings computed")
    combined_embeddings = [
        list(sent_emb) + list(ent_emb) 
        for sent_emb, ent_emb in zip(sentence_embeddings, entity_embeddings)
    ]
    kmeans = KMeans(n_clusters=num_clusters, random_state=31)
    print("Clustering...")
    cluster_labels = kmeans.fit_predict(combined_embeddings)
    return cluster_labels, raw_entities

if __name__ == "__main__":
    arguments = [
        "Player X scored a great goal in the match.",
        "Team Y's defense strategy was ineffective.",
        "Player Z was injured and missed the game.",
        "The referee's decision was controversial.",
        "Team A's midfield was dominant throughout the game.",
        "Manchester United scored a dramatic late goal.",
        "Red Devils secured the win in injury time."
    ]
    num_clusters = 3
    labels, entities = cluster_sentences_with_entities(arguments, num_clusters)
    
    for i, (arg, label, ents) in enumerate(zip(arguments, labels, entities)):
        print(f"Argument {i+1}: {arg}")
        print(f"  - Cluster Label: {label}")
        print(f"  - Entities: {ents}")
