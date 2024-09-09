from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords
from flask import Flask, request, jsonify
import re
import requests
from typing import Dict, Any, List
import spacy
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import MetadataQuery
from dotenv import load_dotenv
import os
import uuid
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)

nlp = spacy.load("en_core_web_lg")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
load_dotenv()


wcd_url = os.getenv("WEAVIATE_URL")
wcd_api_key = os.getenv("WEAVIATE_API_KEY")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,  
    auth_credentials=AuthApiKey(api_key=wcd_api_key)
)



class_obj = {
    "class": "Conversation",
    "vectorizer": "none",  
    "vectorIndexType": "hnsw",
    "properties": [
        {"name": "transcript", "dataType": ["text"]},
        {"name": "duration", "dataType": ["number"]},
        {"name": "language", "dataType": ["string"]},
        {"name": "entities", "dataType": ["string[]"]},
        {"name": "keywords", "dataType": ["string[]"]},
        {"name": "topics", "dataType": ["string[]"]}
    ]
}


try:
    
    
    if not client.collections.exists("Conversation"):
        client.collections.create(
            "Conversation",
            properties=[
                Property(name="transcript", data_type=DataType.TEXT),
                Property(name="duration", data_type=DataType.NUMBER),
                Property(name="language", data_type=DataType.TEXT),
                Property(name="entities", data_type=DataType.TEXT_ARRAY),
                Property(name="keywords", data_type=DataType.TEXT_ARRAY),
                Property(name="topics", data_type=DataType.TEXT_ARRAY),
            ]
        )
        print("Created 'Conversation' collection.")
    else:
        print("'Conversation' collection already exists.")
finally:
    client.close()  


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def improved_ner(text: str) -> Dict[str, List[str]]:
    doc = nlp(text)
    entities = {
        "PERSON": [],
        "ORG": [],
        "PRODUCT": [],
        "GPE": [],
        "TECH": []
    }
    
    
    for sent in doc.sents:
        
        if sent.text.lower().startswith(("hi ", "hey ", "hello ")):
            for token in sent[1:]:
                if token.pos_ == "PROPN":
                    entities["PERSON"].append(token.text)
                    break
    
    
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
        
        elif ent.label_ == "PRODUCT" or (ent.label_ == "ORG" and any(tech_word in ent.text.lower() for tech_word in ["database", "software", "api", "system"])):
            entities["TECH"].append(ent.text)
    
    
    tech_indicators = ["database", "software", "api", "system", "algorithm", "framework"]
    for chunk in doc.noun_chunks:
        if any(indicator in chunk.text.lower() for indicator in tech_indicators):
            entities["TECH"].append(chunk.text)
    
    
    for category in entities:
        entities[category] = list(set(entities[category]))
        entities[category] = [e.title() for e in entities[category]]
    
    return entities

def preprocess_for_lda(text):
    tokens = nltk.word_tokenize(text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 3]

def extract_keywords_and_topics(text: str, num_keywords: int = 10, num_topics: int = 3) -> Dict[str, Any]:
    
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    
    vectorizer = TfidfVectorizer(max_features=num_keywords)
    tfidf_matrix = vectorizer.fit_transform([' '.join(words)])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    keywords = [(feature_names[i], tfidf_scores[i]) for i in tfidf_scores.argsort()[::-1]]
    
    
    nmf_model = NMF(n_components=num_topics, random_state=1)
    nmf_output = nmf_model.fit_transform(tfidf_matrix)
    
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_features_ind = topic.argsort()[:-10 - 1:-1]
        top_features = [(feature_names[i], topic[i]) for i in top_features_ind]
        topics.append({
            'id': topic_idx,
            'terms': top_features
        })
    
    return {
        'keywords': keywords,
        'topics': topics
    }

def generate_embeddings(text: str) -> List[float]:
    embedding = embedding_model.encode([text])[0]
    return embedding.tolist()



def store_conversation(conversation_data: Dict[str, Any]):
    conversation_id = str(uuid.uuid4())
    full_embedding = conversation_data['full_text_embedding']

    
    data_object = {
        "transcript": conversation_data['original_transcription'],
        "duration": conversation_data['duration'],
        "language": conversation_data['language'],
        "entities": [f"{k}:{','.join(v)}" for k, v in conversation_data['entities'].items()],
        "keywords": [f"{k}:{v}" for k, v in conversation_data['keywords']],
        "topics": [f"{t['id']}:{','.join([f'{term}:{score}' for term, score in t['terms']])}" for t in conversation_data['topics']]
    }

    
    client.connect()

    try:
        
        client.collections.get("Conversation").data.insert(
            properties=data_object,
            vector=full_embedding,
            uuid=conversation_id
        )
    finally:
        
        client.close()

    return conversation_id













def process_transcription(transcription_data: Dict[str, Any]) -> Dict[str, Any]:
    cleaned_segments = []
    full_text = transcription_data['transcription']
    
    for segment in transcription_data['segments']:
        cleaned_text = clean_text(segment['text'])
        cleaned_segment = {
            'id': segment['id'],
            'start': segment['start'],
            'end': segment['end'],
            'original_text': segment['text'],
            'cleaned_text': cleaned_text
        }
        cleaned_segments.append(cleaned_segment)
    
    entities = improved_ner(full_text)
    keywords_and_topics = extract_keywords_and_topics(full_text)
    full_text_embedding = generate_embeddings(full_text)

    processed_data = {
        'original_transcription': full_text,
        'cleaned_transcription': clean_text(full_text),
        'cleaned_segments': cleaned_segments,
        'language': transcription_data['language'],
        'duration': transcription_data['duration'],
        'entities': entities,
        'keywords': keywords_and_topics['keywords'],
        'topics': keywords_and_topics['topics'],
        'full_text_embedding': full_text_embedding
    }
    
    conversation_id = store_conversation(processed_data)
    processed_data['id'] = conversation_id

    return processed_data

@app.route('/process', methods=['POST'])
def process():
    audio_file = request.files['file']
    
    files = {'file': (audio_file.filename, audio_file.read(), audio_file.mimetype)}
    
    transcription_response = requests.post(
        'http://127.0.0.1:8000/transcribe/',  
        files=files
    )
    
    if transcription_response.status_code == 200:
        transcription_data = transcription_response.json()
    else:
        return jsonify({"error": "Transcription failed"}), transcription_response.status_code
    
    processed_data = process_transcription(transcription_data)

    return jsonify(processed_data)



@app.before_request
def before_request():
    client.connect()

@app.teardown_appcontext
def teardown_appcontext(exception=None):
    client.close()




@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    limit = request.json.get('limit', 5)
    
    # Generate embedding for the query
    query_embedding = generate_embeddings(query)
    
    # Get the Conversation collection
    conversation_collection = client.collections.get("Conversation")
    
    # Perform the search
    response = conversation_collection.query.near_vector(
        near_vector=query_embedding,
        limit=limit,
        return_metadata=MetadataQuery(distance=True)
    )
    
    # Format the results
    formatted_results = []
    for obj in response.objects:
        formatted_result = {
            'id': obj.uuid,
            'score': 1 - obj.metadata.distance,  # Convert distance to a similarity score
            'distance': obj.metadata.distance,
            'transcript': obj.properties.get('transcript'),
            'duration': obj.properties.get('duration'),
            'language': obj.properties.get('language'),
            'entities': parse_entities(obj.properties.get('entities', [])),
            'keywords': parse_keywords(obj.properties.get('keywords', [])),
            'topics': parse_topics(obj.properties.get('topics', []))
        }
        formatted_results.append(formatted_result)
    
    return jsonify(formatted_results)

def parse_entities(entities):
    return {k: v.split(',') for k, v in [e.split(':') for e in entities]}

def parse_keywords(keywords):
    return {k: float(v) for k, v in [kw.split(':') for kw in keywords]}


def parse_topics(topics):
    parsed_topics = []
    for topic in topics:
        try:
            parts = topic.split(':')
            if len(parts) < 2:
                continue
            topic_id = int(parts[0])
            terms = []
            for term_score in parts[1].split(','):
                term_parts = term_score.split(':')
                if len(term_parts) == 2:
                    terms.append((term_parts[0], float(term_parts[1])))
            parsed_topics.append({
                'id': topic_id,
                'terms': terms
            })
        except Exception as e:
            logging.warning(f"Error parsing topic: {topic}. Error: {str(e)}")
    return parsed_topics


if __name__ == '__main__':
    app.run(debug=True, port=5000)


