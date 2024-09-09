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
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)

nlp = spacy.load("en_core_web_lg")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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
    
    # Process each sentence
    for sent in doc.sents:
        # Look for greeting patterns to identify names
        if sent.text.lower().startswith(("hi ", "hey ", "hello ")):
            for token in sent[1:]:
                if token.pos_ == "PROPN":
                    entities["PERSON"].append(token.text)
                    break
    
    # Use spaCy's NER for other entities
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
        # Classify potential tech terms
        elif ent.label_ == "PRODUCT" or (ent.label_ == "ORG" and any(tech_word in ent.text.lower() for tech_word in ["database", "software", "api", "system"])):
            entities["TECH"].append(ent.text)
    
    # Additional heuristics for tech terms
    tech_indicators = ["database", "software", "api", "system", "algorithm", "framework"]
    for chunk in doc.noun_chunks:
        if any(indicator in chunk.text.lower() for indicator in tech_indicators):
            entities["TECH"].append(chunk.text)
    
    # Remove duplicates and ensure consistent capitalization
    for category in entities:
        entities[category] = list(set(entities[category]))
        entities[category] = [e.title() for e in entities[category]]
    
    return entities

def preprocess_for_lda(text):
    tokens = nltk.word_tokenize(text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 3]

def extract_keywords_and_topics(text: str, num_keywords: int = 10, num_topics: int = 3) -> Dict[str, Any]:
    # Preprocess text
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Keyword Extraction using TF-IDF
    vectorizer = TfidfVectorizer(max_features=num_keywords)
    tfidf_matrix = vectorizer.fit_transform([' '.join(words)])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    keywords = [(feature_names[i], tfidf_scores[i]) for i in tfidf_scores.argsort()[::-1]]
    
    # Topic Modeling using NMF
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
    

    
    return processed_data

@app.route('/process', methods=['POST'])
def process():
    # Receive the audio file from the request
    audio_file = request.files['file']
    
    # Send the file to the Transcription API (FastAPI) with correct file format
    files = {'file': (audio_file.filename, audio_file.read(), audio_file.mimetype)}
    
    transcription_response = requests.post(
        'http://127.0.0.1:8000/transcribe/',  # Assuming FastAPI is running here
        files=files
    )
    
    # Check if Transcription API returned a valid response
    if transcription_response.status_code == 200:
        transcription_data = transcription_response.json()
    else:
        return jsonify({"error": "Transcription failed"}), transcription_response.status_code
    
    # Process the transcription data (cleaning, etc.)
    processed_data = process_transcription(transcription_data)

    # Return the final processed data
    return jsonify(processed_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
