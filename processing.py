from flask import Flask, request, jsonify
import re
import requests
from typing import Dict, Any, List
import spacy
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)

nlp = spacy.load("en_core_web_lg")

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

def perform_topic_modeling(text: str, num_topics: int = 3) -> List[Dict[str, Any]]:
    processed_text = preprocess_for_lda(text)
    
    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary([processed_text])
    
    # Create a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in [processed_text]]
    
    # Generate LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100,
                         update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    
    # Extract topics
    topics = []
    for idx, topic in lda_model.print_topics(-1):
        topic_terms = [(term.split('*')[1].strip().replace('"', ''), float(term.split('*')[0])) 
                       for term in topic.split(' + ')]
        topics.append({
            'id': idx,
            'terms': topic_terms
        })
    
    return topics

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
    topics = perform_topic_modeling(full_text)
    
    processed_data = {
        'original_transcription': full_text,
        'cleaned_transcription': clean_text(full_text),
        'cleaned_segments': cleaned_segments,
        'language': transcription_data['language'],
        'duration': transcription_data['duration'],
        'entities': entities,
	'topics': topics
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
