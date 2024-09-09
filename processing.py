from flask import Flask, request, jsonify
import re
import requests
from typing import Dict, Any, List
import spacy

app = Flask(__name__)

# Load English NER model
nlp = spacy.load("en_core_web_sm")

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

#def perform_ner(text: str) -> Dict[str, list]:
#    doc = nlp(text)
#    entities = {
#        "PERSON": [],
#        "ORG": [],
#        "PRODUCT": [],
#        "GPE": [],  # Geopolitical Entities (e.g., countries, cities)
#        "TECH": []  # Custom category for technology-related terms
#    }
#    
#    for ent in doc.ents:
#        if ent.label_ in entities:
#            entities[ent.label_].append(ent.text)
#        elif ent.label_ == "MISC" and any(tech_term in ent.text.lower() for tech_term in ["database", "api", "software", "algorithm"]):
#            entities["TECH"].append(ent.text)
#    
#    return entities

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
    
#    entities = perform_ner(full_text)
    entities = improved_ner(full_text)
    
    processed_data = {
        'original_transcription': full_text,
        'cleaned_transcription': clean_text(full_text),
        'cleaned_segments': cleaned_segments,
        'language': transcription_data['language'],
        'duration': transcription_data['duration'],
        'entities': entities
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
