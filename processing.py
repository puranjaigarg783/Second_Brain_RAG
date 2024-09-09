from flask import Flask, request, jsonify
import re
import requests
from typing import Dict, Any

app = Flask(__name__)

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_transcription(transcription_data: Dict[str, Any]) -> Dict[str, Any]:
    cleaned_segments = []
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
    
    processed_data = {
        'original_transcription': transcription_data['transcription'],
        'cleaned_transcription': clean_text(transcription_data['transcription']),
        'cleaned_segments': cleaned_segments,
        'language': transcription_data['language'],
        'duration': transcription_data['duration']
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

