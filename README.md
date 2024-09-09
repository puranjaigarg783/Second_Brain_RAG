

# Named Entity Recognition (NER) and Topic Modeling API

This project is a Flask-based API designed for processing audio transcriptions, cleaning the text, and performing Named Entity Recognition (NER) using SpaCy, keyword extraction, and topic modeling. The API also integrates with Weaviate for vector-based storage and retrieval of processed conversations.

## Table of Contents
- [Features](#features)
- [Setup Instructions](#setup-instructions)
  - [1. Install Required Dependencies](#1-install-required-dependencies)
  - [2. Run the Flask API](#2-run-the-flask-api)
  - [3. Transcription Service](#3-transcription-service)
  - [4. Making Requests](#4-making-requests)
  - [5. Example Response](#5-example-response)
  - [6. Weaviate Configuration](#6-weaviate-configuration)
- [Key Dependencies](#key-dependencies)
- [Use Cases](#use-cases)

## Features

- **Text Cleaning**: Removes special characters, converts text to lowercase, and standardizes the transcription.
- **Named Entity Recognition (NER)**: Extracts named entities such as persons, organizations, products, and technology-related terms.
- **Keyword Extraction and Topic Modeling**: Uses TF-IDF and NMF models to extract keywords and topics from the conversation text.
- **Vector Embeddings**: Generates sentence embeddings using the `SentenceTransformer` model for storage in Weaviate.
- **Weaviate Integration**: Stores the processed conversations in Weaviate with vectors, allowing for efficient vector-based search and retrieval.

## Setup Instructions

### 1. Install Required Dependencies

Ensure that you have Python 3.x installed and follow these steps to install the necessary dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install -r requirements.txt

# Download and install SpaCy's large English model
python -m spacy download en_core_web_lg

# Install nltk data
python -m nltk.downloader punkt stopwords
```

### 2. Run the Flask API

To start the Flask API locally, run the following command:

```bash
python processing.py
```

The Flask API will now be running at `http://127.0.0.1:5000`.

### 3. Transcription Service

Ensure that your FastAPI-based transcription service is running on port 8000. This service should handle audio files and convert them into text transcriptions.

### 4. Making Requests

To process an audio file, send a `POST` request to the Flask API. You can use `curl` or a tool like Postman to send the file:

```bash
curl -X POST -F 'file=@path_to_audio.m4a' http://127.0.0.1:5000/process
```

This will:

- Send the audio file to the transcription service.
- Process the transcription to extract named entities, keywords, topics, and embeddings.
- Store the processed data in Weaviate.

### 5. Example Response

Here is an example of the JSON response from the API:

```json
{
  "id": "e13a1c5d-7c12-48c6-b4d9-bfbdc667f33b",
  "original_transcription": "Hey John, great to see you at the conference.",
  "cleaned_transcription": "hey john great to see you at the conference",
  "cleaned_segments": [
    {
      "id": 0,
      "start": 0,
      "end": 3.6,
      "original_text": "Hey John, great to see you at the conference.",
      "cleaned_text": "hey john great to see you at the conference"
    }
  ],
  "language": "English",
  "duration": 62.24,
  "entities": {
    "PERSON": ["John"],
    "ORG": [],
    "PRODUCT": [],
    "GPE": [],
    "TECH": []
  },
  "keywords": [
    {"conference": 0.45},
    {"john": 0.30}
  ],
  "topics": [
    {
      "id": 0,
      "terms": [
        {"conference": 0.35},
        {"attend": 0.15}
      ]
    }
  ],
  "full_text_embedding": [...]
}
```

### 6. Weaviate Configuration

Ensure that you have a Weaviate instance running (either locally or in the cloud). You will need the following environment variables:

- `WEAVIATE_URL`: The URL of your Weaviate instance.
- `WEAVIATE_API_KEY`: The API key for Weaviate.

Add these to a `.env` file in your project:

```bash
WEAVIATE_URL=your-weaviate-instance-url
WEAVIATE_API_KEY=your-weaviate-api-key
```

The API will automatically check if the `Conversation` collection exists in Weaviate, and will create it if it doesn't.

## Key Dependencies

- **Flask**: A lightweight web framework for Python.
- **SpaCy**: A natural language processing library for entity recognition.
- **Sentence-Transformers**: A library for creating sentence embeddings.
- **Weaviate**: A vector database for storing and searching embeddings.
- **NMF & TF-IDF**: Models for topic modeling and keyword extraction.
- **NLTK**: A toolkit for working with text and natural language processing.

## Use Cases

- **Conversation Analysis**: Automatically analyze conversations and meetings to extract key information such as names, companies, and topics discussed.
- **Contextual Search**: Create a searchable knowledge base of conversations that can be queried by embedding similarity.
- **Summary Generation**: Automatically generate summaries by extracting named entities, keywords, and topics from conversations.

## Future Improvements

- **Enhanced Search Capabilities**: Add support for more advanced query options, including filtering by entities or topics.
- **Transformer-based NER**: Explore the use of transformer-based models for improved accuracy in entity recognition.
- **Customizable Embeddings**: Allow for more customization of the sentence embeddings used for vector storage in Weaviate.

---

This project is designed to help you quickly extract key information from conversations, making it easier to organize and retrieve relevant information from large sets of data.
