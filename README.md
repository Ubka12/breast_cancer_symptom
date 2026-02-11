# Breast Symptom Checker

## Overview

This project is a Python-based web application that classifies short free-text breast symptom descriptions into LOW, MEDIUM, or HIGH risk categories.

The system was designed as a structured data processing pipeline that transforms unstructured natural language input into consistent, interpretable outputs.

The focus of the project was building a modular workflow that separates ingestion, transformation, validation, and structured output stages.

---

## Data Processing Pipeline

The system follows a multi-stage processing architecture:

### 1. Input Ingestion
User submits short free-text symptom description via web interface.

### 2. Preprocessing
Text is normalised (lowercasing, light cleaning) while preserving original phrasing.

### 3. Rule-Based Classification
A deterministic rule engine applies predefined severity weights and urgent overrides.

### 4. Semantic Similarity Matching
If no rule match is triggered, Sentence-BERT embeddings are generated and compared using cosine similarity against predefined symptom narratives.  
A similarity threshold (≥ 0.75) determines classification.

### 5. Structured Output
The system returns:
- Risk level (LOW / MEDIUM / HIGH)
- Classification method used
- Supporting rationale

This staged design ensures transparent, repeatable, and auditable data transformation.

---

## Technologies

- Python
- Flask (REST API)
- Sentence-BERT (Hugging Face Transformers)
- NumPy (cosine similarity computation)
- Virtual environments
- Git

---

## Running the Project

### Windows
py -m venv .venv
..venv\Scripts\Activate.ps1
pip install -r requirements.txt
python backend\prep_and_embed.py
python backend\app.py


### macOS / Linux

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python backend/prep_and_embed.py
python backend/app.py


Open: http://127.0.0.1:8000




