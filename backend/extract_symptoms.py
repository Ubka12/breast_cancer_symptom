# backend/extract_symptoms.py

import json
import os
import re
import nltk

# Download NLTK punkt if missing
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

# Input and output paths
DATA_PATH = os.path.join("data", "unified_dataset.json")
OUTPUT_PATH = os.path.join("data", "symptom_sentences.json")

# Symptom-related keywords (expand as needed)
SYMPTOM_KEYWORDS = [
    "lump", "pain", "ache", "sore", "discharge", "swelling",
    "tender", "redness", "itch", "rash", "bruise", "nipple",
    "inverted", "dimple", "thick", "spot", "mass", "change",
    "burn", "hard", "shrink", "size", "skin", "fluid",
    "bleed", "lesion", "weird", "unusual", "symptom"
]

def is_symptom_sentence(sentence):
    s = sentence.lower()
    return any(kw in s for kw in SYMPTOM_KEYWORDS)

def extract_symptom_sentences(text):
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    # Return sentences that contain likely symptoms
    return [s for s in sentences if is_symptom_sentence(s)]

def main():
    # Load unified dataset
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    result = []

    for entry in dataset:
        text = entry.get("text", "")
        if not text.strip():
            continue
        # Extract symptom sentences
        symptom_sents = extract_symptom_sentences(text)
        for sent in symptom_sents:
            result.append({
                "sentence": sent.strip(),
                "source_text": text[:80] + "..."  # Save a snippet of source for traceability
            })

    print(f"Extracted {len(result)} symptom sentences from {len(dataset)} entries.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    # Save results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
