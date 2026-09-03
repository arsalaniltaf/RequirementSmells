"""
=============================================================
Requirement Smell Prediction
=============================================================

This script loads the trained Binary Relevance model and
TF-IDF vectorizer, predicts requirement smells for a new
requirement, and returns the predicted smell labels.

Author:
    Muhammad Arsalan
=============================================================
"""

from pathlib import Path
import pickle
import re

# ==========================================================
# Configuration
# ==========================================================

MODEL_PATH = Path("outputs/models/binary_relevance_model.pkl")
VECTORIZER_PATH = Path("outputs/models/tfidf_vectorizer.pkl")

LABELS = [
    "Subjective",
    "Ambiguous",
    "Nonverifiable",
    "Negative",
    "Vague",
]

# ==========================================================
# Text Cleaning (Same as Training)
# ==========================================================

def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    return text

# ==========================================================
# Load Model
# ==========================================================

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# ==========================================================
# Prediction Function
# ==========================================================

def predict_requirement(requirement):
    """
    Predict requirement smells and return:
    - results: Dictionary of predicted labels
    - probabilities: Dictionary of prediction probabilities
    """

    requirement = clean_text(requirement)

    X = vectorizer.transform([requirement])

    # Predict labels
    raw_prediction = model.predict(X)
    prediction = raw_prediction[0]

    # Calculate probabilities
    probabilities = {}

    for label, estimator in zip(LABELS, model.estimators_):
        probability = estimator.predict_proba(X)[0][1]
        probabilities[label] = probability

    # Convert predictions to dictionary
    results = {}

    for label, value in zip(LABELS, prediction):
        results[label] = bool(value)

    return results, probabilities

# ==========================================================
# Pretty Print
# ==========================================================

def print_results(results):

    print("\nDetected Requirement Smells")
    print("------------------------------------------------------------")

    detected = False

    for label, value in results.items():

        if value:
            print(f"✓ {label}")
            detected = True

    if not detected:
        print("No requirement smells detected.")
# ==========================================================
# Get Predicted Smell Labels
# ==========================================================

def get_predicted_smells(requirement):
    """
    Returns only the predicted smell names.
    """

    results, _ = predict_requirement(requirement)

    predicted_smells = [
        label
        for label, detected in results.items()
        if detected
    ]

    return predicted_smells
# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("Requirement Smell Detection")
    print("=" * 60)

    requirement = input("\nEnter Requirement:\n\n")

    results, _ = predict_requirement(requirement)

    print_results(results)

if __name__ == "__main__":
    main()