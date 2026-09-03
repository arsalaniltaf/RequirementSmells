"""
=============================================================
Train Final Binary Relevance Model
=============================================================

This script trains the final Binary Relevance model
on the complete harmonized dataset.

The trained model is ONLY used for:
    • SHAP Explainability
    • Revision Suggestion Module

It is NOT used for reporting experimental results.

Author:
    Muhammad Arsalan
=============================================================
"""

from pathlib import Path
import pickle
import re

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


# ==========================================================
# Configuration
# ==========================================================

DATASET_PATH = Path("data/processed/arta_harmonized.csv")

MODEL_DIR = Path("outputs/models")

TEXT_COLUMN = "requirement"

LABEL_COLUMNS = [
    "Subjective",
    "Ambiguous",
    "Nonverifiable",
    "Negative",
    "Vague",
]

MAX_FEATURES = 5000

NGRAM_RANGE = (1, 2)

RANDOM_STATE = 42


# ==========================================================
# Text Cleaning
# ==========================================================

def clean_text(text):

    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()

    return text


# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(DATASET_PATH)

df[TEXT_COLUMN] = (
    df[TEXT_COLUMN]
    .astype(str)
    .apply(clean_text)
)

X = df[TEXT_COLUMN]

Y = df[LABEL_COLUMNS]


# ==========================================================
# TF-IDF
# ==========================================================

vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE,
)

X_vectorized = vectorizer.fit_transform(X)


# ==========================================================
# Train Binary Relevance
# ==========================================================

model = OneVsRestClassifier(
    LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
)

model.fit(X_vectorized, Y)


# ==========================================================
# Save
# ==========================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    MODEL_DIR / "binary_relevance_model.pkl",
    "wb",
) as f:

    pickle.dump(model, f)

with open(
    MODEL_DIR / "tfidf_vectorizer.pkl",
    "wb",
) as f:

    pickle.dump(vectorizer, f)

print("=" * 60)
print("Final model saved successfully.")
print("=" * 60)