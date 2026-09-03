"""
=============================================================
Binary Relevance Baseline
=============================================================

Project:
    Explainable Label-Correlation-Aware Multi-Label
    Requirement Smell Detection

Description:
    Baseline implementation using Binary Relevance
    (One-vs-Rest Logistic Regression) with
    5-Fold Multilabel Stratified Cross Validation.

Author:
    Muhammad Arsalan
=============================================================
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
)


# ==========================================================
# Configuration
# ==========================================================

DATASET_PATH = Path("data/processed/arta_harmonized.csv")

OUTPUT_DIR = Path("outputs/metrics")

TEXT_COLUMN = "requirement"

LABEL_COLUMNS = [
    "Subjective",
    "Ambiguous",
    "Nonverifiable",
    "Negative",
    "Vague",
]

RANDOM_STATE = 42

N_SPLITS = 5

MAX_FEATURES = 5000

NGRAM_RANGE = (1, 2)


# ==========================================================
# Utility Functions
# ==========================================================

def clean_text(text: str) -> str:
    """Basic text preprocessing."""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()

    return text


def load_dataset():

    df = pd.read_csv(DATASET_PATH)

    df[TEXT_COLUMN] = (
        df[TEXT_COLUMN]
        .astype(str)
        .apply(clean_text)
    )

    X = df[TEXT_COLUMN]

    Y = df[LABEL_COLUMNS]

    return X, Y


# ==========================================================
# Evaluation
# ==========================================================

def evaluate(y_true, y_pred):

    return {

        "Subset Accuracy":
            accuracy_score(y_true, y_pred),

        "Hamming Loss":
            hamming_loss(y_true, y_pred),

        "Micro Precision":
            precision_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0
            ),

        "Micro Recall":
            recall_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0
            ),

        "Micro F1":
            f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0
            ),

        "Macro F1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),
    }


# ==========================================================
# Cross Validation
# ==========================================================

def run_cross_validation(X, Y):

    cv = MultilabelStratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    fold = 1

    for train_idx, test_idx in cv.split(X, Y):

        print(f"\nFold {fold}")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = Y.iloc[train_idx]
        y_test = Y.iloc[test_idx]

        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
        )

        X_train = vectorizer.fit_transform(X_train)

        X_test = vectorizer.transform(X_test)

        model = OneVsRestClassifier(
    	    LogisticRegression(
        	class_weight="balanced",
        	random_state=RANDOM_STATE,
        	max_iter=1000,
            )
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        metrics = evaluate(
            y_test,
            predictions,
        )

        metrics["Fold"] = fold

        results.append(metrics)

        fold += 1

    return pd.DataFrame(results)


# ==========================================================
# Save Results
# ==========================================================

def save_results(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(

        OUTPUT_DIR /
        "baseline_results.csv",

        index=False,
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("Binary Relevance Baseline")
    print("=" * 60)

    X, Y = load_dataset()

    results = run_cross_validation(X, Y)

    print("\n")

    print(results)

    print("\nAverage Performance")

    print(results.mean(numeric_only=True))

    save_results(results)

    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()