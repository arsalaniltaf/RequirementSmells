"""
=============================================================
Classifier Chains Baseline
=============================================================

Project:
    Explainable Label-Correlation-Aware Multi-Label
    Requirement Smell Detection

Description:
    Classifier Chains using Logistic Regression with
    5-Fold Multilabel Stratified Cross Validation.

Author:
    Muhammad Arsalan
=============================================================
"""

from pathlib import Path
import re

import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain

from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    precision_score,
    recall_score,
    f1_score,
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
# Text Cleaning
# ==========================================================

def clean_text(text):
    """
    Basic preprocessing.
    """

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()

    return text


# ==========================================================
# Dataset Loading
# ==========================================================

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

    metrics = {

        "Subset Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "Hamming Loss": hamming_loss(
            y_true,
            y_pred,
        ),

        "Micro Precision": precision_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),

        "Micro Recall": recall_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),

        "Micro F1": f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),

        "Macro F1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }

    return metrics

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

        print("=" * 60)
        print(f"Fold {fold}")
        print("=" * 60)

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = Y.iloc[train_idx]
        y_test = Y.iloc[test_idx]

        # --------------------------------------------------
        # TF-IDF
        # --------------------------------------------------

        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
        )

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # --------------------------------------------------
        # Base Classifier
        # --------------------------------------------------

        base_model = LogisticRegression(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=1000,
        )

        # --------------------------------------------------
        # Classifier Chain
        # --------------------------------------------------

        try:
            # Newer versions of scikit-learn
            model = ClassifierChain(
                estimator=base_model,
                order="random",
                random_state=RANDOM_STATE,
            )
        except TypeError:
            # Older versions of scikit-learn
            model = ClassifierChain(
                base_estimator=base_model,
                order="random",
                random_state=RANDOM_STATE,
            )

        model.fit(X_train_tfidf, y_train)

        predictions = model.predict(X_test_tfidf)

        metrics = evaluate(
            y_test,
            predictions,
        )

        metrics["Fold"] = fold

        results.append(metrics)

        print(metrics)

        fold += 1

    results = pd.DataFrame(results)

    return results

# ==========================================================
# Save Results
# ==========================================================

def save_results(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save fold-wise results
    results.to_csv(
        OUTPUT_DIR / "classifier_chain_results.csv",
        index=False,
    )

    # Summary statistics
    summary = pd.DataFrame({
        "Mean": results.mean(numeric_only=True),
        "Std": results.std(numeric_only=True),
    })

    summary.to_csv(
        OUTPUT_DIR / "classifier_chain_summary.csv"
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("Classifier Chains")
    print("=" * 60)

    # Load dataset
    X, Y = load_dataset()

    print(f"\nDataset Loaded Successfully")
    print(f"Number of Requirements : {len(X)}")
    print(f"Number of Labels       : {Y.shape[1]}")

    # Run Cross Validation
    results = run_cross_validation(X, Y)

    # Print fold-wise results
    print("\n")
    print("=" * 60)
    print("Fold-wise Results")
    print("=" * 60)
    print(results)

    # Calculate summary statistics
    mean_results = results.mean(numeric_only=True)
    std_results = results.std(numeric_only=True)

    print("\n")
    print("=" * 60)
    print("Average Performance")
    print("=" * 60)
    print(mean_results)

    print("\n")
    print("=" * 60)
    print("Standard Deviation")
    print("=" * 60)
    print(std_results)

    # Save results
    save_results(results)

    print("\nResults saved successfully.")

    print("\nGenerated Files:")
    print("outputs/metrics/classifier_chain_results.csv")
    print("outputs/metrics/classifier_chain_summary.csv")


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()