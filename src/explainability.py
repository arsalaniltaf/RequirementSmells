"""
============================================================
SHAP Explainability for Requirement Smell Detection
============================================================

This script generates SHAP explanations for the trained
Binary Relevance (One-vs-Rest Logistic Regression) model.

Outputs
-------
outputs/explainability/

    Subjective/
        summary_plot.png
        top20_features.csv

    Ambiguous/
        summary_plot.png
        top20_features.csv

    Nonverifiable/
        summary_plot.png
        top20_features.csv

    Negative/
        summary_plot.png
        top20_features.csv

    Vague/
        summary_plot.png
        top20_features.csv

    all_labels_top_features.csv

Author:
Muhammad Arsalan
============================================================
"""

from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import shap

# ==========================================================
# Configuration
# ==========================================================

DATASET_PATH = Path(
    "data/processed/arta_harmonized.csv"
)

MODEL_PATH = Path(
    "outputs/models/binary_relevance_model.pkl"
)

VECTORIZER_PATH = Path(
    "outputs/models/tfidf_vectorizer.pkl"
)

OUTPUT_DIR = Path(
    "outputs/explainability"
)

TEXT_COLUMN = "requirement"

LABEL_COLUMNS = [
    "Subjective",
    "Ambiguous",
    "Nonverifiable",
    "Negative",
    "Vague",
]
# ==========================================================
# Runtime SHAP Cache
# ==========================================================

_RUNTIME_MODEL = None
_RUNTIME_VECTORIZER = None
_RUNTIME_FEATURE_NAMES = None
_RUNTIME_EXPLAINERS = None
TOP_FEATURES = 20

# ==========================================================
# Utility Functions
# ==========================================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def print_header():

    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY")
    print("=" * 60)


def create_output_directories():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for label in LABEL_COLUMNS:

        (
            OUTPUT_DIR /
            label
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    print("\nLoading harmonized dataset...")

    df = pd.read_csv(DATASET_PATH)

    df[TEXT_COLUMN] = (

        df[TEXT_COLUMN]

        .astype(str)

        .apply(clean_text)

    )

    print(f"Requirements : {len(df)}")

    return df

# ==========================================================
# Load Trained Model
# ==========================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(

            f"Model not found:\n{MODEL_PATH}"

        )


    with open(MODEL_PATH, "rb") as file:

        model = pickle.load(file)


    return model

# ==========================================================
# Load TF-IDF Vectorizer
# ==========================================================

def load_vectorizer():

    if not VECTORIZER_PATH.exists():

        raise FileNotFoundError(

            f"Vectorizer not found:\n{VECTORIZER_PATH}"

        )


    with open(VECTORIZER_PATH, "rb") as file:

        vectorizer = pickle.load(file)


    return vectorizer

# ==========================================================
# Prepare Features
# ==========================================================

def prepare_features(df, vectorizer):

    print("\nPreparing TF-IDF features...")

    X = vectorizer.transform(
        df[TEXT_COLUMN]
    )

    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    print(f"Feature matrix shape: {X.shape}")

    return X, feature_names

# ==========================================================
# Compute SHAP Feature Importance
# ==========================================================

def compute_feature_importance(
    shap_values,
    feature_names,
):

    importance = np.abs(
        shap_values.values
    ).mean(axis=0)

    feature_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance

    })

    feature_df = feature_df.sort_values(

        by="Importance",

        ascending=False,

    )

    return feature_df.head(TOP_FEATURES)

# ==========================================================
# Explain One Requirement Smell
# ==========================================================

def explain_label(

    classifier,

    X,

    feature_names,

    label_name,

):

    print(f"\nExplaining {label_name}...")

    # Convert sparse matrix to dense

    X_dense = X.toarray()

    # SHAP Linear Explainer

    explainer = shap.LinearExplainer(

        classifier,

        X_dense,

    )

    shap_values = explainer(X_dense)

    # --------------------------------------------------
    # Summary Plot
    # --------------------------------------------------

    plt.figure(figsize=(12, 7))

    shap.summary_plot(

        shap_values.values,

        X_dense,

        feature_names=feature_names,

        max_display=20,

        show=False,

    )

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR /

        label_name /

        "summary_plot.png",

        dpi=300,

        bbox_inches="tight",

    )

    plt.close()

    # --------------------------------------------------
    # Top Features
    # --------------------------------------------------

    feature_df = compute_feature_importance(

        shap_values,

        feature_names,

    )

    feature_df.to_csv(

        OUTPUT_DIR /

        label_name /

        "top20_features.csv",

        index=False,

    )

    print("Completed.")

    return feature_df

# ==========================================================
# Explain All Labels
# ==========================================================

def explain_all_labels(

    model,

    X,

    feature_names,

):

    summary_tables = []

    for label_name, classifier in zip(

        LABEL_COLUMNS,

        model.estimators_,

    ):

        table = explain_label(

            classifier,

            X,

            feature_names,

            label_name,

        )

        table.insert(

            0,

            "Label",

            label_name,

        )

        summary_tables.append(table)

    summary_df = pd.concat(

        summary_tables,

        ignore_index=True,

    )

    return summary_df

# ==========================================================
# Save Summary Table
# ==========================================================

def save_summary_table(summary_df):

    output_file = OUTPUT_DIR / "all_labels_top_features.csv"

    summary_df.to_csv(
        output_file,
        index=False,
    )

    print(f"\nSummary saved to: {output_file}")

def initialize_runtime_explainers(
    background_size=100,
):
    """
    Initializes one SHAP explainer per Binary Relevance classifier.

    This function is called only once and reused for all
    runtime explanations.
    """

    global _RUNTIME_MODEL
    global _RUNTIME_VECTORIZER
    global _RUNTIME_FEATURE_NAMES
    global _RUNTIME_EXPLAINERS

    # Already initialized
    if _RUNTIME_EXPLAINERS is not None:
        return

    print("Initializing runtime SHAP explainers...")

    _RUNTIME_MODEL = load_model()

    _RUNTIME_VECTORIZER = load_vectorizer()

    df = load_dataset()

    background_texts = (
        df.sample(
            n=min(background_size, len(df)),
            random_state=42,
        )[TEXT_COLUMN]
    )

    background = _RUNTIME_VECTORIZER.transform(
        background_texts
    ).toarray()

    _RUNTIME_FEATURE_NAMES = np.array(
        _RUNTIME_VECTORIZER.get_feature_names_out()
    )

    _RUNTIME_EXPLAINERS = {}

    for label_name, classifier in zip(
        LABEL_COLUMNS,
        _RUNTIME_MODEL.estimators_,
    ):

        _RUNTIME_EXPLAINERS[label_name] = shap.LinearExplainer(
            classifier,
            background,
        )

    print("Runtime explainers ready.")

# ==========================================================
# Explain a Single Requirement
# ==========================================================

def get_shap_keywords(
    requirement,
    top_k=5,
):
    """
    Returns the top SHAP keywords for each predicted label.
    """

    initialize_runtime_explainers()

    requirement = clean_text(requirement)

    X = _RUNTIME_VECTORIZER.transform(
        [requirement]
    )

    X_dense = X.toarray()

    tfidf_values = X_dense[0]

    nonzero_indices = np.where(
        tfidf_values > 0
    )[0]

    if len(nonzero_indices) == 0:
        return {
            label: []
            for label in LABEL_COLUMNS
        }

    shap_keywords = {}

    for label_name in LABEL_COLUMNS:

        explainer = _RUNTIME_EXPLAINERS[label_name]

        shap_values = explainer(
            X_dense
        )

        importance = np.abs(
            shap_values.values[0]
        )

        filtered_importance = importance[
            nonzero_indices
        ]

        ranked = nonzero_indices[
            np.argsort(
                filtered_importance
            )[::-1]
        ]

        keywords = _RUNTIME_FEATURE_NAMES[
            ranked[:top_k]
        ]

        shap_keywords[label_name] = (
            keywords.tolist()
        )

    return shap_keywords

# ==========================================================
# Main
# ==========================================================

def main():

    print_header()

    create_output_directories()

    # Load resources
    df = load_dataset()

    model = load_model()

    vectorizer = load_vectorizer()

    # Prepare TF-IDF features
    X, feature_names = prepare_features(
        df,
        vectorizer,
    )

    # Generate explanations
    summary_df = explain_all_labels(
        model,
        X,
        feature_names,
    )

    # Save combined results
    save_summary_table(summary_df)

    print("\n" + "=" * 60)
    print("Explainability completed successfully.")
    print("=" * 60)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()

