"""
=============================================================
Preprocessing Module
=============================================================

Project:
    Explainable Label-Correlation-Aware Multi-Label
    Requirement Smell Detection

Description:
    Loads the harmonized dataset, validates its structure,
    performs lightweight text preprocessing, extracts the
    requirement text and smell labels, and saves the cleaned
    dataset for downstream experiments.

Author:
    Muhammad Arsalan
=============================================================
"""

from pathlib import Path
import re

import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = Path("data/processed/arta_harmonized.csv")

OUTPUT_PATH = Path("data/interim/arta_cleaned.csv")

TEXT_COLUMN = "requirement"

METADATA_COLUMNS = [
    "dataset",
    "source_file",
]

LABEL_COLUMNS = [
    "Subjective",
    "Ambiguous",
    "Nonverifiable",
    "Negative",
    "Vague",
]


# ============================================================
# Load Dataset
# ============================================================

def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Load the harmonized dataset.

    Parameters
    ----------
    dataset_path : Path
        Path to the harmonized dataset.

    Returns
    -------
    pd.DataFrame
    """

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{dataset_path}"
        )

    return pd.read_csv(dataset_path)


# ============================================================
# Validate Dataset
# ============================================================

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate dataset structure.
    """

    required_columns = (
        [TEXT_COLUMN]
        + METADATA_COLUMNS
        + LABEL_COLUMNS
    )

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns:\n{missing}"
        )

    if df[TEXT_COLUMN].isna().any():
        raise ValueError(
            "Requirement column contains missing values."
        )

    print("✓ Dataset validation passed.")


# ============================================================
# Text Preprocessing
# ============================================================

def clean_text(text: str) -> str:
    """
    Perform lightweight text preprocessing.

    Operations
    ----------
    1. Remove leading/trailing whitespace
    2. Collapse multiple spaces
    3. Convert to lowercase

    Notes
    -----
    We intentionally do NOT remove punctuation,
    stopwords, or perform stemming/lemmatization.
    """

    text = text.strip()

    text = re.sub(r"\s+", " ", text)

    text = text.lower()

    return text


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to all requirements.
    """

    df = df.copy()

    df[TEXT_COLUMN] = (
        df[TEXT_COLUMN]
        .astype(str)
        .apply(clean_text)
    )

    print("✓ Text preprocessing completed.")

    return df


# ============================================================
# Prepare Data
# ============================================================

def prepare_features_and_labels(df: pd.DataFrame):
    """
    Separate requirement text and smell labels.

    Returns
    -------
    X : pandas.Series
        Requirement text.

    Y : pandas.DataFrame
        Multi-label targets.
    """

    X = df[TEXT_COLUMN]

    Y = df[LABEL_COLUMNS]

    return X, Y


# ============================================================
# Save Cleaned Dataset
# ============================================================

def save_dataset(df: pd.DataFrame,
                 output_path: Path) -> None:
    """
    Save cleaned dataset.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(f"✓ Cleaned dataset saved to:\n{output_path}")


# ============================================================
# Summary
# ============================================================

def print_summary(df: pd.DataFrame):
    """
    Print dataset summary.
    """

    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    print(f"Requirements : {len(df)}")
    print(f"Labels       : {len(LABEL_COLUMNS)}")

    print("\nPositive label counts")

    for label in LABEL_COLUMNS:
        print(
            f"{label:<20}"
            f"{int(df[label].sum())}"
        )

    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Requirement Smell Detection")
    print("Preprocessing")
    print("=" * 60)

    print("\nLoading dataset...")

    df = load_dataset(DATASET_PATH)

    validate_dataset(df)

    df = preprocess_text(df)

    X, Y = prepare_features_and_labels(df)

    save_dataset(df, OUTPUT_PATH)

    print_summary(df)

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()