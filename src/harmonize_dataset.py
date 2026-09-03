"""
===========================================================
Dataset Harmonization

Project:
Explainable Label-Correlation Learning for
Multi-Label Requirement Smell Detection

Author:
Muhammad Arsalan
===========================================================
"""

from pathlib import Path
import pandas as pd

from config import *


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load an Excel dataset.
    """

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found:\n{path}")

    return pd.read_excel(path)


def harmonize_dataset(
    df: pd.DataFrame,
    mapping: dict,
    dataset_name: str
) -> pd.DataFrame:
    """
    Harmonize a single dataset.
    """

    # Rename columns
    df = df.rename(columns=mapping)

    # Add dataset identifier
    df["dataset"] = dataset_name

    # Keep only required columns
    df = df[
        [
            "requirement",
            "dataset",
            "source_file",
            "Subjective",
            "Ambiguous",
            "Nonverifiable",
            "Negative",
            "Vague",
        ]
    ]

    return df


def convert_labels_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert smell annotations into binary labels.

    Any value different from "-" becomes 1.
    "-" becomes 0.
    """

    for label in TARGET_LABELS:

        df[label] = (
            df[label]
            .fillna("-")
            .astype(str)
            .str.strip()
            .apply(lambda x: 0 if x == "-" else 1)
        )

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove empty requirements and duplicates.
    """

    # Remove missing requirements
    df = df.dropna(subset=["requirement"])

    # Remove blank requirements
    df["requirement"] = df["requirement"].astype(str).str.strip()

    df = df[df["requirement"] != ""]

    # Remove duplicate requirements
    df = df.drop_duplicates(subset=["requirement"])

    return df.reset_index(drop=True)


def main():

    print("=" * 70)
    print("ARTA Dataset Harmonization")
    print("=" * 70)

    ds1 = harmonize_dataset(
        load_dataset(DS1_PATH),
        DS1_MAPPING,
        "DS1"
    )

    ds2 = harmonize_dataset(
        load_dataset(DS2_PATH),
        DS23_MAPPING,
        "DS2"
    )

    ds3 = harmonize_dataset(
        load_dataset(DS3_PATH),
        DS23_MAPPING,
        "DS3"
    )

    ds4 = harmonize_dataset(
        load_dataset(DS4_PATH),
        DS4_MAPPING,
        "DS4"
    )

    # Merge all datasets
    merged = pd.concat(
        [ds1, ds2, ds3, ds4],
        ignore_index=True
    )

    print(f"\nTotal rows before cleaning : {len(merged)}")

    # Save raw merged dataset
    merged.to_csv(RAW_MERGED_DATASET, index=False)

    # Convert labels
    merged = convert_labels_to_binary(merged)

    # Clean dataset
    merged = clean_dataset(merged)

    print(f"Total rows after cleaning  : {len(merged)}")

    # Save harmonized dataset
    merged.to_csv(HARMONIZED_DATASET, index=False)

    print("\nDataset successfully saved.")

    print(f"\nRaw Dataset:")
    print(RAW_MERGED_DATASET)

    print(f"\nHarmonized Dataset:")
    print(HARMONIZED_DATASET)

    print("\nPreview:\n")
    print(merged.head())


if __name__ == "__main__":
    main()