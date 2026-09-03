"""
===========================================================
Duplicate Analysis

Project:
Explainable Label-Correlation Learning for
Multi-Label Requirement Smell Detection

Author:
Muhammad Arsalan
===========================================================
"""

import pandas as pd

from config import *


def main():

    print("=" * 70)
    print("Duplicate Analysis")
    print("=" * 70)

    # Load raw merged dataset
    df = pd.read_csv(RAW_MERGED_DATASET)

    print(f"\nTotal rows: {len(df)}")

    # --------------------------------------------------
    # Exact duplicate requirements
    # --------------------------------------------------

    duplicates = df[df.duplicated(subset=["requirement"], keep=False)]

    print(f"Duplicate rows: {len(duplicates)}")

    print(f"Unique duplicate requirements: "
          f"{duplicates['requirement'].nunique()}")

    # --------------------------------------------------
    # Dataset distribution
    # --------------------------------------------------

    print("\nDataset distribution")

    print(df["dataset"].value_counts())

    # --------------------------------------------------
    # Duplicate distribution
    # --------------------------------------------------

    print("\nDuplicate rows by dataset")

    print(duplicates["dataset"].value_counts())

    # --------------------------------------------------
    # Save reports
    # --------------------------------------------------

    duplicates.sort_values(
        by=["requirement", "dataset"]
    ).to_csv(
        REPORTS_DIR / "duplicate_requirements.csv",
        index=False
    )

    summary = pd.DataFrame({

        "Metric": [

            "Total Rows",
            "Duplicate Rows",
            "Unique Duplicate Requirements"

        ],

        "Value": [

            len(df),
            len(duplicates),
            duplicates["requirement"].nunique()

        ]

    })

    summary.to_csv(

        REPORTS_DIR / "duplicate_summary.csv",

        index=False

    )

    print("\nReports saved successfully.")


if __name__ == "__main__":
    main()