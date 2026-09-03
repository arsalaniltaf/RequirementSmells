"""
===========================================================
Dataset Statistics Generator

Project:
Explainable Label-Correlation Learning for
Multi-Label Requirement Smell Detection

Author:
Muhammad Arsalan
===========================================================
"""

import json
import pandas as pd

from config import *


def label_cardinality(df, labels):
    """Average number of labels per requirement"""
    return df[labels].sum(axis=1).mean()


def label_density(df, labels):
    """Average proportion of labels per requirement"""
    return label_cardinality(df, labels) / len(labels)


def main():

    print("=" * 70)
    print("Generating Dataset Statistics")
    print("=" * 70)

    df = pd.read_csv(HARMONIZED_DATASET)

    print(f"\nDataset Size : {len(df)}")

    print("\nLabel Frequencies")
    print("-" * 40)

    frequencies = df[TARGET_LABELS].sum().sort_values(ascending=False)

    percentages = (
        frequencies / len(df) * 100
    ).round(2)

    label_df = pd.DataFrame({

        "Label": frequencies.index,
        "Frequency": frequencies.values,
        "Percentage": percentages.values

    })

    print(label_df)

    # ---------------------------------------------------
    # Dataset Distribution
    # ---------------------------------------------------

    dataset_dist = (
        df["dataset"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    dataset_dist.columns = ["Dataset", "Requirements"]

    print("\nDataset Distribution")
    print("-" * 40)
    print(dataset_dist)

    # ---------------------------------------------------
    # Multi-label statistics
    # ---------------------------------------------------

    labels_per_requirement = df[TARGET_LABELS].sum(axis=1)

    multilabel = pd.DataFrame({

        "Metric": [

            "Total Requirements",
            "Single Label",
            "Multi Label",
            "Maximum Labels",
            "Average Labels",
            "Label Cardinality",
            "Label Density"

        ],

        "Value": [

            len(df),
            (labels_per_requirement == 1).sum(),
            (labels_per_requirement > 1).sum(),
            labels_per_requirement.max(),
            round(labels_per_requirement.mean(), 3),
            round(label_cardinality(df, TARGET_LABELS), 3),
            round(label_density(df, TARGET_LABELS), 3)

        ]

    })

    print("\nMulti-label Statistics")
    print("-" * 40)
    print(multilabel)

    # ---------------------------------------------------
    # Co-occurrence Matrix
    # ---------------------------------------------------

    cooccurrence = pd.DataFrame(
        index=TARGET_LABELS,
        columns=TARGET_LABELS,
        dtype=int
    )

    for l1 in TARGET_LABELS:
        for l2 in TARGET_LABELS:
            cooccurrence.loc[l1, l2] = (
                (df[l1] == 1) &
                (df[l2] == 1)
            ).sum()

    # ---------------------------------------------------
    # Save reports
    # ---------------------------------------------------

    label_df.to_csv(
        REPORTS_DIR / "label_distribution.csv",
        index=False
    )

    dataset_dist.to_csv(
        REPORTS_DIR / "dataset_distribution.csv",
        index=False
    )

    multilabel.to_csv(
        REPORTS_DIR / "multilabel_statistics.csv",
        index=False
    )

    cooccurrence.to_csv(
        REPORTS_DIR / "cooccurrence_matrix.csv"
    )

    summary = {

        "requirements": int(len(df)),
        "labels": len(TARGET_LABELS),
        "label_cardinality":
            round(label_cardinality(df, TARGET_LABELS), 3),
        "label_density":
            round(label_density(df, TARGET_LABELS), 3)

    }

    with open(
        REPORTS_DIR / "dataset_report.json",
        "w"
    ) as f:

        json.dump(summary, f, indent=4)

    print("\nAll reports generated successfully.")

    print("\nSaved to:")

    print(REPORTS_DIR)


if __name__ == "__main__":
    main()