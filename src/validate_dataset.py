"""
===========================================================
Dataset Validation

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
    print("Dataset Validation")
    print("=" * 70)

    df = pd.read_csv(HARMONIZED_DATASET)

    print(f"\nDataset Size: {len(df)}")

    # ----------------------------------------------------
    # Missing Values
    # ----------------------------------------------------

    print("\nChecking Missing Values...")

    missing = df.isnull().sum()

    if missing.sum() == 0:
        print("PASS ✓ No missing values found.")
    else:
        print("FAIL ✗ Missing values detected.")
        print(missing)

    # ----------------------------------------------------
    # Empty Requirements
    # ----------------------------------------------------

    print("\nChecking Empty Requirements...")

    empty = (df["requirement"].astype(str).str.strip() == "").sum()

    if empty == 0:
        print("PASS ✓ No empty requirements.")
    else:
        print(f"FAIL ✗ {empty} empty requirements found.")

    # ----------------------------------------------------
    # Duplicate Requirements
    # ----------------------------------------------------

    print("\nChecking Duplicate Requirements...")

    duplicates = df.duplicated(subset=["requirement"]).sum()

    if duplicates == 0:
        print("PASS ✓ No duplicate requirements.")
    else:
        print(f"FAIL ✗ {duplicates} duplicate requirements found.")

    # ----------------------------------------------------
    # Binary Labels
    # ----------------------------------------------------

    print("\nChecking Binary Labels...")

    binary_ok = True

    for label in TARGET_LABELS:

        values = set(df[label].unique())

        if not values.issubset({0, 1}):

            binary_ok = False

            print(f"{label}: Invalid values -> {values}")

    if binary_ok:
        print("PASS ✓ All labels are binary (0/1).")
    else:
        print("FAIL ✗ Non-binary labels detected.")

    # ----------------------------------------------------
    # Requirement Length
    # ----------------------------------------------------

    print("\nRequirement Length Statistics")

    lengths = df["requirement"].str.split().str.len()

    print(f"Minimum Words : {lengths.min()}")
    print(f"Maximum Words : {lengths.max()}")
    print(f"Average Words : {lengths.mean():.2f}")

    # ----------------------------------------------------
    # Label Summary
    # ----------------------------------------------------

    print("\nLabel Summary")

    label_count = df[TARGET_LABELS].sum(axis=1)

    no_label = (label_count == 0).sum()
    single = (label_count == 1).sum()
    multi = (label_count > 1).sum()

    print(f"No Label     : {no_label}")
    print(f"Single Label : {single}")
    print(f"Multi Label  : {multi}")

    # ----------------------------------------------------
    # Final Status
    # ----------------------------------------------------

    print("\n" + "=" * 70)
    print("Validation Completed")
    print("=" * 70)

    if (
        missing.sum() == 0 and
        empty == 0 and
        duplicates == 0 and
        binary_ok
    ):
        print("\nDATASET STATUS: PASS ✓")
        print("Dataset is ready for machine learning.")
    else:
        print("\nDATASET STATUS: REVIEW REQUIRED")


if __name__ == "__main__":
    main()