"""
===========================================================
Utility Functions

Project:
Explainable Label-Correlation Learning for
Multi-Label Requirement Smell Detection

Author:
Muhammad Arsalan
===========================================================
"""

import json
import random

import numpy as np
import pandas as pd

from config import *


# ==========================================================
# Random Seed
# ==========================================================

RANDOM_SEED = 42


def set_random_seed(seed=RANDOM_SEED):
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)


# ==========================================================
# Dataset Loading
# ==========================================================

def load_dataset():
    """
    Load the harmonized dataset.
    """

    return pd.read_csv(HARMONIZED_DATASET)


# ==========================================================
# Save DataFrame
# ==========================================================

def save_dataframe(df, filename):
    """
    Save a dataframe into the reports directory.
    """

    path = REPORTS_DIR / filename

    df.to_csv(path, index=False)

    print(f"Saved: {path}")


# ==========================================================
# Save JSON
# ==========================================================

def save_json(data, filename):
    """
    Save dictionary as JSON.
    """

    path = REPORTS_DIR / filename

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved: {path}")


# ==========================================================
# Console Header
# ==========================================================

def print_header(title):
    """
    Print a formatted console header.
    """

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ==========================================================
# Label Statistics
# ==========================================================

def label_cardinality(df):
    """
    Average number of labels per requirement.
    """

    return df[TARGET_LABELS].sum(axis=1).mean()


def label_density(df):
    """
    Average proportion of labels per requirement.
    """

    return label_cardinality(df) / len(TARGET_LABELS)


# ==========================================================
# Dataset Summary
# ==========================================================

def dataset_summary(df):
    """
    Print basic dataset information.
    """

    print(f"Requirements : {len(df)}")
    print(f"Labels       : {len(TARGET_LABELS)}")
    print(f"Cardinality  : {label_cardinality(df):.3f}")
    print(f"Density      : {label_density(df):.3f}")