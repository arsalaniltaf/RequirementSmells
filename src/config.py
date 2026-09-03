"""
===========================================================
Configuration File
Project:
Explainable Label-Correlation Learning for
Multi-Label Requirement Smell Detection

Author:
Muhammad Arsalan

Description:
Contains all project paths, dataset mappings,
column definitions, and constants used throughout
the project.
===========================================================
"""

from pathlib import Path

# =========================================================
# Project Directories
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

ORIGINAL_DATA_DIR = DATA_DIR / "original"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"

# =========================================================
# Original Dataset Paths
# =========================================================

DS1_PATH = ORIGINAL_DATA_DIR / "DS1.xlsx"
DS2_PATH = ORIGINAL_DATA_DIR / "DS2.xlsx"
DS3_PATH = ORIGINAL_DATA_DIR / "DS3.xlsx"
DS4_PATH = ORIGINAL_DATA_DIR / "DS4.xlsx"

# =========================================================
# Output Files
# =========================================================

RAW_MERGED_DATASET = INTERIM_DATA_DIR / "arta_merged_raw.csv"

HARMONIZED_DATASET = PROCESSED_DATA_DIR / "arta_harmonized.csv"

FINAL_DATASET = PROCESSED_DATA_DIR / "arta_final.csv"

# =========================================================
# Dataset Names
# =========================================================

DATASET_NAMES = {
    "DS1": "DS1",
    "DS2": "DS2",
    "DS3": "DS3",
    "DS4": "DS4"
}

# =========================================================
# Final Label Names
# =========================================================

TARGET_LABELS = [
    "Subjective",
    "Ambiguous",
    "Nonverifiable",
    "Negative",
    "Vague"
]

# =========================================================
# Final Dataset Columns
# =========================================================

FINAL_COLUMNS = [
    "requirement",
    "dataset",
    "source_file",
    *TARGET_LABELS
]

# =========================================================
# Column Mapping
# =========================================================

DS1_MAPPING = {

    "Requirement_text": "requirement",
    "File": "source_file",

    "Subjective_lang.": "Subjective",
    "Ambiguous_adv._adj.": "Ambiguous",
    "Nonverifiable_term": "Nonverifiable",
    "Negative": "Negative",
    "Vague_pron.": "Vague"
}

DS23_MAPPING = {

    "text": "requirement",
    "filename": "source_file",

    "subjective_language": "Subjective",
    "ambiguous_adverbs_adjectives": "Ambiguous",
    "open_ended": "Nonverifiable",
    "negative_statements": "Negative",
    "vague_pronouns": "Vague"
}

DS4_MAPPING = {

    "Requirement": "requirement",
    "File name": "source_file",

    "Subjective Language": "Subjective",
    "Ambiguouse_adverb_and_adjective": "Ambiguous",
    "Open ended NonVarifiable": "Nonverifiable",
    "negative_sentence": "Negative",
    "vague_pronuns": "Vague"
}

# =========================================================
# Directories to Create Automatically
# =========================================================

REQUIRED_DIRECTORIES = [
    ORIGINAL_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR
]