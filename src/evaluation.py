"""
===========================================================
Evaluation and Model Comparison

Project:
Explainable Label-Correlation Learning for
Multi-Label Requirement Smell Detection

Author:
Muhammad Arsalan
===========================================================
"""

from pathlib import Path

import pandas as pd


# ==========================================================
# Paths
# ==========================================================

METRICS_DIR = Path("outputs/metrics")

BR_RESULTS = METRICS_DIR / "baseline_results.csv"

CC_RESULTS = METRICS_DIR / "classifier_chain_results.csv"

OUTPUT_FILE = METRICS_DIR / "model_comparison.csv"


# ==========================================================
# Load Results
# ==========================================================

def load_results():

    br = pd.read_csv(BR_RESULTS)

    cc = pd.read_csv(CC_RESULTS)

    return br, cc


# ==========================================================
# Summary Statistics
# ==========================================================

def summarize(df):

    numeric = df.select_dtypes(include="number")

    
    numeric = numeric.drop(columns=["Fold"], errors="ignore")

    summary = pd.DataFrame({
        "Mean": numeric.mean(),
        "Std": numeric.std()
    })

    return summary


# ==========================================================
# Compare Models
# ==========================================================

def compare_models(br_summary, cc_summary):

    comparison = pd.DataFrame({

        "Binary Relevance": br_summary["Mean"],

        "Classifier Chains": cc_summary["Mean"]

    })

    return comparison


# ==========================================================
# Save Comparison
# ==========================================================

def save_results(comparison):

    comparison.to_csv(OUTPUT_FILE)

    print(f"\nComparison saved to:\n{OUTPUT_FILE}")


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    br, cc = load_results()

    br_summary = summarize(br)

    cc_summary = summarize(cc)

    print("\nBinary Relevance\n")
    print(br_summary)

    print("\nClassifier Chains\n")
    print(cc_summary)

    comparison = compare_models(
        br_summary,
        cc_summary
    )

    print("\nFinal Comparison\n")
    print(comparison)

    save_results(comparison)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()