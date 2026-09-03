"""
SHAP top-20 global feature classification: function/stopword vs.
content-bearing vs. mixed bigram, per requirement smell label.

Uses the NLTK English stopword list (not sklearn's default list, which
incorrectly flags domain content words like "system" and "call" as
stopwords for this dataset -- see write-up for details).

Reads: outputs/explainability/<Label>/top20_features.csv (from the
       existing project's offline SHAP analysis, one file per label)
Writes: outputs/shap_top20_classified_NLTK.csv (full classified list)
        printed summary counts/percentages per label
"""
from pathlib import Path

import pandas as pd
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

# Path to the existing project's SHAP explainability outputs
BASE = Path("/home/claude/project/RequirementSmells/outputs/explainability")
LABELS = ["Subjective", "Ambiguous", "Nonverifiable", "Negative", "Vague"]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def classify(feature: str) -> str:
    """
    Classify a TF-IDF feature (unigram or bigram) as:
      - "Function/Stopword": unigram in the stopword list, or bigram
        where BOTH tokens are stopwords
      - "Mixed (bigram)": bigram with exactly one stopword token
      - "Content": unigram not in the stopword list, or bigram with
        neither token a stopword
    """
    tokens = feature.split()
    if len(tokens) == 1:
        return "Function/Stopword" if tokens[0] in STOPWORDS else "Content"
    stop_flags = [t in STOPWORDS for t in tokens]
    if all(stop_flags):
        return "Function/Stopword"
    elif any(stop_flags):
        return "Mixed (bigram)"
    else:
        return "Content"


def main():
    all_rows = []
    for label in LABELS:
        df = pd.read_csv(BASE / label / "top20_features.csv")
        df["Label"] = label
        df["Class"] = df["Feature"].apply(classify)
        all_rows.append(df)

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(OUTPUT_DIR / "shap_top20_classified_NLTK.csv", index=False)

    summary = full.groupby(["Label", "Class"]).size().unstack(fill_value=0)
    for c in ["Content", "Function/Stopword", "Mixed (bigram)"]:
        if c not in summary.columns:
            summary[c] = 0
    summary = summary[["Content", "Function/Stopword", "Mixed (bigram)"]].reindex(LABELS)
    summary["Total"] = summary.sum(axis=1)

    print("Counts (out of 20 per label):")
    print(summary)
    print()
    pct = summary.div(summary["Total"], axis=0) * 100
    print("Percentages:")
    print(pct.round(1))
    print()

    overall = full["Class"].value_counts()
    print("Overall (100 features across 5 labels):")
    print(overall)
    print((overall / 100 * 100).round(1))

    print()
    for label in LABELS:
        content = full[(full.Label == label) & (full.Class == "Content")]["Feature"].tolist()
        print(f"{label} content features: {content}")


if __name__ == "__main__":
    main()
