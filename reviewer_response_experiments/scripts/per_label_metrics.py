"""
Per-label Precision/Recall/F1 for Binary Relevance, using the EXACT same
CV protocol as src/baseline_binary_relevance.py (same TF-IDF settings,
same random_state=42, same MultilabelStratifiedKFold), to answer
Reviewer 1's request for per-label metrics rather than only aggregated
Micro/Macro.
"""
from pathlib import Path
import re

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

DATASET_PATH = Path("data/processed/arta_harmonized.csv")
TEXT_COLUMN = "requirement"
LABEL_COLUMNS = ["Subjective", "Ambiguous", "Nonverifiable", "Negative", "Vague"]
RANDOM_STATE = 42
N_SPLITS = 5
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)


def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).apply(clean_text)
    return df[TEXT_COLUMN], df[LABEL_COLUMNS]


def main():
    X, Y = load_dataset()
    cv = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    per_fold_per_label = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, Y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]

        vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
        X_train_v = vectorizer.fit_transform(X_train)
        X_test_v = vectorizer.transform(X_test)

        model = OneVsRestClassifier(
            LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000)
        )
        model.fit(X_train_v, y_train)
        preds = model.predict(X_test_v)

        p = precision_score(y_test, preds, average=None, zero_division=0)
        r = recall_score(y_test, preds, average=None, zero_division=0)
        f = f1_score(y_test, preds, average=None, zero_division=0)

        for i, label in enumerate(LABEL_COLUMNS):
            per_fold_per_label.append({
                "Fold": fold, "Label": label,
                "Precision": p[i], "Recall": r[i], "F1": f[i],
            })

    df = pd.DataFrame(per_fold_per_label)
    df.to_csv("outputs/per_label_metrics_all_folds.csv", index=False)

    summary = df.groupby("Label")[["Precision", "Recall", "F1"]].agg(["mean", "std"])
    summary = summary.reindex(LABEL_COLUMNS)
    summary.to_csv("outputs/per_label_metrics_summary.csv")

    print(summary.round(3))

    # simple mean-only version for a compact paper table
    compact = df.groupby("Label")[["Precision", "Recall", "F1"]].mean().reindex(LABEL_COLUMNS)
    compact.to_csv("outputs/per_label_metrics_compact.csv")
    print()
    print("Compact (mean only, for paper table):")
    print(compact.round(3))


if __name__ == "__main__":
    main()
