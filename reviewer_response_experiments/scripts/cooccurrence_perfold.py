"""
Corrected co-occurrence-informed Classifier Chains run.

Unlike the first pass, the label order here is derived ONLY from each
training fold's labels (never from the test fold), so there is no leakage
from the co-occurrence matrix into the held-out data.

Rule (stated explicitly for the paper):
  For each training fold, compute the pairwise label co-occurrence matrix
  over y_train only. For each label, sum its co-occurrence counts with all
  other labels. Order labels in DESCENDING order of this total
  co-occurrence sum (ties broken by descending raw label frequency in
  y_train, then alphabetically). Labels most entangled with others are
  chained first.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import (
    accuracy_score, hamming_loss, precision_score, recall_score, f1_score
)

DATASET_PATH = Path("data/processed/arta_harmonized.csv")
OUTPUT_DIR = Path("outputs")

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


def evaluate(y_true, y_pred):
    return {
        "Subset Accuracy": accuracy_score(y_true, y_pred),
        "Hamming Loss": hamming_loss(y_true, y_pred),
        "Micro Precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "Micro Recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "Micro F1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def make_base_model():
    return LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000)


def cooccurrence_order_from_training(y_train):
    """Derive chain order strictly from training-fold labels."""
    cols = LABEL_COLUMNS
    n = len(cols)
    co = np.zeros((n, n), dtype=int)
    Y = y_train.values
    for i in range(n):
        for j in range(n):
            if i != j:
                co[i, j] = int(((Y[:, i] == 1) & (Y[:, j] == 1)).sum())
    total_co = co.sum(axis=1)
    freq = Y.sum(axis=0)

    # sort by (co-occurrence sum desc, frequency desc, name asc)
    order_idx = sorted(
        range(n),
        key=lambda i: (-total_co[i], -freq[i], cols[i]),
    )
    return order_idx, total_co, freq


def main():
    X, Y = load_dataset()
    cv = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    order_log = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, Y), start=1):
        X_train_txt, X_test_txt = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]

        vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
        X_train = vectorizer.fit_transform(X_train_txt)
        X_test = vectorizer.transform(X_test_txt)

        order_idx, total_co, freq = cooccurrence_order_from_training(y_train)
        order_labels = [LABEL_COLUMNS[i] for i in order_idx]

        model = ClassifierChain(estimator=make_base_model(), order=order_idx)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        m = evaluate(y_test, preds)
        m["Fold"] = fold_i
        m["Order"] = str(order_idx)
        m["OrderLabels"] = str(order_labels)
        rows.append(m)

        order_log.append({
            "Fold": fold_i,
            "OrderLabels": str(order_labels),
            "TotalCooccurrence": str(dict(zip(LABEL_COLUMNS, total_co.tolist()))),
            "TrainFrequency": str(dict(zip(LABEL_COLUMNS, freq.tolist()))),
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_DIR / "cc_cooccurrence_informed_PERFOLD_results.csv", index=False)

    order_log_df = pd.DataFrame(order_log)
    order_log_df.to_csv(OUTPUT_DIR / "cc_cooccurrence_informed_PERFOLD_orders.csv", index=False)

    metrics = ["Subset Accuracy","Hamming Loss","Micro Precision","Micro Recall","Micro F1","Macro F1"]
    summary = pd.DataFrame({
        "Mean": results_df[metrics].mean(),
        "Std": results_df[metrics].std(),
    })
    summary.to_csv(OUTPUT_DIR / "cc_cooccurrence_informed_PERFOLD_summary.csv")

    print("Per-fold chain orders (derived from training data only):")
    for row in order_log:
        print(f"  Fold {row['Fold']}: {row['OrderLabels']}")
    print()
    print("Summary (mean ± std across 5 folds):")
    print(summary)


if __name__ == "__main__":
    main()
