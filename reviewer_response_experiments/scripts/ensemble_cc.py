"""
Ensemble / co-occurrence-informed Classifier Chains experiment.

Reuses the exact same CV protocol, TF-IDF settings, and base learner as the
original classifier_chains.py / baseline_binary_relevance.py, and adds:
  1. 10 different chain orders per fold (50 total CC fits), averaged, to
     remove single-order dependence as a confound.
  2. One co-occurrence-informed chain order (from Table IV of the paper),
     run across the same 5 folds, as a secondary illustrative comparison.

Outputs written to outputs/ in this directory.
"""

from pathlib import Path
import re
import json

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
OUTPUT_DIR.mkdir(exist_ok=True)

TEXT_COLUMN = "requirement"
LABEL_COLUMNS = ["Subjective", "Ambiguous", "Nonverifiable", "Negative", "Vague"]

RANDOM_STATE = 42       # CV split seed -- IDENTICAL to original paper's protocol
N_SPLITS = 5
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
N_RANDOM_ORDERS = 10    # ensemble size


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


def fit_chain(X_train, y_train, X_test, y_test, order):
    """order: 'random' with a given random_state, or an explicit list/array of label indices."""
    kwargs = dict(estimator=make_base_model())
    if isinstance(order, str) and order == "random":
        raise ValueError("pass an explicit random_state via order_seed, not this path")
    model = ClassifierChain(order=order, random_state=RANDOM_STATE, **kwargs)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return evaluate(y_test, preds), getattr(model, "order_", order)


def main():
    X, Y = load_dataset()
    label_index = {name: i for i, name in enumerate(LABEL_COLUMNS)}

    cv = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X, Y))

    # ------------------------------------------------------------------
    # Pre-fit TF-IDF per fold once, reuse across all order variants
    # ------------------------------------------------------------------
    fold_data = []
    for train_idx, test_idx in folds:
        X_train_txt, X_test_txt = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]
        vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
        X_train = vectorizer.fit_transform(X_train_txt)
        X_test = vectorizer.transform(X_test_txt)
        fold_data.append((X_train, y_train, X_test, y_test))

    # ------------------------------------------------------------------
    # Original single fixed-order replication (order_seed=42), for
    # sanity-check against the paper's reported classifier_chain_summary.csv
    # ------------------------------------------------------------------
    original_rows = []
    for fold_i, (X_train, y_train, X_test, y_test) in enumerate(fold_data, start=1):
        model = ClassifierChain(estimator=make_base_model(), order="random", random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        m = evaluate(y_test, preds)
        m["Fold"] = fold_i
        m["Order"] = str(model.order_.tolist())
        original_rows.append(m)
    original_df = pd.DataFrame(original_rows)
    original_df.to_csv(OUTPUT_DIR / "cc_original_single_order_replication.csv", index=False)

    # ------------------------------------------------------------------
    # Ensemble of N_RANDOM_ORDERS distinct random orders x 5 folds
    # ------------------------------------------------------------------
    rng = np.random.default_rng(123)
    order_seeds = rng.integers(0, 100000, size=N_RANDOM_ORDERS).tolist()

    ensemble_rows = []
    for order_i, seed in enumerate(order_seeds, start=1):
        # derive one fixed permutation for this "order" from the seed,
        # apply it consistently across all 5 folds
        perm_rng = np.random.default_rng(seed)
        order = perm_rng.permutation(len(LABEL_COLUMNS))
        for fold_i, (X_train, y_train, X_test, y_test) in enumerate(fold_data, start=1):
            model = ClassifierChain(estimator=make_base_model(), order=order.tolist())
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            m = evaluate(y_test, preds)
            m["Fold"] = fold_i
            m["OrderIndex"] = order_i
            m["Order"] = str(order.tolist())
            m["OrderLabels"] = str([LABEL_COLUMNS[i] for i in order])
            ensemble_rows.append(m)

    ensemble_df = pd.DataFrame(ensemble_rows)
    ensemble_df.to_csv(OUTPUT_DIR / "cc_ensemble_all_runs.csv", index=False)

    # per-order-averaged-across-folds, then averaged across orders (+ std across orders)
    per_order_mean = ensemble_df.groupby("OrderIndex")[
        ["Subset Accuracy", "Hamming Loss", "Micro Precision", "Micro Recall", "Micro F1", "Macro F1"]
    ].mean()
    per_order_mean.to_csv(OUTPUT_DIR / "cc_ensemble_per_order_mean.csv")

    ensemble_summary = pd.DataFrame({
        "Mean": per_order_mean.mean(),
        "Std_across_orders": per_order_mean.std(),
    })
    ensemble_summary.to_csv(OUTPUT_DIR / "cc_ensemble_summary.csv")

    # also overall mean/std across all 50 individual fold runs (fold+order variance combined)
    overall_summary = pd.DataFrame({
        "Mean": ensemble_df[["Subset Accuracy", "Hamming Loss", "Micro Precision", "Micro Recall", "Micro F1", "Macro F1"]].mean(),
        "Std_across_all_runs": ensemble_df[["Subset Accuracy", "Hamming Loss", "Micro Precision", "Micro Recall", "Micro F1", "Macro F1"]].std(),
    })
    overall_summary.to_csv(OUTPUT_DIR / "cc_ensemble_overall_summary.csv")

    # ------------------------------------------------------------------
    # Co-occurrence-informed order (from Table IV: chain strongest-linked
    # labels first -> Ambiguous, Nonverifiable, Negative, Subjective, Vague)
    # ------------------------------------------------------------------
    co_occ_order_labels = ["Ambiguous", "Nonverifiable", "Negative", "Subjective", "Vague"]
    co_occ_order = [label_index[l] for l in co_occ_order_labels]

    co_occ_rows = []
    for fold_i, (X_train, y_train, X_test, y_test) in enumerate(fold_data, start=1):
        model = ClassifierChain(estimator=make_base_model(), order=co_occ_order)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        m = evaluate(y_test, preds)
        m["Fold"] = fold_i
        co_occ_rows.append(m)
    co_occ_df = pd.DataFrame(co_occ_rows)
    co_occ_df.to_csv(OUTPUT_DIR / "cc_cooccurrence_informed_results.csv", index=False)

    co_occ_summary = pd.DataFrame({
        "Mean": co_occ_df[["Subset Accuracy", "Hamming Loss", "Micro Precision", "Micro Recall", "Micro F1", "Macro F1"]].mean(),
        "Std": co_occ_df[["Subset Accuracy", "Hamming Loss", "Micro Precision", "Micro Recall", "Micro F1", "Macro F1"]].std(),
    })
    co_occ_summary.to_csv(OUTPUT_DIR / "cc_cooccurrence_informed_summary.csv")

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Original single fixed 'random' order (order_seed=42, same every fold)")
    print("=" * 70)
    print(f"Order used every fold: {original_df['Order'].iloc[0]}")
    print(original_df[["Subset Accuracy","Hamming Loss","Micro Precision","Micro Recall","Micro F1","Macro F1"]].mean())

    print("\n" + "=" * 70)
    print(f"Ensemble of {N_RANDOM_ORDERS} random orders x {N_SPLITS} folds ({N_RANDOM_ORDERS*N_SPLITS} fits)")
    print("=" * 70)
    print(overall_summary)

    print("\n" + "=" * 70)
    print("Co-occurrence-informed order:", co_occ_order_labels)
    print("=" * 70)
    print(co_occ_summary)


if __name__ == "__main__":
    main()
