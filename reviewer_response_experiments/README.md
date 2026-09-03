# Camera-Ready Experiment Scripts — Reviewer Response Package

This package contains all additional experiments run in response to
Reviewer #1's comments on Paper #467 ("An XAI Framework for Multi-Label
Requirement Smell Detection and AI-Assisted Requirement Revision").

All scripts reuse the exact TF-IDF settings, base learner, and
`MultilabelStratifiedKFold(random_state=42)` protocol from the original
project (`RequirementSmells/src/`). They expect the harmonized dataset
at `data/processed/arta_harmonized.csv` (same relative path as the
original project) and, for the SHAP script, the original project's
`outputs/explainability/<Label>/top20_features.csv` files.

## Scripts (run in this order if reproducing from scratch)

### 1. `ensemble_cc.py` — Order-robustness evaluation (Reviewer 1, Comment 1)
Trains Classifier Chains with 10 independently sampled random label
orders across all 5 CV folds (50 fits total), plus replicates the
original single fixed-order result for a sanity check, plus a first
(later superseded) co-occurrence-informed order run.
**Outputs:** `cc_original_single_order_replication.csv`,
`cc_ensemble_all_runs.csv`, `cc_ensemble_per_order_mean.csv`,
`cc_ensemble_summary.csv`, `cc_ensemble_overall_summary.csv`,
`cc_ensemble_fold_level_avg.csv` (fold-level averages used for the
paired t-test), `cc_cooccurrence_informed_results.csv` /
`_summary.csv` (superseded — see script 2).

### 2. `cooccurrence_perfold.py` — Leakage-free co-occurrence order (Reviewer 1, Comment 1, corrected)
Supersedes the co-occurrence run inside `ensemble_cc.py`. Computes the
chain order strictly from each **training fold's** labels only (never
the test fold), so there is no leakage from the co-occurrence matrix
into held-out data. Rule: sum each label's co-occurrence with the
other four using training data only, sort descending (ties broken by
training-fold frequency, then alphabetically).
**Outputs:** `cc_cooccurrence_informed_PERFOLD_results.csv`,
`cc_cooccurrence_informed_PERFOLD_orders.csv` (the actual order used
per fold, for transparency), `cc_cooccurrence_informed_PERFOLD_summary.csv`.

### 3. `paired_ttests.py` — Significance tests (Reviewer 1, Comment 1)
Paired t-tests (5 folds, matching the original paper's design) for
BR vs. the 10-order average, and BR vs. the leakage-free
co-occurrence order. Prints t-statistics, p-values, and direction for
all six evaluation metrics.

### 4. `shap_stopword_classification.py` — SHAP function-word analysis (Reviewer 1, Comment 2)
Classifies each label's top-20 global SHAP features (from the
original project's offline SHAP output) as Content, Function/Stopword,
or Mixed (bigram with one stopword token), using the NLTK English
stopword list. Note: an earlier pass using sklearn's default stopword
list was discarded because it incorrectly flags domain content words
("system", "call") as stopwords for this dataset.
**Output:** `shap_top20_classified_NLTK.csv`, plus printed per-label
and overall percentage summaries.

### 5. `per_label_metrics.py` — Per-label BR metrics (Reviewer 1, suggestion)
Per-label Precision/Recall/F1 for Binary Relevance, same CV protocol
as `baseline_binary_relevance.py` in the original project.
**Outputs:** `per_label_metrics_all_folds.csv`,
`per_label_metrics_summary.csv` (mean ± std), `per_label_metrics_compact.csv`
(mean only, paper-table-ready).

### 6. `cc_per_label_metrics.py` — Per-label CC metrics (Reviewer 1, suggestion)
Same as above but for Classifier Chains, using the original single
fixed-order configuration (`order="random", random_state=42`) reported
in Table V, so it's directly comparable to script 5's BR numbers.
**Outputs:** `cc_per_label_metrics_all_folds.csv`,
`cc_per_label_metrics_compact.csv`.

## Sanity checks already performed (documented in the outputs)

- `ensemble_cc.py`'s single-order replication reproduces the original
  `classifier_chain_summary.csv` numbers to within rounding.
- `per_label_metrics.py`'s mean of 5 per-label F1 scores (0.517)
  matches the paper's reported BR Macro F1 (0.516 ± 0.019).
- `cc_per_label_metrics.py`'s mean of 5 per-label F1 scores (0.490)
  exactly matches the paper's reported CC Macro F1 (0.490 ± 0.015).
- `cc_per_label_metrics.py` confirms the chain order `[1,4,2,0,3]`
  (= Ambiguous, Vague, Nonverifiable, Subjective, Negative) is used
  identically every fold, confirming the original paper's "randomly
  determined order per fold" was in fact one fixed order reused across
  folds — this was flagged as a wording issue for Section IV-E.

## Requirements

```
pip install pandas numpy scikit-learn scipy nltk iterative-stratification --break-system-packages
```

(Package is `iterative-stratification` on PyPI; imports as `iterstrat`.)
