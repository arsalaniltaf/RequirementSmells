"""
Paired t-tests: Binary Relevance vs. each Classifier Chains configuration,
using the same 5-fold paired design as the original paper (5 fold-level
values per model, paired by fold).

Requires the CSVs already produced by:
  - baseline_binary_relevance.py (original project)   -> BR fold results
  - ensemble_cc.py                                     -> 10-order average, per fold
  - cooccurrence_perfold.py                            -> leakage-free co-occurrence order, per fold
"""
import pandas as pd
from scipy import stats

METRICS = ["Subset Accuracy", "Hamming Loss", "Micro Precision", "Micro Recall", "Micro F1", "Macro F1"]

BR_PATH = "/home/claude/project/RequirementSmells/outputs/metrics/baseline_results.csv"
CC_ENSEMBLE_FOLD_AVG_PATH = "outputs/cc_ensemble_fold_level_avg.csv"
CC_COOCC_PERFOLD_PATH = "outputs/cc_cooccurrence_informed_PERFOLD_results.csv"


def paired_ttest(br_df, cc_df, cc_label):
    print(f"=== Paired t-test: BR vs {cc_label} ===")
    for m in METRICS:
        br_vals = br_df.sort_values("Fold")[m].values
        cc_vals = cc_df.sort_values("Fold")[m].values
        t, p = stats.ttest_rel(br_vals, cc_vals)
        direction = "BR higher" if br_vals.mean() > cc_vals.mean() else "CC higher"
        sig = "  *" if p < 0.05 else ""
        print(f"{m:18s}  BR={br_vals.mean():.4f}  CC={cc_vals.mean():.4f}  t={t:.3f}  p={p:.4f}  ({direction}){sig}")
    print()


def main():
    br = pd.read_csv(BR_PATH)
    cc_ensemble = pd.read_csv(CC_ENSEMBLE_FOLD_AVG_PATH)
    cc_coocc = pd.read_csv(CC_COOCC_PERFOLD_PATH)

    paired_ttest(br, cc_ensemble, "10-order average CC (order-robustness evaluation)")
    paired_ttest(br, cc_coocc, "leakage-free co-occurrence-informed order CC")


if __name__ == "__main__":
    main()
