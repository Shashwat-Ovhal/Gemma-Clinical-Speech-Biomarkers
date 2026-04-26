import numpy as np
import scipy.stats
import joblib
import os
import sys

CV_RESULTS  = "medgemma_pd/models/cv_results.pkl"
DELONG_OUT  = "medgemma_pd/models/delong_test_results.txt"

def compute_midrank(x):
    """Computes midranks."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    # Note: argsort of argsort gives the inverse permutation
    T2[J] = T + 1
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    tx = np.empty([positive_examples.shape[0], m], dtype=np.float64)
    ty = np.empty([negative_examples.shape[0], n], dtype=np.float64)
    tz = np.empty([predictions_sorted_transposed.shape[0], m + n], dtype=np.float64)
    for r in range(predictions_sorted_transposed.shape[0]):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values."""
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return 10 ** (np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10))

def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count

def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return aucs[0], delongcov

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes p-value for hypothesis that two ROC AUCs are different
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)

def run_delong_test():
    if not os.path.exists(CV_RESULTS):
        print("No cv_results.pkl found. Cannot run DeLong Test.")
        return

    results = joblib.load(CV_RESULTS)
    
    base_r = next((r for r in results if "Baseline" in r["model"]), None)
    best_r = max((r for r in results if "Tuned" in r["model"]), key=lambda r: r["auc"])
    
    if not base_r or not best_r:
        print("Missing Baseline or Tuned model in CV results.")
        return
        
    y_true = np.array(base_r["y_true"])
    y_prob_base = np.array(base_r["y_prob"])
    y_prob_best = np.array(best_r["y_prob"])
    
    # Calculate DeLong's test p-value
    p_value = delong_roc_test(y_true, y_prob_base, y_prob_best)[0][0]
    
    auc_base, var_base = delong_roc_variance(y_true, y_prob_base)
    auc_best, var_best = delong_roc_variance(y_true, y_prob_best)
    
    ci_base = [auc_base - 1.96 * np.sqrt(var_base), auc_base + 1.96 * np.sqrt(var_base)]
    ci_best = [auc_best - 1.96 * np.sqrt(var_best), auc_best + 1.96 * np.sqrt(var_best)]
    
    output = []
    output.append("=== DeLong's Test for Statistical Significance ===")
    output.append(f"Model A (Baseline) : {base_r['model']}")
    output.append(f"Model B (Proposed) : {best_r['model']}")
    output.append(f"AUC A: {auc_base:.4f} 95% CI [{ci_base[0][0]:.4f} - {ci_base[1][0]:.4f}]")
    output.append(f"AUC B: {auc_best:.4f} 95% CI [{ci_best[0][0]:.4f} - {ci_best[1][0]:.4f}]")
    output.append(f"DeLong p-value: {p_value:.6e}")
    if p_value < 0.05:
        output.append("Conclusion: The Proposed model is STATISTICALLY SIGNIFICANTLY better (p < 0.05).")
    else:
        output.append("Conclusion: No significant difference found (p >= 0.05).")
        
    with open(DELONG_OUT, "w") as f:
        f.write("\n".join(output))
        
    print("\n".join(output))
    print(f"Results saved to {DELONG_OUT}")

if __name__ == "__main__":
    run_delong_test()
