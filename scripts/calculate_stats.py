import csv
import math
import sys
# Try importing scipy, if fails, use manual T-Test approximation
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: Scipy not found. Using manual T-Test approximation.")

def calculate_mean_std(values):
    if not values: return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum([(x - mean)**2 for x in values]) / len(values)
    return mean, math.sqrt(variance)

def manual_ttest(group1, group2):
    # Welch's T-Test approximation
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2: return 0.99
    m1, s1 = calculate_mean_std(group1)
    m2, s2 = calculate_mean_std(group2)
    
    numerator = abs(m1 - m2)
    denominator = math.sqrt((s1**2 / n1) + (s2**2 / n2))
    if denominator == 0: return 0.0 if m1 != m2 else 1.0
    t_stat = numerator / denominator
    # Rough p-value from t-stat (assuming large N > 30 for Z-score approximation)
    # This is not precise but sufficient for demo if scipy is missing
    # P = 2 * (1 - CDF(|t|))
    # Simple lookup for demonstration:
    if t_stat > 3.29: return 0.001
    if t_stat > 2.58: return 0.01
    if t_stat > 1.96: return 0.05
    return 0.5

def main():
    print("--- MedGemma-PD Statistical Validation ---")
    
    # Load Data
    hc_jitter, pd_jitter = [], []
    hc_shimmer, pd_shimmer = [], []
    hc_hnr, pd_hnr = [], []
    
    with open("results.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check fallback
            if row.get("Fallback") == "True": continue
            
            try:
                j = float(row["Jitter"])
                s = float(row["Shimmer"])
                h = float(row["HNR"])
                
                if row["Group"] == "HC":
                    hc_jitter.append(j)
                    hc_shimmer.append(s)
                    hc_hnr.append(h)
                else:
                    pd_jitter.append(j)
                    pd_shimmer.append(s)
                    pd_hnr.append(h)
            except:
                continue
                
    print(f"Loaded: {len(pd_jitter)} PD samples, {len(hc_jitter)} HC samples.")
    
    metrics = [
        ("Jitter (%)", pd_jitter, hc_jitter),
        ("Shimmer (%)", pd_shimmer, hc_shimmer),
        ("HNR (dB)", pd_hnr, hc_hnr)
    ]
    
    print("\n### Statistical Significance Table")
    print("| Metric | Parkinson's (Avg) | Healthy Control (Avg) | P-Value (Significance) |")
    print("| :--- | :--- | :--- | :--- |")
    
    for name, pd_data, hc_data in metrics:
        pd_mean, pd_std = calculate_mean_std(pd_data)
        hc_mean, hc_std = calculate_mean_std(hc_data)
        
        if SCIPY_AVAILABLE:
            t_stat, p_val = stats.ttest_ind(pd_data, hc_data, equal_var=False)
        else:
            p_val = manual_ttest(pd_data, hc_data)
            
        sig = ""
        if p_val < 0.001: sig = "(Significant)"
        elif p_val < 0.01: sig = "(High)"
        elif p_val < 0.05: sig = "(Mod)"
        else: sig = "(Not Significant)"
        
        p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
        
        print(f"| {name} | {pd_mean:.3f} | {hc_mean:.3f} | {p_str} {sig} |")

if __name__ == "__main__":
    main()
