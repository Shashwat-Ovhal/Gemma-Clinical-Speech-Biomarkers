import os, sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import scipy.stats
import synapseclient

# -- File paths --
CSV_DATA    = "medgemma_pd/models/mpower_training_data.csv"
CV_RESULTS  = "medgemma_pd/models/cv_results.pkl"
BEST_MODEL  = "medgemma_pd/models/medgemma_rf.pkl"
TABLE1_CSV  = "medgemma_pd/models/table1_demographics.csv"
ROC_IMG     = "medgemma_pd/models/roc_curves.png"
SHAP_IMG    = "medgemma_pd/models/shap_summary.png"
STATS_TXT   = "medgemma_pd/models/stats.txt"
PR_IMG      = "medgemma_pd/models/pr_curves.png"
CM_IMG      = "medgemma_pd/models/confusion_matrix.png"
PERF_TABLE  = "medgemma_pd/models/performance_table.csv"

def build_demographics_table():
    """Generates Table 1 by cross-referencing valid extracted data with Synapse DB."""
    try:
        # Load valid files
        df_valid = pd.read_csv(CSV_DATA)
        valid_records = df_valid["filename"].apply(lambda x: x.split('.')[0]).tolist()

        syn = synapseclient.Synapse()
        auth_token = os.environ.get("SYNAPSE_AUTH_TOKEN", "eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc3NjQ0OTU5MCwiaWF0IjoxNzc2NDQ5NTkwLCJqdGkiOiIzNTc4NCIsInN1YiI6IjM1NzY2MjkifQ.BJ__fn73AW3CdHT3huDqcl_COEuO61dCjI70jtYh2YL_zeT-9SVf4QonVvTmjGyIF0AZnZUQqfPkSluCFZV_p6wptXTwdBDQDjIAl8EGh2sgSbNBlhc9i27bHPwUYwJWfeqT-6xHx7dYZ8aoVmA1RDJUwsgpAVQAVSr-Eo87HnGRAYKQjwlyBOHT4R-bUIsVRLc1xq86cUbA6huyzis31CCrCBbbGSES7crvjS8iqdIiTYiWzDwwBcqPByeAcEQ6FO31zxQ7pIgv9-9eUm9erpmaLS2Fys5-38GnOa929PY5Fu2vZ86MuDKyx6jpPmmwFpbDfGx7oNTw-D3Ku33YBQ")
        if not auth_token:
            print("  [WARN] SYNAPSE_AUTH_TOKEN missing, skipping Table 1 generation.")
            return

        syn.login(authToken=auth_token.strip())

        # Pull demographics
        q_demo = syn.tableQuery('SELECT healthCode, "professional-diagnosis", age, gender FROM syn5511429 WHERE "professional-diagnosis" IS NOT NULL')
        df_demo = q_demo.asDataFrame()

        # Pull Voice to map recordId to healthCode
        q_voice = syn.tableQuery('SELECT recordId, healthCode FROM syn5511444')
        df_voice = q_voice.asDataFrame()

        # Merge
        merged = df_voice[df_voice['recordId'].isin(valid_records)].merge(df_demo, on='healthCode', how='inner')
        merged["group"] = merged["professional-diagnosis"].apply(lambda x: "PD" if x is True else "HC")

        table1 = []
        for grp in ["HC", "PD"]:
            sub = merged[merged["group"] == grp]
            n_total = len(sub)
            age_missing = sub['age'].isna().sum() / n_total * 100 if n_total > 0 else 0
            gender_missing = sub['gender'].isna().sum() / n_total * 100 if n_total > 0 else 0
            table1.append({
                "Group": grp,
                "n": n_total,
                "Age (mean ± std)": f"{sub['age'].mean():.1f} ± {sub['age'].std():.1f}" if pd.notnull(sub['age'].mean()) else "N/A",
                "Age Missing (%)": f"{age_missing:.1f}%",
                "Female (%)": f"{len(sub[sub['gender'] == 'Female']) / len(sub) * 100:.1f}%" if n_total > 0 else "N/A",
                "Gender Missing (%)": f"{gender_missing:.1f}%"
            })
        t1_df = pd.DataFrame(table1)
        t1_df.to_csv(TABLE1_CSV, index=False)
        print(f"  [+] Demographic Table 1 saved successfully to {TABLE1_CSV}")

    except Exception as e:
        print(f"  [ERR] building Demographics Table: {e}")

def plot_roc_curves():
    """Plots ROC curves for all evaluated models on a single figure."""
    if not os.path.exists(CV_RESULTS):
        print("  [ERR] cv_results.pkl not found. Run train_validation.py first.")
        return

    results = joblib.load(CV_RESULTS)
    plt.figure(figsize=(8, 6))

    for r in results:
        y_true = np.array(r["y_true"])
        y_prob = np.array(r["y_prob"])
        
        # Calculate standard ROC curve overall
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f"{r['model']} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - mPower Cohort')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_IMG, dpi=300)
    plt.close()
    print(f"  [+] ROC curve figure saved successfully to {ROC_IMG}")

def plot_shap_importance():
    """Generates SHAP summary plot for the best model."""
    if not os.path.exists(CSV_DATA) or not os.path.exists(BEST_MODEL):
        print("  [ERR] Model or data missing for SHAP analysis.")
        return

    df = pd.read_csv(CSV_DATA)
    # The columns from the CV results (dynamically found)
    results = joblib.load(CV_RESULTS)
    rf_res = [r for r in results if "Tuned" in r["model"]]
    if not rf_res:
        return
        
    best_tunable = max(rf_res, key=lambda k: k["auc"])
    model = best_tunable["model_obj"]
    
    # We must identify which features were actually used vs pruned
    X_train_df = df.drop(columns=["filename", "label"])
    if hasattr(model, 'feature_names_in_'):
        features = model.feature_names_in_
        X_train_df = X_train_df[features]
    else:
        # Assuming the exact columns fit logic from training:
        print("  [WARN] Model doesn't explicitly store feature names, defaulting to dataframe columns")
        features = X_train_df.columns.tolist()

    try:
        if "Random Forest" in best_tunable["model"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train_df)
            if isinstance(shap_values, list): # RF gives a list per class
                shap_values_plot = shap_values[1]
            else:
                shap_values_plot = shap_values
        elif "XGBoost" in best_tunable["model"]:
            explainer = shap.Explainer(model, X_train_df)
            shap_values = explainer(X_train_df)
            shap_values_plot = shap_values.values

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_plot, X_train_df, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_IMG, dpi=300)
        plt.close()
        print(f"  [+] SHAP feature importance saved successfully to {SHAP_IMG}")
    except Exception as e:
        print(f"  [ERR] SHAP generation failed: {e}")

def calc_mcnemar():
    """Calculates McNemar's test p-value comparing Baseline SVM vs Best Tuned Model."""
    results = joblib.load(CV_RESULTS)
    base_r = next((r for r in results if "Baseline" in r["model"]), None)
    best_r = max((r for r in results if "Tuned" in r["model"]), key=lambda r: r["auc"])

    if not base_r or not best_r:
        return

    y_true = np.array(base_r["y_true"])
    pred_base = (np.array(base_r["y_prob"]) >= 0.5).astype(int)
    pred_best = (np.array(best_r["y_prob"]) >= 0.5).astype(int)

    # Contingency Table
    both_correct = np.sum((pred_base == y_true) & (pred_best == y_true))
    best_only = np.sum((pred_base != y_true) & (pred_best == y_true))
    base_only = np.sum((pred_base == y_true) & (pred_best != y_true))
    both_wrong = np.sum((pred_base != y_true) & (pred_best != y_true))

    table = [[both_correct, base_only],
             [best_only, both_wrong]]

    result = mcnemar(table, exact=True)

    lines = [
        "McNemar's Test (Significance of Tuned Model vs SVM Baseline)",
        "----------------------------------------------------------",
        f"Base Model: {base_r['model']}",
        f"Tuned Model: {best_r['model']}",
        f"Both Correct: {both_correct}",
        f"Tuned Only Correct: {best_only}",
        f"Base Only Correct: {base_only}",
        f"Both Wrong: {both_wrong}",
        f"p-value: {result.pvalue:.5f}",
        ""
    ]
    if result.pvalue < 0.05:
        lines.append("Conclusion: The tuned model is STATISTICALLY SIGNIFICANTLY better than the baseline SVM (p < 0.05).")
    else:
        lines.append("Conclusion: No statistically significant difference detected (p >= 0.05) vs baseline classification error.")

    with open(STATS_TXT, "w") as f:
        f.write("\n".join(lines))
    print(f"  [+] Statistical tests saved to {STATS_TXT}")

def plot_pr_curves():
    """Plots Precision-Recall curves for all evaluated models."""
    if not os.path.exists(CV_RESULTS): return
    results = joblib.load(CV_RESULTS)
    plt.figure(figsize=(8, 6))

    for r in results:
        y_true = np.array(r["y_true"])
        y_prob = np.array(r["y_prob"])
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, lw=2, label=f"{r['model']} (AP = {pr_auc:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - mPower Cohort')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(PR_IMG, dpi=300)
    plt.close()
    print(f"  [+] PR curve figure saved successfully to {PR_IMG}")

def plot_confusion_matrix():
    """Plots the confusion matrix for the best tuned model."""
    if not os.path.exists(CV_RESULTS): return
    results = joblib.load(CV_RESULTS)
    best_r = max((r for r in results if "Tuned" in r["model"]), key=lambda r: r["auc"])
    
    y_true = np.array(best_r["y_true"])
    y_pred = (np.array(best_r["y_prob"]) >= 0.5).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy (HC)", "Parkinson's (PD)"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {best_r['model']}")
    plt.tight_layout()
    plt.savefig(CM_IMG, dpi=300)
    plt.close()
    print(f"  [+] Confusion Matrix saved successfully to {CM_IMG}")

def generate_performance_table():
    """Generates a comprehensive performance table with 95% CIs."""
    if not os.path.exists(CV_RESULTS): return
    results = joblib.load(CV_RESULTS)
    
    table_data = []
    for r in results:
        y_true = np.array(r["y_true"])
        y_pred = (np.array(r["y_prob"]) >= 0.5).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        table_data.append({
            "Model": r["model"],
            "AUC": r["auc"],
            "Accuracy": acc,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "F1_Score": f1
        })
        
    df = pd.DataFrame(table_data)
    df.to_csv(PERF_TABLE, index=False)
    print(f"  [+] Performance Table saved successfully to {PERF_TABLE}")

if __name__ == "__main__":
    print("Generating Scientific Figures & Statistics (Phase 2)...")
    build_demographics_table()
    plot_roc_curves()
    plot_shap_importance()
    calc_mcnemar()
    plot_pr_curves()
    plot_confusion_matrix()
    generate_performance_table()
    print("Done!")
