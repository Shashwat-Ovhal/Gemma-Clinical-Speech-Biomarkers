import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
import time

# Verify environment
sys.path.append(os.getcwd())
try:
    from medgemma_pd.audio_pipeline.features import FeatureExtractor
    from medgemma_pd.audio_pipeline.preprocessing import AudioPreprocessor
except ImportError:
    print("CRITICAL: Run this script from the project root directory.")
    sys.exit(1)

warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_ROOT = r"dataset- MDVR-KCL Dataset/26_29_09_2017_KCL/26-29_09_2017_KCL/ReadText"
OUTPUT_CSV = "medgemma_pd/models/training_data.csv"
REPORT_FILE = "medgemma_pd/models/validation_report.txt"

def extract_dataset_features():
    print(f"--- 1. Data Extraction from: {DATA_ROOT} ---")
    data = []
    
    # 1. Find all files
    hc_path = os.path.join(DATA_ROOT, "HC")
    pd_path = os.path.join(DATA_ROOT, "PD")
    
    if not os.path.exists(hc_path) or not os.path.exists(pd_path):
        print(f"Error: Dataset paths not found. \nExpected:\n  {hc_path}\n  {pd_path}")
        return None

    files = []
    for f in os.listdir(hc_path):
        if f.endswith(".wav"): files.append((os.path.join(hc_path, f), 0, f)) # 0 = Healthy
    for f in os.listdir(pd_path):
        if f.endswith(".wav"): files.append((os.path.join(pd_path, f), 1, f)) # 1 = PD
        
    print(f"Found {len(files)} files ({len([x for x in files if x[1]==0])} HC, {len([x for x in files if x[1]==1])} PD)")
    
    # 2. Extract Features
    print("Extracting features... (This may take a moment)")
    start_time = time.time()
    
    for path, label, fname in files:
        try:
            # Preprocess
            y, sr, _ = AudioPreprocessor.process(path)
            
            # Extract
            feats = FeatureExtractor.extract_features(y, sr)
            
            if not feats.get("valid_voice_detected", False):
                print(f"  [WARN] No voice detected in {fname}, skipping.")
                continue
                
            row = {
                "filename": fname,
                "label": label,
                "jitter": feats.get("jitter_local", 0.0) * 100, # %
                "shimmer": feats.get("shimmer_local", 0.0) * 100, # %
                "hnr": feats.get("hnr", 0.0),
                "f0_std": feats.get("f0_std", 0.0)
            }
            data.append(row)
            sys.stdout.write(".")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"\n  [ERR] Failed {fname}: {e}")
            
    print(f"\nExtraction complete in {time.time() - start_time:.2f}s")
    
    df = pd.DataFrame(data)
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved dataset to {OUTPUT_CSV}")
    return df

def train_and_validate(df):
    X = df[["jitter", "shimmer", "hnr", "f0_std"]]
    y = df["label"]
    
    models = {
        "Baseline (Logistic Regression)": LogisticRegression(class_weight='balanced', random_state=42),
        "MedGemma AI (Random Forest)": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    }
    
    report_buffer = []
    
    for name, model in models.items():
        print(f"\nEvaluating: {name}...")
        
        # Leave-One-Out CV
        loo = LeaveOneOut()
        y_true = []
        y_pred = []
        
        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            y_true.append(y_test.values[0])
            y_pred.append(pred[0])
            
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        
        res = f"""
### {name} Performance
- Accuracy:  {acc:.2%}
- Precision: {prec:.2%}
- Sensitivity (Recall): {rec:.2%}
- F1-Score:  {f1:.2%}
- Confusion Matrix:
{confusion_matrix(y_true, y_pred)}
"""
        print(res)
        report_buffer.append(res)
        
        if "Random Forest" in name:
            model.fit(X, y) # Final fit on all data
            imps = model.feature_importances_
            imp_str = "\nFeature Importance:\n"
            for i, col in enumerate(X.columns):
                imp_str += f"- {col}: {imps[i]:.4f}\n"
            print(imp_str)
            report_buffer.append(imp_str)
            
            # SAVE MODEL
            import joblib
            model_path = "medgemma_pd/models/medgemma_rf.pkl"
            joblib.dump(model, model_path)
            print(f"Saved Random Forest Model to {model_path}")

    # Save Report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# MedGemma-PD Validation Report\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n".join(report_buffer))
        
    print(f"Validation Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    if os.path.exists(OUTPUT_CSV):
        print(f"Loading cached data from {OUTPUT_CSV}")
        df = pd.read_csv(OUTPUT_CSV)
    else:
        df = extract_dataset_features()
    
    if df is not None and not df.empty:
        train_and_validate(df)
