"""
train_validation.py — Level 1 Publication Validation Engine
Implements:
  - Stratified 5-Fold Cross-Validation (replaces LOO)
  - Three models: Logistic Regression, SVM (new baseline), Random Forest
  - AUC-ROC with 95% CI via bootstrapping (1000 iterations)
  - Per-fold metrics + aggregate table in publication-ready format
  - Saves best model + validation report
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

sys.path.append(os.getcwd())
warnings.filterwarnings("ignore")

try:
    from medgemma_pd.audio_pipeline.features import FeatureExtractor
    from medgemma_pd.audio_pipeline.preprocessing import AudioPreprocessor
except ImportError:
    print("CRITICAL: Run from project root directory.")
    sys.exit(1)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT   = "./data/mpower_dataset"    # mPower dataset (Level 1 target)
OUTPUT_CSV  = "medgemma_pd/models/mpower_training_data.csv"
REPORT_FILE = "medgemma_pd/models/level1_validation_report.md"
MODEL_PATH  = "medgemma_pd/models/medgemma_rf.pkl"

N_FOLDS        = 5
N_BOOTSTRAP    = 1000
RANDOM_STATE   = 42

FEATURES    = ["jitter", "shimmer", "hnr", "f0_std"]


# ── 1. Feature Extraction ────────────────────────────────────────────────────
def extract_dataset_features() -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  STAGE 1 — Feature Extraction from: {DATA_ROOT}")
    print(f"{'='*60}")

    files = []
    for label_val, cls in [(0, "HC"), (1, "PD")]:
        cls_path = os.path.join(DATA_ROOT, cls)
        if not os.path.exists(cls_path):
            print(f"  [ERR] Missing directory: {cls_path}")
            continue
        for f in os.listdir(cls_path):
            if f.endswith((".wav", ".m4a")):
                files.append((os.path.join(cls_path, f), label_val, f))

    print(f"  Found {len(files)} audio files "
          f"({sum(1 for _,l,_ in files if l==0)} HC, {sum(1 for _,l,_ in files if l==1)} PD)")

    data, failed = [], 0
    start = time.time()

    for i, (path, label, fname) in enumerate(files):
        try:
            y, sr, _ = AudioPreprocessor.process(path)
            feats = FeatureExtractor.extract_features(y, sr)
            if not feats.get("valid_voice_detected", False):
                print(f"\n  [SKIP] No voice in {fname}")
                continue
            data.append({
                "filename": fname,
                "label"   : label,
                "jitter"  : feats.get("jitter_local",  0.0) * 100,
                "shimmer" : feats.get("shimmer_local",  0.0) * 100,
                "hnr"     : feats.get("hnr",            0.0),
                "f0_std"  : feats.get("f0_std",         0.0),
            })
            sys.stdout.write(f"\r  Processed {i+1}/{len(files)} ({len(data)} valid)")
            sys.stdout.flush()
        except Exception as e:
            failed += 1
            print(f"\n  [ERR] {fname}: {e}")

    print(f"\n  Done in {time.time()-start:.1f}s | {len(data)} valid, {failed} failed")

    df = pd.DataFrame(data).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved feature set -> {OUTPUT_CSV}")
    return df


# -- 2. Bootstrap CI ----------------------------------------------------------
def bootstrap_ci(y_true, y_score, metric_fn, n=N_BOOTSTRAP, ci=0.95):
    """Returns (mean, lower, upper) for a given metric via bootstrapping."""
    rng = np.random.default_rng(RANDOM_STATE)
    scores = []
    y_true, y_score = np.array(y_true), np.array(y_score)
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_score[idx]))
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(scores, [alpha * 100, (1 - alpha) * 100])
    return float(np.mean(scores)), lo, hi


# -- 3. Cross-Validation Engine -----------------------------------------------
def run_stratified_cv(df: pd.DataFrame) -> list:
    global FEATURES

    # Step 1: Feature correlation pruning
    corr_matrix = df[FEATURES].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    cols_to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.9)]
    if cols_to_drop:
        print(f"  Dropping correlated features: {cols_to_drop}")
        FEATURES = [f for f in FEATURES if f not in cols_to_drop]

    X = df[FEATURES].values
    y = df["label"].values

    # Tuning grids
    rf_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.3]
    }
    xgb_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0]
    }

    models = {
        "Logistic Regression (LR)": (
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
            None # No tuning grid
        ),
        "SVM - RBF (Baseline)": (
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            None
        ),
        "Random Forest (Tuned)": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            rf_grid
        ),
        "XGBoost (Tuned)": (
            XGBClassifier(random_state=RANDOM_STATE, scale_pos_weight=(len(y)-sum(y))/sum(y), use_label_encoder=False, eval_metric="logloss"),
            xgb_grid
        )
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_results = []

    print(f"\n{'='*60}")
    print(f"  STAGE 2 - Stratified {N_FOLDS}-Fold Cross-Validation with SMOTE & Tuning")
    print(f"{'='*60}")

    for model_name, (base_model, param_grid) in models.items():
        print(f"\n  > {model_name}")
        fold_metrics = []

        yt_all, yp_all, yp_prob_all = [], [], []

        best_model_overall = None
        best_fold_auc = -1

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # SMOTE only on training split
            if len(np.unique(y_tr)) > 1:
                smote = SMOTE(random_state=RANDOM_STATE)
                X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

            if param_grid is not None:
                inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
                search = RandomizedSearchCV(
                    estimator=base_model, param_distributions=param_grid,
                    n_iter=20, scoring='roc_auc', cv=inner_cv, random_state=42, n_jobs=-1, verbose=0
                )
                search.fit(X_tr, y_tr)
                model = search.best_estimator_
            else:
                model = base_model
                model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]

            fold_auc  = roc_auc_score(y_te, y_prob)
            fold_acc  = accuracy_score(y_te, y_pred)
            fold_f1   = f1_score(y_te, y_pred)
            fold_metrics.append({"auc": fold_auc, "acc": fold_acc, "f1": fold_f1})

            yt_all.extend(y_te.tolist())
            yp_all.extend(y_pred.tolist())
            yp_prob_all.extend(y_prob.tolist())

            print(f"    Fold {fold+1}: AUC={fold_auc:.3f}  Acc={fold_acc:.3f}  F1={fold_f1:.3f}")

            if fold_auc > best_fold_auc:
                best_fold_auc = fold_auc
                best_model_overall = model

        # Overall metrics across all folds
        mean_auc, lo_auc, hi_auc = bootstrap_ci(yt_all, yp_prob_all, roc_auc_score)
        mean_acc, lo_acc, hi_acc = bootstrap_ci(
            yt_all, yp_prob_all,
            lambda yt, yp: accuracy_score(yt, (yp >= 0.5).astype(int))
        )
        cm = confusion_matrix(yt_all, yp_all)

        result = {
            "model"  : model_name,
            "auc"    : mean_auc,
            "auc_lo" : lo_auc,
            "auc_hi" : hi_auc,
            "acc"    : mean_acc,
            "acc_lo" : lo_acc,
            "acc_hi" : hi_acc,
            "f1"     : f1_score(yt_all, yp_all),
            "sens"   : recall_score(yt_all, yp_all),
            "spec"   : cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm[0, 0] + cm[0, 1] > 0 else 0,
            "cm"     : cm,
            "model_obj": best_model_overall,
            "y_true" : yt_all,
            "y_prob" : yp_prob_all,
        }
        all_results.append(result)

        print(f"\n  -- {model_name} Aggregate --")
        print(f"     AUC-ROC : {mean_auc:.3f}  [{lo_auc:.3f} - {hi_auc:.3f}]  95% CI")
        print(f"     Accuracy: {mean_acc:.3f}  [{lo_acc:.3f} - {hi_acc:.3f}]  95% CI")
        print(f"     F1      : {result['f1']:.3f}")
        print(f"     Sens    : {result['sens']:.3f}  (PD recall)")
        print(f"     Spec    : {result['spec']:.3f}  (HC recall)")

    return all_results


# -- 4. Report Writer ---------------------------------------------------------
def write_report(results: list, df: pd.DataFrame):
    n_pd = int(df["label"].sum())
    n_hc = len(df) - n_pd

    lines = [
        "# MedGemma-PD - Level 1 Validation Report",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Dataset:** mPower (Synapse syn4993293)",
        f"**Cohort:** {len(df)} patients ({n_hc} HC, {n_pd} PD)",
        f"**Validation:** Stratified {N_FOLDS}-Fold CV - 95% CI via {N_BOOTSTRAP} bootstraps",
        "",
        "---",
        "",
        "## Results Table",
        "",
        "| Model | AUC-ROC [95% CI] | Accuracy [95% CI] | F1 | Sensitivity | Specificity |",
        "| :---- | :--------------: | :---------------: | :-: | :---------: | :---------: |",
    ]

    best_auc = max(results, key=lambda r: r["auc"])

    for r in results:
        marker = " *" if r["model"] == best_auc["model"] else ""
        lines.append(
            f"| {r['model']}{marker} "
            f"| {r['auc']:.3f} [{r['auc_lo']:.3f}-{r['auc_hi']:.3f}] "
            f"| {r['acc']:.3f} [{r['acc_lo']:.3f}-{r['acc_hi']:.3f}] "
            f"| {r['f1']:.3f} "
            f"| {r['sens']:.3f} "
            f"| {r['spec']:.3f} |"
        )

    # Feature importance for best tunable model (RF or XGB)
    best_tunable = max((r for r in results if "Tuned" in r["model"]), key=lambda k: k["auc"])
    if best_tunable and hasattr(best_tunable["model_obj"], "feature_importances_"):
        best_tunable["model_obj"].fit(df[FEATURES].values, df["label"].values)
        imps = best_tunable["model_obj"].feature_importances_
        lines += [
            "",
            f"## {best_tunable['model']} Feature Importance",
            "",
            "| Feature | Importance |",
            "| :------ | ---------: |",
        ]
        for feat, imp in sorted(zip(FEATURES, imps), key=lambda x: -x[1]):
            lines.append(f"| {feat} | {imp:.4f} ({imp*100:.1f}%) |")

        # Save model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(best_tunable["model_obj"], MODEL_PATH)
        print(f"  Best model ({best_tunable['model']}) saved -> {MODEL_PATH}")

    # Also save the results list for the stats script
    joblib.dump(results, os.path.join(os.path.dirname(MODEL_PATH), "cv_results.pkl"))

    lines += ["", "---", "*Generated by MedGemma-PD Level 1 Validation Engine*"]

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Report saved -> {REPORT_FILE}")


# -- Main ---------------------------------------------------------------------
if __name__ == "__main__":
    if os.path.exists(OUTPUT_CSV):
        print(f"Loading cached features from {OUTPUT_CSV}")
        df = pd.read_csv(OUTPUT_CSV)
    else:
        df = extract_dataset_features()

    if df is None or df.empty:
        print("No data extracted. Ensure mPower audio is in ./data/mpower_dataset/")
        sys.exit(1)

    if len(df) < 20:
        print(f"Only {len(df)} samples available - need at least 20 for 5-fold CV. Aborting.")
        sys.exit(1)

    results = run_stratified_cv(df)
    write_report(results, df)

    print(f"\n{'='*60}")
    print("  Level 1 validation complete!")
    print(f"  Report: {REPORT_FILE}")
    print(f"{'='*60}\n")
