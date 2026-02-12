import numpy as np
import pandas as pd

class MLSignalGenerator:
    """
    Layer 5: Baseline ML Models
    These are "Weak Learners" that generate signals, NOT decisions.
    """

    @staticmethod
    def predict_risk_score(features: dict) -> dict:
        """
        Predicts risk using MedGemma-RF (Random Forest) if available,
        otherwise falls back to Calibrated Logistic Regression rules.
        """
        # Prepare Features
        jitter = features.get("jitter_local", 0) * 100 # %
        shimmer = features.get("shimmer_local", 0) * 10 
        hnr = features.get("hnr", 20)
        f0_std = features.get("f0_std", 0.0)
        
        # 1. Try Loading Trained Model
        try:
            import joblib
            import os
            # Path relative to execution root or specific location
            model_path = os.path.join("medgemma_pd", "models", "medgemma_rf.pkl")
            
            if os.path.exists(model_path):
                clf = joblib.load(model_path)
                
                # Feature Vector: [jitter, shimmer, hnr, f0_std] matches training
                X = np.array([[jitter, shimmer, hnr, f0_std]])
                
                # Get Probability of Class 1 (PD)
                prob = clf.predict_proba(X)[0][1]
                
                return {
                    "model": "MedGemma-RF (v1.0)",
                    "risk_score": float(np.round(prob, 4)),
                    "confidence_interval": [max(0.0, prob-0.1), min(1.0, prob+0.1)],
                    "interpretation": "Elevated Risk (ML Verified)" if prob > 0.5 else "Within Normal Limits (ML Verified)"
                }
        except Exception as e:
            print(f"[ML Warning] Could not load RF model: {e}")

        # 2. Fallback: Calibrated Rule-Based System
        # Coefficients (Simulated from literature)
        # Bias -3.0 assumes baseline health.
        # Jitter > 1.0% cancels bias (-3.0 + 3.0*1.0 = 0 -> 50% risk).
        logits = -3.0 + (3.0 * jitter) + (0.5 * shimmer) - (0.1 * hnr)
        
        # Sigmoid
        probability = 1 / (1 + np.exp(-logits))
        
        return {
            "model": "LogisticRegression_RuleBased",
            "risk_score": float(np.round(probability, 4)),
            "confidence_interval": [float(probability - 0.1), float(probability + 0.1)],
            "interpretation": "Elevated vocal instability" if probability > 0.6 else "Within normal limits"
        }

class TrendAnalyzer:
    """
    Analyzes longitudinal progression.
    """
    
    @staticmethod
    def analyze_progression(history: pd.DataFrame) -> dict:
        """
        Input: DataFrame with 'session_month' and 'total_updrs'/'jitter_percent'.
        Output: Slope and degradation status.
        """
        if history.empty:
            return {"status": "insufficient_data"}
            
        results = {}
        
        # 1. UPDRS Trend
        if "total_updrs" in history.columns:
            slope = np.polyfit(history['session_month'], history['total_updrs'], 1)[0]
            results["updrs_slope"] = float(slope)
            results["updrs_trend"] = "worsening" if slope > 0.5 else "stable"
            
        # 2. Biomarker Trend (e.g. Jitter)
        if "jitter_percent" in history.columns:
            slope = np.polyfit(history['session_month'], history['jitter_percent'], 1)[0]
            results["jitter_slope"] = float(slope)
            
        return results
