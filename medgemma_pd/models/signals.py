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
        Simulates a Logistic Regression model output.
        Inputs: Dictionary of speech biomarkers.
        Output: Risk probability (0.0 to 1.0) and top contributing features.
        """
        # Coefficients (Simulated from literature)
        # Jitter: +, Shimmer: +, HNR: -
        jitter = features.get("jitter_local", 0) * 100 # Scale to %
        shimmer = features.get("shimmer_local", 0) * 10 
        hnr = features.get("hnr", 20)
        
        # Simple Linear Combination
        # Log-odds = b0 + b1*Jitter + b2*Shimmer - b3*HNR
        logits = -2.0 + (0.5 * jitter) + (0.3 * shimmer) - (0.1 * hnr)
        
        # Sigmoid
        probability = 1 / (1 + np.exp(-logits))
        
        return {
            "model": "LogisticRegression_v1",
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
