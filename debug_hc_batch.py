import os
import sys
import numpy as np
import pandas as pd
import warnings
import time

sys.path.append(os.getcwd())
warnings.filterwarnings("ignore")

from medgemma_pd.audio_pipeline.features import FeatureExtractor
from medgemma_pd.audio_pipeline.preprocessing import AudioPreprocessor
from medgemma_pd.models.signals import MLSignalGenerator

HC_PATH = r"dataset- MDVR-KCL Dataset/26_29_09_2017_KCL/26-29_09_2017_KCL/ReadText/HC"

def test_hc_folder():
    print(f"--- Diagnosing HC Folder: {HC_PATH} ---")
    files = [f for f in os.listdir(HC_PATH) if f.endswith(".wav")]
    print(f"Found {len(files)} files.")
    
    results = []
    
    for f in files:
        path = os.path.join(HC_PATH, f)
        try:
            y, sr, _ = AudioPreprocessor.process(path)
            feats = FeatureExtractor.extract_features(y, sr)
            
            jitter = feats.get("jitter_local", 0.0) * 100
            shimmer = feats.get("shimmer_local", 0.0) * 100
            hnr = feats.get("hnr", 0.0)
            f0_std = feats.get("f0_std", 0.0)
            
            risk_data = MLSignalGenerator.predict_risk_score(feats)
            risk = risk_data['risk_score']
            
            # Check stability logic
            # FeatureExtractor now uses "stable segment"
            # We want to know if it worked.
            
            status = "FAIL" if risk > 0.5 else "PASS"
            if jitter > 1.04: status = "FAIL (Jitter)"
            
            print(f"{f}: J={jitter:.3f}% S={shimmer:.3f}% H={hnr:.1f} F0_std={f0_std:.1f} -> Risk={risk:.2f} [{status}]")
            
            results.append(risk)
            
        except Exception as e:
            print(f"{f}: ERROR {e}")

    avg_risk = np.mean(results)
    pass_rate = len([r for r in results if r < 0.5]) / len(results)
    print(f"\nAverage Risk: {avg_risk:.2f}")
    print(f"Pass Rate: {pass_rate*100:.1f}%")

if __name__ == "__main__":
    test_hc_folder()
