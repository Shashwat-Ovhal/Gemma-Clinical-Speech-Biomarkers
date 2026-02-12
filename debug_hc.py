import numpy as np
import sys
import os
sys.path.append(os.getcwd())
import warnings
warnings.filterwarnings("ignore")

from medgemma_pd.audio_pipeline.features import FeatureExtractor
from medgemma_pd.audio_pipeline.preprocessing import AudioPreprocessor
from medgemma_pd.models.signals import MLSignalGenerator

def analyze_file(path):
    print(f"--- Analyzing: {os.path.basename(path)} ---")
    
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return

    # 1. Preprocess
    print("Step 1: Preprocessing...")
    y, sr, log = AudioPreprocessor.process(path)
    print(f"   > Duration: {len(y)/sr:.2f}s")
    print(f"   > Trim Log: {log}")

    # 2. Extract Features (Debug Mode)
    print("Step 2: Feature Extraction...")
    # We'll peek into intermediate values by subclassing or just running it
    features = FeatureExtractor.extract_features(y, sr)
    
    jitter = features.get("jitter_local", 0.0) * 100
    shimmer = features.get("shimmer_local", 0.0) * 100
    hnr = features.get("hnr", 0.0)
    f0_mean = features.get("f0_mean", 0.0)
    f0_std = features.get("f0_std", 0.0)
    
    print(f"   > F0 Mean: {f0_mean:.2f} Hz")
    print(f"   > F0 Std:  {f0_std:.2f} Hz (High variation suggests speech/intonation)")
    print(f"   > Jitter:  {jitter:.4f}%")
    print(f"   > Shimmer: {shimmer:.4f}%")
    print(f"   > HNR:     {hnr:.2f} dB")
    
    # 3. Model Prediction
    print("Step 3: Risk Prediction...")
    risk = MLSignalGenerator.predict_risk_score(features)
    print(f"   > Risk Score: {risk['risk_score']:.4f} ({risk['interpretation']})")
    
    # Diagnosis
    if jitter > 1.0:
        print("\n[DIAGNOSIS]: High Jitter detected.")
        if f0_std > 20.0:
            print("   > CAUSE: High Pitch Variance (Speech Intonation).")
            print("   > FIX NEEDED: Segment stable vowels only, ignore speech transitions.")
        else:
            print("   > CAUSE: True vocal instability (or noise).")


def find_hc_file():
    for root, dirs, files in os.walk("."):
        for file in files:
            if "hc" in file.lower() and file.endswith(".wav"):
               return os.path.join(root, file)
    return None

if __name__ == "__main__":
    hc_file = find_hc_file()
    if hc_file:
         analyze_file(hc_file)
    else:
         print("No HC file found.")
