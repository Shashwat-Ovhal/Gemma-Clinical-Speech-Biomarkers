import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from medgemma_pd.audio_pipeline.features import FeatureExtractor
from medgemma_pd.models.signals import MLSignalGenerator

def generate_sine_wave(freq=150, duration=1.0, sr=16000, jitter_level=0.0):
    t = np.linspace(0, duration, int(sr*duration))
    # Add phase jitter
    phase_jitter = np.random.normal(0, jitter_level, len(t))
    y = np.sin(2 * np.pi * freq * t + phase_jitter)
    return y, sr

def log(msg, f):
    print(msg)
    f.write(msg + "\n")

def test_pipeline():
    with open("verify_result.txt", "w", encoding="utf-8") as f:
        log("--- MedGemma-PD Logic Verification ---", f)
        
        # 1. Test Healthy (Pure Sine Wave)
        log("\n[Test 1] Pure Sine Wave (Healthy Proxy)", f)
        y_healthy, sr = generate_sine_wave(jitter_level=0.0)
        feats_healthy = FeatureExtractor.extract_features(y_healthy, sr)
        risk_healthy = MLSignalGenerator.predict_risk_score(feats_healthy)
        
        log(f"Jitter: {feats_healthy.get('jitter_local', 0)*100:.4f}%", f)
        log(f"HNR: {feats_healthy.get('hnr', 0):.2f} dB", f)
        log(f"Risk Score: {risk_healthy['risk_score']:.4f} ({risk_healthy['interpretation']})", f)
        
        if risk_healthy['risk_score'] < 0.3:
            log("✅ PASSED: Healthy signal incorrectly identified as Low Risk.", f)
        else:
            log("❌ FAILED: Healthy signal flagged as High Risk.", f)

        # 2. Test Pathological (High Jitter/Noise)
        log("\n[Test 2] Noisy Sine Wave (Tremor Proxy)", f)
        y_path, sr = generate_sine_wave(jitter_level=0.5) # Heavy phase jitter
        # Mix in noise
        noise = np.random.normal(0, 0.1, len(y_path))
        y_path = y_path + noise
        
        feats_path = FeatureExtractor.extract_features(y_path, sr)
        risk_path = MLSignalGenerator.predict_risk_score(feats_path)
        
        log(f"Jitter: {feats_path.get('jitter_local', 1.0)*100:.4f}%", f) # Default 1.0 if not detected
        log(f"HNR: {feats_path.get('hnr', 0):.2f} dB", f)
        log(f"Risk Score: {risk_path['risk_score']:.4f} ({risk_path['interpretation']})", f)
        
        if risk_path['risk_score'] > 0.4:
            log("✅ PASSED: Pathological signal correctly identified as Elevated Risk.", f)
        else:
            log("❌ FAILED: Pathological signal missed.", f)

if __name__ == "__main__":
    test_pipeline()
