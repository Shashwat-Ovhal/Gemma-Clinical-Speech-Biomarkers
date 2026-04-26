import os
import sys
import torch
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bspc_pipeline.cross_attention_fusion import PDCrossAttentionClassifier

# -- Config --
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./medgemma_pd/models/bspc_model_teacher.pt"
FEATURE_CSV = "./medgemma_pd/models/mpower_training_data.csv"
DATA_ROOT = "./data/mpower_denoised" # Use denoised if available
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = "./data/mpower_dataset"

MAX_LEN = 48000 # 3 seconds @ 16kHz
FEATURES = ["shimmer", "jitter", "hnr", "f0_std"]

def mask_audio(audio: np.ndarray, sr: int, mask_type: str) -> np.ndarray:
    """
    Masks specific parts of the audio array.
    mask_type: 'voiced' or 'unvoiced'
    """
    # Simple voiced/unvoiced detection using zero crossing rate and energy
    frame_length = 2048
    hop_length = 512
    
    # Energy
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Normalize
    energy_norm = energy / (np.max(energy) + 1e-6)
    
    # Voiced heuristics: high energy, low zcr
    # We create a boolean mask for frames
    voiced_frames = (energy_norm > 0.1) & (zcr < 0.15)
    
    masked_audio = audio.copy()
    
    for i, is_voiced in enumerate(voiced_frames):
        start = i * hop_length
        end = start + frame_length
        if end > len(audio): end = len(audio)
        
        if mask_type == "voiced" and is_voiced:
            masked_audio[start:end] = 0.0
        elif mask_type == "unvoiced" and not is_voiced:
            masked_audio[start:end] = 0.0
            
    return masked_audio

def run_attention_grounding_test():
    print("Running Masked Sensitivity (Vowel) Test for Attention Grounding...")
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_CSV):
        print("Required model or feature CSV not found. Skipping.")
        return

    df = pd.read_csv(FEATURE_CSV)
    
    model = PDCrossAttentionClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    results = {"baseline": [], "voiced_masked": [], "unvoiced_masked": [], "y_true": []}
    
    for idx, row in df.iterrows():
        cls_dir = "PD" if row["label"] == 1 else "HC"
        path = os.path.join(DATA_ROOT, cls_dir, row["filename"])
        if not os.path.exists(path):
            continue
            
        y_audio, sr = librosa.load(path, sr=16000, mono=True)
        
        # Create 3 versions
        y_base = y_audio.copy()
        y_vm   = mask_audio(y_audio, sr, "voiced")
        y_uvm  = mask_audio(y_audio, sr, "unvoiced")
        
        # Pad to MAX_LEN
        def pad(a):
            if len(a) < MAX_LEN: return np.pad(a, (0, MAX_LEN - len(a)))
            return a[:MAX_LEN]
            
        y_base = pad(y_base)
        y_vm = pad(y_vm)
        y_uvm = pad(y_uvm)
        
        feats = torch.tensor([row[f] for f in FEATURES], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            prob_base = torch.softmax(model(torch.tensor(y_base, dtype=torch.float32).unsqueeze(0).to(DEVICE), feats)[0], dim=-1)[0, 1].item()
            prob_vm   = torch.softmax(model(torch.tensor(y_vm, dtype=torch.float32).unsqueeze(0).to(DEVICE), feats)[0], dim=-1)[0, 1].item()
            prob_uvm  = torch.softmax(model(torch.tensor(y_uvm, dtype=torch.float32).unsqueeze(0).to(DEVICE), feats)[0], dim=-1)[0, 1].item()
            
        results["baseline"].append(prob_base)
        results["voiced_masked"].append(prob_vm)
        results["unvoiced_masked"].append(prob_uvm)
        results["y_true"].append(row["label"])
        
    y_true = np.array(results["y_true"])
    auc_base = roc_auc_score(y_true, results["baseline"])
    auc_vm = roc_auc_score(y_true, results["voiced_masked"])
    auc_uvm = roc_auc_score(y_true, results["unvoiced_masked"])
    
    print("\n--- Masked Sensitivity Test Results ---")
    print(f"Baseline AUC              : {auc_base:.3f}")
    print(f"Voiced Masked AUC         : {auc_vm:.3f} (Drop: {auc_base - auc_vm:.3f})")
    print(f"Unvoiced Masked AUC       : {auc_uvm:.3f} (Drop: {auc_base - auc_uvm:.3f})")
    print("---------------------------------------")
    
    if (auc_base - auc_vm) > (auc_base - auc_uvm):
        print("Conclusion: Model strongly attends to VOICED (vowel) segments, aligning with physiological expectations for dysarthria.")
    else:
        print("Conclusion: Model does not selectively attend to voiced segments.")

if __name__ == "__main__":
    run_attention_grounding_test()
