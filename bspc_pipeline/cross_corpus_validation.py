"""
cross_corpus_validation.py — Phase 7: Zero-Shot Cross-Corpus Generalizability
===============================================================================
Validates that the mPower-trained PDCrossAttentionClassifier generalises to an 
entirely independent, lab-grade dataset: MDVR-KCL (King's College London).

Experimental Design:
    TRAIN  →  mPower (unconstrained, ambient smartphone audio, N=120)
    TEST   →  MDVR-KCL ReadText (clean, lab-recorded, 21 HC + 16 PD, N=37)

    NO retraining. NO fine-tuning. The model weights are frozen after mPower
    training and applied directly to KCL audio. This is "Zero-Shot" transfer.

Why MDVR-KCL is better than PC-GITA for this study:
  - English corpus → avoids language confound (PC-GITA is Spanish)
  - KCL is lab-recorded → extreme acoustic contrast with mPower, maximally 
    challenging the model to prove it learned pathology not ambient noise
  - UPDRS severity scores in filename (e.g., ID06_pd_3_1_1.wav → UPDRS=3)
    giving us bonus severity-correlation analysis at no extra cost

BSPC Deliverable:
  Table 5 — Zero-Shot Cross-Corpus Transfer Results
    Dataset | N  | AUC [95% CI]    | Acc  | Sens | Spec
    mPower  | 120| 0.XXX [lo-hi]   | ...  | ...  | ...
    KCL     | 37 | 0.XXX [lo-hi]   | ...  | ...  | ...

  Result interpretation:
    If KCL AUC ≥ 0.70: Strong evidence model learned PD dysarthria, not noise.
    If KCL AUC < 0.65: Model may be partially tracking mPower-specific noise.
"""

import os
import json
import numpy as np
import librosa
import torch
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report,
)
from typing import Tuple, Optional

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bspc_pipeline.tqwt_denoise import tqwt_denoise
from bspc_pipeline.cross_attention_fusion import PDCrossAttentionClassifier

try:
    from medgemma_pd.audio_pipeline.preprocessing import AudioPreprocessor
    from medgemma_pd.audio_pipeline.features import FeatureExtractor
except ImportError:
    print("CRITICAL: Run from project root directory.")
    sys.exit(1)


# ── KCL Dataset Paths ─────────────────────────────────────────────────────────

KCL_ROOT    = r"./dataset- MDVR-KCL Dataset/26_29_09_2017_KCL/26-29_09_2017_KCL/ReadText"
TARGET_SR   = 16000
FEATURES    = ["shimmer", "jitter", "hnr", "f0_std"]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ── UPDRS Severity Extraction from KCL Filename ───────────────────────────────

def parse_kcl_metadata(filename: str) -> dict:
    """
    Parse KCL filename convention to extract UPDRS severity metadata.

    Filename format: ID{subj}_{label}_{updrs_speech}_{updrs_face}_{updrs_rigid}.wav
    Example: ID06_pd_3_1_1.wav → PD patient, UPDRS speech severity = 3

    UPDRS speech severity 0 = no impairment, 4 = severe impairment.
    This allows us to perform a bonus severity-correlation analysis.
    """
    name = os.path.splitext(filename)[0]
    parts = name.split("_")
    return {
        "subject_id"   : parts[0],
        "label_str"    : parts[1] if len(parts) > 1 else "unk",
        "label"        : 1 if (len(parts) > 1 and parts[1] == "pd") else 0,
        "updrs_speech" : int(parts[2]) if len(parts) > 2 else -1,
        "updrs_face"   : int(parts[3]) if len(parts) > 3 else -1,
        "updrs_rigid"  : int(parts[4]) if len(parts) > 4 else -1,
    }


# ── KCL Audio + Feature Loader ────────────────────────────────────────────────

def load_kcl_sample(wav_path: str) -> Optional[dict]:
    """
    Load a single KCL wav file, apply TQWT denoising, and extract classical features.

    Returns dict with:
        'audio'          : denoised numpy array (float32, sr=16000)
        'classical_feats': [shimmer, jitter, hnr, f0_std] numpy array
        'metadata'       : parsed KCL filename metadata dict
        'valid'          : bool — False if voice activity detection failed
    """
    filename = os.path.basename(wav_path)
    metadata = parse_kcl_metadata(filename)

    try:
        # Load and resample
        audio, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

        # Apply TQWT denoising (same pipeline as mPower training)
        audio_denoised, snr = tqwt_denoise(audio, sr)

        # Extract classical features
        y, sr_proc, _ = AudioPreprocessor.process(wav_path)
        feats = FeatureExtractor.extract_features(y, sr_proc)

        if not feats.get("valid_voice_detected", False):
            print(f"    [SKIP] {filename}: no valid voice detected")
            return None

        classical = np.array([
            feats.get("shimmer_local", 0.0) * 100,
            feats.get("jitter_local",  0.0) * 100,
            feats.get("hnr",           0.0),
            feats.get("f0_std",        0.0),
        ], dtype=np.float32)

        return {
            "audio"           : audio_denoised,
            "classical_feats" : classical,
            "metadata"        : metadata,
            "snr_db"          : snr,
            "valid"           : True,
        }
    except Exception as e:
        print(f"    [ERROR] {filename}: {e}")
        return None


def load_kcl_dataset(kcl_root: str = KCL_ROOT) -> Tuple[list, list, list]:
    """
    Load all KCL samples from HC/ and PD/ subdirectories.

    Returns:
        samples   : list of dicts from load_kcl_sample()
        labels    : list of int (0=HC, 1=PD)
        metadata  : list of metadata dicts
    """
    samples, labels, meta_list = [], [], []
    for cls_dir, lbl in [("HC", 0), ("PD", 1)]:
        cls_path = os.path.join(kcl_root, cls_dir)
        if not os.path.exists(cls_path):
            print(f"  [WARN] KCL class dir not found: {cls_path}")
            continue
        files = sorted([f for f in os.listdir(cls_path) if f.endswith(".wav")])
        print(f"  Loading KCL {cls_dir}: {len(files)} files...")
        for fname in files:
            fpath  = os.path.join(cls_path, fname)
            sample = load_kcl_sample(fpath)
            if sample and sample["valid"]:
                samples.append(sample)
                labels.append(lbl)
                meta_list.append(sample["metadata"])
    return samples, labels, meta_list


# ── Zero-Shot Inference ───────────────────────────────────────────────────────

def run_zero_shot_inference(
    model: PDCrossAttentionClassifier,
    samples: list,
    labels: list,
    max_audio_len: int = 48000,   # 3 seconds at 16kHz
) -> dict:
    """
    Run zero-shot inference on KCL samples using the mPower-trained model.
    No weight updates. No fine-tuning. Pure cross-corpus transfer.

    Args:
        model   : mPower-trained PDCrossAttentionClassifier (fully loaded)
        samples : KCL samples list from load_kcl_dataset()
        labels  : Ground-truth labels (0=HC, 1=PD)
        max_audio_len: Truncate/pad audio to this many samples (for batch homogeneity)

    Returns:
        results dict with AUC, accuracy, sensitivity, specificity, F1, CI
    """
    model.eval().to(DEVICE)
    y_true, y_prob, y_pred = [], [], []

    for i, (sample, label) in enumerate(zip(samples, labels)):
        audio = sample["audio"]
        feats = sample["classical_feats"]

        # Pad or truncate to max_audio_len for wav2vec compatibility
        if len(audio) < max_audio_len:
            audio = np.pad(audio, (0, max_audio_len - len(audio)))
        else:
            audio = audio[:max_audio_len]

        audio_t = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)    # (1, L)
        feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)    # (1, 4)

        with torch.no_grad():
            logits, _ = model(audio_t, feats_t)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            pred = int(prob >= 0.5)

        y_true.append(label)
        y_prob.append(prob)
        y_pred.append(pred)
        print(f"  [{i+1:02d}] {sample['metadata']['subject_id']} "
              f"({sample['metadata']['label_str'].upper()}) "
              f"→ P(PD)={prob:.3f}  pred={'PD' if pred else 'HC'}")

    # --- Metrics ---
    auc  = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Bootstrap 95% CI for AUC
    rng = np.random.default_rng(42)
    boot_aucs = []
    for _ in range(1000):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt  = np.array(y_true)[idx]
        yp  = np.array(y_prob)[idx]
        if len(set(yt)) > 1:
            boot_aucs.append(roc_auc_score(yt, yp))
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5]) if boot_aucs else (0, 0)

    return {
        "dataset"    : "MDVR-KCL",
        "n_samples"  : len(y_true),
        "n_pd"       : sum(y_true),
        "n_hc"       : sum(1 for l in y_true if l == 0),
        "auc"        : round(auc,  3),
        "ci_lo"      : round(ci_lo, 3),
        "ci_hi"      : round(ci_hi, 3),
        "accuracy"   : round(acc,  3),
        "sensitivity": round(sens, 3),
        "specificity": round(spec, 3),
        "f1"         : round(f1,   3),
        "y_true"     : y_true,
        "y_prob"     : y_prob,
    }


# ── Severity Correlation Analysis (Bonus BSPC Section) ────────────────────────

def severity_correlation(
    samples: list,
    y_prob: list,
    labels: list,
) -> dict:
    """
    Pearson correlation between model PD risk probability and UPDRS speech severity.
    BSPC reviewers love severity correlation because it proves the model is 
    tracking disease progression, not just a binary sick/healthy switch.

    Only meaningful for PD samples where UPDRS score is known (>= 0).
    """
    from scipy.stats import pearsonr, spearmanr

    pd_probs, pd_updrs = [], []
    for sample, prob, lbl in zip(samples, y_prob, labels):
        u = sample["metadata"].get("updrs_speech", -1)
        if lbl == 1 and u >= 0:
            pd_probs.append(prob)
            pd_updrs.append(u)

    if len(pd_probs) < 3:
        return {"pearson_r": None, "pearson_p": None, "spearman_r": None, "n_pd": len(pd_probs)}

    pr, pp = pearsonr(pd_probs, pd_updrs)
    sr, sp = spearmanr(pd_probs, pd_updrs)
    return {
        "n_pd_with_severity" : len(pd_probs),
        "pearson_r"          : round(pr, 4),
        "pearson_p"          : round(pp, 4),
        "spearman_r"         : round(sr, 4),
        "spearman_p"         : round(sp, 4),
    }


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_phase7(
    model_checkpoint: str,
    output_dir: str = "./outputs/bspc",
) -> dict:
    """
    Phase 7 entry point. Loads trained model, runs zero-shot KCL inference.

    Args:
        model_checkpoint: Path to saved mPower-trained model .pt file
        output_dir: Directory to save results

    Returns:
        Combined results dict ready for BSPC Table 5
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print("  Phase 7 — Zero-Shot Cross-Corpus Validation (MDVR-KCL)")
    print(f"{'='*60}\n")

    # Load KCL dataset
    print("Loading MDVR-KCL dataset...")
    samples, labels, metadata = load_kcl_dataset()
    print(f"  KCL samples loaded: {len(samples)} ({sum(labels)} PD, {sum(1 for l in labels if l==0)} HC)\n")

    if len(samples) < 5:
        print("  [WARN] Too few samples for meaningful evaluation. Check dataset path.")
        return {}

    # Load trained model
    print("Loading mPower-trained model...")
    model = PDCrossAttentionClassifier(n_classical_features=4)
    if os.path.exists(model_checkpoint):
        state = torch.load(model_checkpoint, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        print(f"  Model loaded from: {model_checkpoint}")
    else:
        print(f"  [WARN] Checkpoint not found at {model_checkpoint}.")
        print(f"  Running with untrained weights (for sanity-check only).")

    # Zero-shot inference
    print("\nRunning zero-shot inference on KCL...\n")
    results = run_zero_shot_inference(model, samples, labels)

    # Severity correlation
    sev_corr = severity_correlation(samples, results["y_prob"], labels)
    results["severity_correlation"] = sev_corr

    # Clean y_true / y_prob from serializable results
    serializable = {k: v for k, v in results.items() if k not in ["y_true", "y_prob"]}

    # Save
    out_path = os.path.join(output_dir, "phase7_kcl_results.json")
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    # Print summary
    print(f"\n  {'='*50}")
    print(f"  MDVR-KCL Zero-Shot Results:")
    print(f"  {'='*50}")
    print(f"  Samples     : {results['n_samples']} ({results['n_pd']} PD / {results['n_hc']} HC)")
    print(f"  AUC         : {results['auc']} [{results['ci_lo']}–{results['ci_hi']}]")
    print(f"  Accuracy    : {results['accuracy']}")
    print(f"  Sensitivity : {results['sensitivity']}")
    print(f"  Specificity : {results['specificity']}")
    print(f"  F1 Score    : {results['f1']}")
    if sev_corr.get("pearson_r") is not None:
        print(f"\n  UPDRS Severity Correlation (PD only):")
        print(f"  Pearson r   : {sev_corr['pearson_r']} (p={sev_corr['pearson_p']})")
        print(f"  Spearman r  : {sev_corr['spearman_r']} (p={sev_corr['spearman_p']})")
    print(f"\n  Results saved → {out_path}")

    return results


if __name__ == "__main__":
    CHECKPOINT = "./medgemma_pd/models/bspc_model_teacher.pt"
    run_phase7(CHECKPOINT)
