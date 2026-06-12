"""
explainability.py — Phase 5: Attention Weight Clinical Coherence Analysis
==========================================================================
Extracts and validates the PhysiologicalCrossAttention attention weights
against frame-level Shimmer measurements to prove the model is detecting
genuine PD dysarthria — NOT memorizing background noise.

BSPC Publication Deliverables:
  1. Attention heatmap overlaid on waveform for all 3 case archetypes
  2. Pearson correlation: attention weight magnitude vs frame-level Shimmer
     (r > 0.5 = clinically coherent, r > 0.7 = strongly coherent)
  3. Figure caption: "The cross-attention model attends significantly to 
     high-Shimmer temporal frames (r=X.XX, p<0.001), confirming the model 
     extracts PD-specific laryngeal amplitude perturbations."
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from typing import Optional, Tuple
import torch
import os


# ── Frame-Level Shimmer Extraction ────────────────────────────────────────────

def compute_frame_shimmer(audio: np.ndarray, sr: int, frame_len: int = 320) -> np.ndarray:
    """
    Compute local shimmer (amplitude perturbation quotient) per frame.
    
    Frame length = 320 samples @ 16kHz = 20ms, matching wav2vec frame rate.
    Shimmer = |A_i - A_{i+1}| / mean(A_i)  per consecutive frame pair.
    
    Higher shimmer = greater amplitude instability = PD dysarthric marker.
    """
    n_frames = len(audio) // frame_len
    amplitudes = []
    for i in range(n_frames):
        frame = audio[i * frame_len : (i + 1) * frame_len]
        amplitudes.append(np.mean(np.abs(frame)))

    amplitudes = np.array(amplitudes) + 1e-8   # Prevent division by zero
    shimmer    = np.abs(np.diff(amplitudes)) / amplitudes[:-1]

    # Pad to match n_frames length
    return np.append(shimmer, shimmer[-1])


# ── Attention Weight Extraction ───────────────────────────────────────────────

def get_attention_weights(model, audio: np.ndarray, classical_feats: np.ndarray) -> np.ndarray:
    """
    Run inference on a single audio sample and return attention weight array.
    
    Args:
        model: PDCrossAttentionClassifier (from cross_attention_fusion.py)
        audio: Raw audio numpy array (1D float32)
        classical_feats: [Shimmer, Jitter, HNR, F0] (1D array of length 4)

    Returns:
        attn: 1D array of shape (T,) — one weight per wav2vec frame
    """
    model.eval()
    with torch.no_grad():
        audio_tensor    = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)       # (1, L)
        feats_tensor    = torch.tensor(classical_feats, dtype=torch.float32).unsqueeze(0)  # (1, 4)
        _, attn_weights = model(audio_tensor, feats_tensor)

    attn = attn_weights[0, 0, :].numpy()   # (T,)
    return attn


# ── Pearson Coherence Test ─────────────────────────────────────────────────────

def attention_shimmer_correlation(
    attn_weights: np.ndarray,
    frame_shimmer: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Pearson correlation between attention weight magnitude and frame Shimmer.
    
    Interpretation for BSPC paper:
        r > 0.3 : Weak coherence (model somewhat attends to Shimmer frames)
        r > 0.5 : Moderate coherence — clinically coherent (minimum publishable)
        r > 0.7 : Strong coherence — highly clinically coherent (ideal result)

    Returns:
        (r_value, p_value)
    """
    # Resize to common length via linear interpolation
    target_len  = min(len(attn_weights), len(frame_shimmer))
    attn_resized = np.interp(
        np.linspace(0, 1, target_len),
        np.linspace(0, 1, len(attn_weights)),
        attn_weights,
    )
    shimmer_resized = frame_shimmer[:target_len]

    r, p = pearsonr(attn_resized, shimmer_resized)
    return float(r), float(p)


# ── Visualization: Attention Heatmap ──────────────────────────────────────────

def plot_attention_heatmap(
    audio: np.ndarray,
    sr: int,
    attn_weights: np.ndarray,
    frame_shimmer: np.ndarray,
    case_label: str = "Case Study",
    pd_risk: float = 0.0,
    save_path: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Generate publication-ready Figure 3 for BSPC paper.
    
    Layout (3-panel subplot):
      Panel 1: Raw audio waveform
      Panel 2: Cross-attention weights over time (the 'where model looks')
      Panel 3: Frame-level Shimmer values with attention shown as colour
    
    The visual immediately shows the model attends to high-Shimmer frames,
    providing intuitive clinical validation of the model's decision logic.

    Returns:
        (r_value, p_value) — Pearson correlation metrics
    """
    r, p = attention_shimmer_correlation(attn_weights, frame_shimmer)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(
        f"{case_label}  |  PD Risk: {pd_risk:.2f}  |  "
        f"Attention–Shimmer Coherence: r = {r:.3f} (p = {p:.4f})",
        fontsize=13, fontweight='bold'
    )
    gs = gridspec.GridSpec(3, 1, hspace=0.45)

    # Panel 1: Waveform
    ax1 = fig.add_subplot(gs[0])
    times = np.linspace(0, len(audio) / sr, num=len(audio))
    ax1.plot(times, audio, color="#4A90D9", linewidth=0.6, alpha=0.8)
    ax1.set_ylabel("Amplitude", fontsize=9)
    ax1.set_title("Raw Audio Waveform", fontsize=10)
    ax1.set_xlim([0, times[-1]])
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    # Panel 2: Attention Weights
    ax2 = fig.add_subplot(gs[1])
    attn_times = np.linspace(0, len(audio) / sr, num=len(attn_weights))
    ax2.fill_between(attn_times, attn_weights, alpha=0.75, color="#E24C3F")
    ax2.plot(attn_times, attn_weights, color="#C0392B", linewidth=1.0)
    ax2.set_ylabel("Attention Weight", fontsize=9)
    ax2.set_title("Cross-Attention Weight Profile", fontsize=10)
    ax2.set_xlim([0, len(audio) / sr])
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    # Panel 3: Shimmer with Attention Overlay
    ax3 = fig.add_subplot(gs[2])
    shimmer_times = np.linspace(0, len(audio) / sr, num=len(frame_shimmer))
    attn_interp   = np.interp(shimmer_times, attn_times, attn_weights)

    sc = ax3.scatter(shimmer_times, frame_shimmer, c=attn_interp,
                     cmap="YlOrRd", s=8, alpha=0.85, zorder=3)
    ax3.plot(shimmer_times, frame_shimmer, color="#888", linewidth=0.5, alpha=0.5, zorder=2)
    plt.colorbar(sc, ax=ax3, label="Attention Weight", shrink=0.7)
    ax3.set_ylabel("Shimmer", fontsize=9)
    ax3.set_xlabel("Time (s)", fontsize=9)
    ax3.set_title(f"Frame-Level Shimmer (coloured by Attention Weight, r={r:.3f})", fontsize=10)
    ax3.set_xlim([0, len(audio) / sr])
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Export CSV Data
        import pandas as pd
        df_attn = pd.DataFrame({
            "Time_s": shimmer_times,
            "Frame_Shimmer": frame_shimmer,
            "Attention_Weight": attn_interp
        })
        df_attn.to_csv(save_path.replace(".png", ".csv"), index=False)
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Figure saved: {save_path}")
    plt.show()

    return r, p


# ── Batch Coherence Report ─────────────────────────────────────────────────────

def generate_coherence_report(
    model,
    cases: list,   # list of dicts: {'audio', 'classical_feats', 'label', 'pd_risk', 'name'}
    sr: int = 16000,
    output_dir: str = "./outputs/explainability/",
) -> list:
    """
    Run coherence analysis on all case study archetypes.
    Generates individual attention heatmap figures + a summary table.
    
    Returns:
        List of dicts: {'case', 'r', 'p', 'coherence_level'}
    """
    report = []
    os.makedirs(output_dir, exist_ok=True)

    for case in cases:
        audio         = case["audio"]
        classical     = case["classical_feats"]
        label         = case["label"]
        risk          = case.get("pd_risk", 0.0)
        name          = case.get("name", "Unknown")

        attn    = get_attention_weights(model, audio, classical)
        shimmer = compute_frame_shimmer(audio, sr)
        fig_path = os.path.join(output_dir, f"attention_heatmap_{name}.png")

        r, p = plot_attention_heatmap(
            audio, sr, attn, shimmer,
            case_label=f"{name} ({label})",
            pd_risk=risk,
            save_path=fig_path,
        )

        coherence = "Strong" if r > 0.7 else "Moderate" if r > 0.5 else "Weak"
        report.append({"case": name, "label": label, "r": r, "p": p,
                       "coherence_level": coherence, "pd_risk": risk})
        print(f"  {name} ({label}): r={r:.3f}, p={p:.4f} — {coherence} coherence")

    return report
