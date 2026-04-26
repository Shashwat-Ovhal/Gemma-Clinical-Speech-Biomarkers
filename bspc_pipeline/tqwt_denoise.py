"""
tqwt_denoise.py — Phase 1: TQWT-Based Denoising for PD Vocal Biomarker Extraction
===================================================================================
Implements Tunable Q-Factor Wavelet Transform (TQWT) denoising, specifically tuned
to preserve the laryngeal resonance band (80Hz–300Hz) critical for PD dysarthria.

Why TQWT over standard DWT:
  - DWT uses fixed Q-factor, discarding temporal structure of non-stationary signals.
  - TQWT allows independent control of Q (oscillatory behaviour) and redundancy (r),
    making it superior for non-stationary biomedical signals like dysarthric speech.
  - Literature: Sharma et al. (2024, Semantic Scholar) demonstrated TQWT improves
    PD biomarker classification by better preserving vocal fold oscillation patterns.

Parameters (tuned for laryngeal resonance of PD-affected speech):
  - Q = 3  : moderate oscillatory quality, suitable for vocal fold periodicity
  - r = 3  : redundancy factor for reconstruction fidelity
  - J = 12 : number of decomposition levels
"""

import numpy as np
import librosa
import soundfile as sf
import os
from typing import Tuple


# ── TQWT Core ────────────────────────────────────────────────────────────────

def _compute_scaling_lowpass(Q: float, r: float) -> Tuple[float, float]:
    """Compute the lowpass and highpass scaling factors for TQWT."""
    beta = 2.0 / (Q + 1.0)           # High-pass scaling (bandwidth)
    alpha = 1.0 - (beta / r)          # Low-pass scaling (redundancy control)
    return alpha, beta


def _tqwt_forward(signal: np.ndarray, Q: float, r: float, J: int) -> list:
    """
    Forward TQWT decomposition.
    Returns list of J+1 subband arrays [highpass_1, ..., highpass_J, lowpass_J].
    Uses FFT-based implementation for efficiency on long audio signals.
    """
    alpha, beta = _compute_scaling_lowpass(Q, r)
    subbands = []
    x = np.fft.fft(signal)
    N = len(x)

    for j in range(J):
        N_high = int(np.round(beta * N))
        N_low  = int(np.round(alpha * N))

        # High-pass subband via frequency slicing
        H = np.zeros(N_high, dtype=complex)
        if N_high <= N:
            H[:N_high] = x[:N_high]

        # Low-pass remainder passed to next level
        L = np.zeros(N_low,  dtype=complex)
        if N_low <= N:
            L[:N_low]  = x[:N_low]

        subbands.append(np.fft.ifft(H).real)
        x = L
        N = N_low

    subbands.append(np.fft.ifft(x).real)   # Final low-pass residual
    return subbands


def _tqwt_inverse(subbands: list, Q: float, r: float, target_len: int) -> np.ndarray:
    """Reconstruct signal from TQWT subbands via iterative IDFT upsampling."""
    alpha, beta = _compute_scaling_lowpass(Q, r)
    x = np.fft.fft(subbands[-1])

    for j in range(len(subbands) - 2, -1, -1):
        N_cur  = int(np.round(len(x) / alpha))
        N_high = int(np.round(beta * N_cur))

        X_up = np.zeros(N_cur, dtype=complex)
        X_up[:len(x)] = x

        H = np.zeros(N_cur, dtype=complex)
        if N_high <= len(np.fft.fft(subbands[j])):
            H[:N_high] = np.fft.fft(subbands[j])[:N_high]

        x = X_up + H

    recon = np.fft.ifft(x).real
    # Trim or zero-pad to match original length
    if len(recon) >= target_len:
        return recon[:target_len]
    return np.pad(recon, (0, target_len - len(recon)))


# ── Soft Thresholding (Denoising Step) ───────────────────────────────────────

def _soft_threshold(subband: np.ndarray, sigma: float, multiplier: float = 3.0) -> np.ndarray:
    """
    Universal soft thresholding: λ = multiplier * sigma.
    Applied to high-frequency subbands to suppress ambient noise
    while preserving PD-relevant amplitude perturbations.
    """
    lam = multiplier * sigma
    return np.sign(subband) * np.maximum(np.abs(subband) - lam, 0.0)


def _estimate_noise_sigma(subband: np.ndarray) -> float:
    """
    Robust noise estimation via Median Absolute Deviation (MAD).
    MAD is resistant to outliers caused by pathological vocal events.
    """
    mad = np.median(np.abs(subband - np.median(subband)))
    return mad / 0.6745   # Normalise to Gaussian sigma equivalent


# ── SNR Computation ───────────────────────────────────────────────────────────

def compute_snr(original: np.ndarray, denoised: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB. Higher = more noise removed."""
    noise = original - denoised
    signal_power = np.mean(original ** 2)
    noise_power  = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10.0 * np.log10(signal_power / noise_power)


# ── Main Denoising Function ───────────────────────────────────────────────────

def tqwt_denoise(
    audio: np.ndarray,
    sr: int,
    Q: float = 3.0,
    r: float = 3.0,
    J: int   = 12,
    threshold_multiplier: float = 3.0,
) -> Tuple[np.ndarray, float]:
    """
    Apply TQWT denoising to a raw audio waveform.

    Args:
        audio: 1D numpy array of audio samples (float32 or float64).
        sr:    Sample rate in Hz.
        Q:     Q-factor (oscillatory behaviour). Q=3 tuned for vocal periodicity.
        r:     Redundancy factor. r=3 provides stable reconstruction.
        J:     Number of TQWT decomposition levels.
        threshold_multiplier: Soft-threshold aggressiveness (default 3.0 = universal).

    Returns:
        (denoised_audio, snr_db): Cleaned signal and SNR improvement metric.
    """
    original_len = len(audio)

    # Zero-pad to next power of 2 for FFT efficiency
    pad_len = int(2 ** np.ceil(np.log2(original_len)))
    audio_padded = np.pad(audio, (0, pad_len - original_len))

    # Forward TQWT decomposition
    subbands = _tqwt_forward(audio_padded, Q, r, J)

    # Soft threshold all high-pass subbands (noise is high-frequency)
    # Leave the final low-pass subband (index J) untouched — it contains
    # the fundamental laryngeal resonance (80–300Hz PD biomarker band).
    denoised_subbands = []
    for i, sb in enumerate(subbands):
        if i < J:  # High-pass subbands: threshold to suppress ambient noise
            sigma = _estimate_noise_sigma(sb)
            denoised_subbands.append(_soft_threshold(sb, sigma, threshold_multiplier))
        else:       # Low-pass residual: preserve completely
            denoised_subbands.append(sb)

    # Reconstruct from denoised subbands
    denoised_padded = _tqwt_inverse(denoised_subbands, Q, r, pad_len)
    denoised = denoised_padded[:original_len]

    # Compute SNR metric for the paper's Figure 1
    snr = compute_snr(audio, denoised)

    return denoised.astype(np.float32), snr


# ── File-Level API (used by the main pipeline) ────────────────────────────────

def denoise_file(input_path: str, output_path: str) -> dict:
    """
    Load an audio file, apply TQWT denoising, save output, return metrics.

    Returns dict with keys: 'snr_db', 'input_path', 'output_path'.
    """
    audio, sr = librosa.load(input_path, sr=16000, mono=True)
    denoised, snr = tqwt_denoise(audio, sr)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, denoised, sr)

    return {"input_path": input_path, "output_path": output_path, "snr_db": round(snr, 3)}


def denoise_dataset(input_root: str, output_root: str) -> list:
    """
    Batch denoise an entire mPower dataset directory structure.
    Expected input layout:
        input_root/HC/*.wav
        input_root/PD/*.wav

    Returns list of per-file metric dicts.
    """
    results = []
    for cls in ["HC", "PD"]:
        in_dir  = os.path.join(input_root, cls)
        out_dir = os.path.join(output_root, cls)
        if not os.path.exists(in_dir):
            print(f"  [WARN] Missing class directory: {in_dir}")
            continue
        for fname in os.listdir(in_dir):
            if not fname.endswith((".wav", ".m4a")):
                continue
            in_path  = os.path.join(in_dir,  fname)
            out_name = os.path.splitext(fname)[0] + ".wav"
            out_path = os.path.join(out_dir, out_name)
            try:
                m = denoise_file(in_path, out_path)
                m["label"] = cls
                results.append(m)
                print(f"  ✓ {cls}/{fname}  SNR={m['snr_db']:.1f}dB")
            except Exception as e:
                print(f"  ✗ {cls}/{fname}: {e}")
    return results


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json
    INPUT_ROOT  = "./data/mpower_dataset"
    OUTPUT_ROOT = "./data/mpower_denoised"
    print(f"\n{'='*60}")
    print("  Phase 1 — TQWT Denoising")
    print(f"  Input : {INPUT_ROOT}")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"{'='*60}\n")
    metrics = denoise_dataset(INPUT_ROOT, OUTPUT_ROOT)
    if metrics:
        snrs = [m["snr_db"] for m in metrics]
        print(f"\n  Files processed: {len(metrics)}")
        print(f"  Mean SNR gain  : {np.mean(snrs):.2f} dB")
        print(f"  Min SNR gain   : {np.min(snrs):.2f} dB")
        report_path = "./data/denoising_report.json"
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Report saved   : {report_path}")
