"""
augmentation.py — Phase 2: Data Augmentation for Low-Resource PD Speech
=========================================================================
Implements 5 scientifically-grounded augmentation strategies to expand an
N=120 dataset to ~500 effective training samples per fold:

  1. Time Stretching      — simulates motor symptom variability in speech rate
  2. Pitch Perturbation   — simulates inter-session laryngeal fatigue variation
  3. Additive Noise       — mPower-realistic ambient noise via MUSAN-style injection
  4. SpecAugment          — frequency + time masking on Mel-spectrograms
  5. Gain Jitter          — random amplitude scaling (microphone sensitivity variation)

CRITICAL: All augmentation is applied ONLY to training folds.
Testing folds are NEVER augmented. This prevents data leakage.
"""

import numpy as np
import librosa
import torch
import random
from typing import Tuple, Optional


# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_SR = 16000   # All audio standardized to 16kHz for wav2vec compatibility
N_MELS    = 80      # Mel bands for SpecAugment
HOP_LEN   = 160     # 10ms hop length at 16kHz
WIN_LEN   = 400     # 25ms window at 16kHz


# ── 1. Time Stretching ────────────────────────────────────────────────────────

def time_stretch(audio: np.ndarray, sr: int, rate_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """
    Randomly stretch or compress time axis by ±10%.
    Simulates the variability in PD patients' speech rate (festination/bradyphrenia).
    Phase-vocoder implementation via librosa preserves pitch (unlike naive resampling).
    """
    rate = random.uniform(*rate_range)
    return librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)


# ── 2. Pitch Perturbation ─────────────────────────────────────────────────────

def pitch_shift(audio: np.ndarray, sr: int, semitone_range: Tuple[float, float] = (-2.0, 2.0)) -> np.ndarray:
    """
    Randomly shift pitch by ±2 semitones without altering duration.
    Simulates inter-session laryngeal fatigue and microphone placement variation.
    ±2 semitones is capped to stay within clinically plausible F0 deviation range.
    """
    n_steps = random.uniform(*semitone_range)
    return librosa.effects.pitch_shift(audio.astype(np.float32), sr=sr, n_steps=n_steps)


# ── 3. Additive Noise Injection ───────────────────────────────────────────────

def add_background_noise(
    audio: np.ndarray,
    sr: int,
    snr_db_range: Tuple[float, float] = (5.0, 15.0),
    noise_type: str = "gaussian",
    noise_audio: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Inject background noise at a controlled SNR (5–15 dB).
    This simulates the ambient noise conditions of real-world mPower recordings.

    Args:
        noise_type: "gaussian" (default) or "real" (pass noise_audio).
        noise_audio: Optional real-world noise segment (e.g., from MUSAN corpus).
        snr_db_range: SNR range in dB for noise injection.

    SNR formula: Noise power = Signal power / 10^(SNR_dB / 10)
    """
    target_snr = random.uniform(*snr_db_range)
    signal_power = np.mean(audio ** 2)

    if noise_type == "real" and noise_audio is not None:
        # Crop or loop noise to match audio length
        n_repeats = int(np.ceil(len(audio) / len(noise_audio)))
        noise = np.tile(noise_audio, n_repeats)[:len(audio)]
    else:
        noise = np.random.randn(len(audio))

    noise_power = np.mean(noise ** 2)
    target_noise_power = signal_power / (10 ** (target_snr / 10))
    scale = np.sqrt(target_noise_power / (noise_power + 1e-8))
    noisy = audio + scale * noise

    # Normalise to prevent clipping
    max_val = np.max(np.abs(noisy))
    if max_val > 0:
        noisy = noisy / max_val * 0.95
    return noisy.astype(np.float32)


# ── 4. SpecAugment ────────────────────────────────────────────────────────────

def specaugment(
    audio: np.ndarray,
    sr: int,
    freq_mask_max: int = 10,
    time_mask_max: int = 30,
    n_freq_masks: int = 2,
    n_time_masks: int = 2,
) -> np.ndarray:
    """
    Apply SpecAugment (Park et al., 2019) to the Mel-spectrogram domain,
    then reconstruct audio via Griffin-Lim. 

    Frequency masking: randomly zeros F consecutive Mel bins — forces model 
    to learn dysarthria from partial spectral information, improving robustness.
    Time masking: randomly zeros T consecutive time frames — simulates
    intermittent signal dropouts common in mobile device recordings.
    """
    # Compute Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS,
                                         hop_length=HOP_LEN, win_length=WIN_LEN)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Frequency masking
    for _ in range(n_freq_masks):
        f = random.randint(0, freq_mask_max)
        f0 = random.randint(0, max(N_MELS - f, 1))
        mel_db[f0:f0 + f, :] = mel_db.min()

    # Time masking
    T = mel_db.shape[1]
    for _ in range(n_time_masks):
        t = random.randint(0, min(time_mask_max, T))
        t0 = random.randint(0, max(T - t, 1))
        mel_db[:, t0:t0 + t] = mel_db.min()

    # Convert back to linear scale and reconstruct via Griffin-Lim
    mel_power = librosa.db_to_power(mel_db)
    # Griffin-Lim inversion from mel requires STFT inversion approximation
    stft_approx = librosa.feature.inverse.mel_to_stft(mel_power, sr=sr, n_fft=WIN_LEN)
    reconstructed = librosa.griffinlim(stft_approx, hop_length=HOP_LEN, win_length=WIN_LEN)

    # Match output length to input
    if len(reconstructed) > len(audio):
        return reconstructed[:len(audio)].astype(np.float32)
    return np.pad(reconstructed, (0, len(audio) - len(reconstructed))).astype(np.float32)


# ── 5. Gain Jitter ────────────────────────────────────────────────────────────

def gain_jitter(audio: np.ndarray, gain_db_range: Tuple[float, float] = (-3.0, 3.0)) -> np.ndarray:
    """
    Apply random amplitude gain in dB range.
    Simulates microphone sensitivity variation across mPower recording sessions.
    """
    gain_db = random.uniform(*gain_db_range)
    gain = 10 ** (gain_db / 20.0)
    augmented = audio * gain
    # Clip to prevent overflow
    return np.clip(augmented, -1.0, 1.0).astype(np.float32)


# ── Augmentation Strategy Selector ────────────────────────────────────────────

AUGMENTATION_STRATEGIES = {
    "time_stretch"  : lambda a, sr: time_stretch(a, sr),
    "pitch_shift"   : lambda a, sr: pitch_shift(a, sr),
    "noise_inject"  : lambda a, sr: add_background_noise(a, sr),
    "specaugment"   : lambda a, sr: specaugment(a, sr),
    "gain_jitter"   : lambda a, sr: gain_jitter(a),
}


def augment_sample(
    audio: np.ndarray,
    sr: int,
    strategies: Optional[list] = None,
    n_augments: int = 4,
) -> list:
    """
    Generate n_augments augmented versions of a single audio sample.
    Each augmented version applies one randomly selected strategy, ensuring diversity.

    Args:
        strategies: List of strategy names. Defaults to all 5.
        n_augments: Number of augmented copies to generate.

    Returns:
        List of (augmented_audio, strategy_name) tuples.
    """
    if strategies is None:
        strategies = list(AUGMENTATION_STRATEGIES.keys())

    results = []
    selected = random.choices(strategies, k=n_augments)
    for strat_name in selected:
        fn = AUGMENTATION_STRATEGIES[strat_name]
        try:
            aug_audio = fn(audio, sr)
            results.append((aug_audio, strat_name))
        except Exception as e:
            print(f"  [AUG WARN] {strat_name} failed: {e}")
    return results


def augment_fold(
    X_audio: list,
    y: np.ndarray,
    sr: int,
    n_augments_per_sample: int = 4,
) -> Tuple[list, np.ndarray]:
    """
    Augment all training samples in a CV fold.
    Returns augmented X_audio and y arrays merged with originals.

    Args:
        X_audio: List of raw audio numpy arrays (training fold only).
        y: Label array aligned with X_audio.
        n_augments_per_sample: How many augmented copies per sample (default=4).
                               With N=96 train samples × 4 augments = 480 total.
    Returns:
        (augmented_list, augmented_labels)
    """
    aug_X, aug_y = list(X_audio), list(y)
    for audio, label in zip(X_audio, y):
        augmented = augment_sample(audio, sr, n_augments=n_augments_per_sample)
        for aug_audio, _ in augmented:
            aug_X.append(aug_audio)
            aug_y.append(label)

    return aug_X, np.array(aug_y)
