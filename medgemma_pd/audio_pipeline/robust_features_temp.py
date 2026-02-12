import numpy as np
from scipy import signal

def autocorrelation(y):
    """
    Computes Autocorrelation Function (ACF) using FFT.
    """
    n = len(y)
    # Pad to avoids circular convolution artifacts
    y_pad = np.pad(y, (0, n), mode='constant')
    f = np.fft.fft(y_pad)
    acf = np.fft.ifft(f * np.conj(f)).real
    return acf[:n]

def extract_pitch_robust(y, sr, min_f0=75, max_f0=600):
    """
    Robust Pitch Detection using ACF.
    Returns: f0 (Hz), voiced_flag (bool)
    """
    # 1. ACF
    acf = autocorrelation(y)
    
    # 2. Find Peak in valid range
    min_lag = int(sr / max_f0)
    max_lag = int(sr / min_f0)
    
    if max_lag >= len(acf):
        max_lag = len(acf) - 1
        
    valid_acf = acf[min_lag:max_lag]
    if len(valid_acf) == 0:
        return 0.0, False
        
    peak_idx = np.argmax(valid_acf)
    peak_val = valid_acf[peak_idx]
    true_lag = min_lag + peak_idx
    
    # 3. Voiced/Unvoiced Decision (Voicing Strength)
    # Normalized ACF peak height (0.0 - 1.0)
    # HNR proxy
    energy = acf[0]
    if energy == 0: return 0.0, False
    
    voicing_strength = peak_val / energy
    
    # Threshold for speech (typically 0.3-0.4 for sustained vowels)
    if voicing_strength < 0.3:
        return 0.0, False # Unvoiced / Noise
        
    f0 = sr / true_lag
    return f0, True

def extract_jitter_shimmer_hnr(y, sr):
    """
    Extracts clinical metrics from a sustained vowel-like signal.
    """
    # 1. Pitch Contour (Frame-based analysis)
    frame_dur = 0.03 # 30ms frames
    hop_dur = 0.01   # 10ms hop
    frame_len = int(sr * frame_dur)
    hop_len = int(sr * hop_dur)
    
    num_frames = (len(y) - frame_len) // hop_len
    if num_frames < 5:
        return {"jitter": 0.0, "shimmer": 0.0, "hnr": 0.0, "f0": 0.0}
        
    f0s = []
    peaks = [] # Peak amplitudes for Shimmer
    
    for i in range(num_frames):
        start = i * hop_len
        frame = y[start : start + frame_len]
        
        # Apply window
        frame_w = frame * np.hanning(len(frame))
        
        f0, voiced = extract_pitch_robust(frame_w, sr)
        if voiced:
            f0s.append(f0)
            peaks.append(np.max(np.abs(frame))) # Simple peak amp
            
    if len(f0s) < 3:
         return {"jitter": 0.0, "shimmer": 0.0, "hnr": 0.0, "f0": 0.0}
         
    # 2. Calculate Metrics
    f0_arr = np.array(f0s)
    amp_arr = np.array(peaks)
    
    # Jitter (Local): Avg absolute difference between consecutive periods
    # Period = 1 / F0
    periods = 1.0 / (f0_arr + 1e-6)
    period_diffs = np.abs(np.diff(periods))
    avg_period = np.mean(periods)
    jitter = np.mean(period_diffs) / avg_period if avg_period > 0 else 0.0
    
    # Shimmer (Local): Avg absolute difference between consecutive amplitudes
    amp_diffs = np.abs(np.diff(amp_arr))
    avg_amp = np.mean(amp_arr)
    shimmer = np.mean(amp_diffs) / avg_amp if avg_amp > 0 else 0.0
    
    # HNR (Harmonic-to-Noise Ratio)
    # Estimate from overall ACF of the middle stable segment
    mid = len(y) // 2
    seg_len = min(len(y), 4096)
    segment = y[mid - seg_len//2 : mid + seg_len//2]
    
    acf = autocorrelation(segment)
    # Find main peak
    min_lag = int(sr / 600)
    max_lag = int(sr / 75)
    if max_lag < len(acf):
        peak_val = np.max(acf[min_lag:max_lag])
        total_energy = acf[0]
        
        # HNR = 10 * log10(HarmonicEnergy / NoiseEnergy)
        # HarmonicEnergy ~ PeakVal
        # NoiseEnergy ~ TotalEnergy - PeakVal
        if total_energy > peak_val:
            hnr = 10 * np.log10(peak_val / (total_energy - peak_val + 1e-6))
        else:
            hnr = 100.0 # Infinite HNR (Perfect)
    else:
        hnr = 0.0
        
    return {
        "jitter_local": jitter,
        "shimmer_local": shimmer,
        "hnr": hnr,
        "f0_mean": np.mean(f0_arr)
    }
