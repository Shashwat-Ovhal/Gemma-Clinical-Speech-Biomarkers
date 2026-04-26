import os
import sys
import numpy as np
import librosa
import pandas as pd
from scipy.signal import butter, filtfilt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bspc_pipeline.tqwt_denoise import tqwt_denoise, compute_snr

DATA_ROOT = "./data/mpower_dataset"
BENCHMARK_OUTPUT = "./outputs/bspc/denoising_benchmark.csv"

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def compute_lsd(original, processed, sr=16000):
    """Computes Log-Spectral Distance (LSD) between two audio signals."""
    S_orig = np.abs(librosa.stft(original))
    S_proc = np.abs(librosa.stft(processed))
    
    # Avoid log of zero
    S_orig = np.maximum(S_orig, 1e-10)
    S_proc = np.maximum(S_proc, 1e-10)
    
    lsd = np.mean(np.sqrt(np.mean((10 * np.log10(S_orig / S_proc)) ** 2, axis=0)))
    return lsd

def run_benchmark(n_samples=20):
    print(f"Running Denoising Benchmark on {n_samples} random samples...")
    results = []
    
    # Gather samples
    all_files = []
    for cls in ["HC", "PD"]:
        cls_path = os.path.join(DATA_ROOT, cls)
        if not os.path.exists(cls_path): continue
        for fname in os.listdir(cls_path):
            if fname.endswith((".wav", ".m4a")):
                all_files.append((cls, os.path.join(cls_path, fname)))
    
    if len(all_files) == 0:
        print("No files found in", DATA_ROOT)
        return
        
    np.random.seed(42)
    idx = np.random.choice(len(all_files), min(n_samples, len(all_files)), replace=False)
    selected_files = [all_files[i] for i in idx]
    
    for cls, path in selected_files:
        try:
            y, sr = librosa.load(path, sr=16000, mono=True)
            
            # 1. Bandpass Filter (50Hz - 3000Hz)
            y_bandpass = butter_bandpass_filter(y, 50.0, 3000.0, sr)
            snr_bp = compute_snr(y, y_bandpass)
            lsd_bp = compute_lsd(y, y_bandpass, sr)
            
            # 2. TQWT Filter
            y_tqwt, snr_tqwt = tqwt_denoise(y, sr)
            lsd_tqwt = compute_lsd(y, y_tqwt, sr)
            
            results.append({
                "Filename": os.path.basename(path),
                "Class": cls,
                "Bandpass_SNR_Gain_dB": snr_bp,
                "Bandpass_LSD": lsd_bp,
                "TQWT_SNR_Gain_dB": snr_tqwt,
                "TQWT_LSD": lsd_tqwt
            })
        except Exception as e:
            print(f"Failed on {path}: {e}")
            
    df = pd.DataFrame(results)
    
    # Calculate means, omitting infs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    means = df.mean(numeric_only=True)
    
    print("\n--- Benchmark Results ---")
    print(f"Average Bandpass SNR Gain: {means['Bandpass_SNR_Gain_dB']:.2f} dB")
    print(f"Average TQWT SNR Gain:     {means['TQWT_SNR_Gain_dB']:.2f} dB")
    print(f"Average Bandpass LSD:      {means['Bandpass_LSD']:.2f}")
    print(f"Average TQWT LSD:          {means['TQWT_LSD']:.2f}")
    print("-------------------------")
    
    os.makedirs(os.path.dirname(BENCHMARK_OUTPUT), exist_ok=True)
    df.to_csv(BENCHMARK_OUTPUT, index=False)
    print(f"Full results saved to {BENCHMARK_OUTPUT}")

if __name__ == "__main__":
    run_benchmark(20)
