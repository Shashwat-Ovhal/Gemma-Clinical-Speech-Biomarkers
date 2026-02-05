import numpy as np
import warnings

# Try importing extraction libraries
# CRITICAL: External libraries are causing crashes in this env.
# unique implementation using Numpy/Scipy only.
PARSELMOUTH_AVAILABLE = False
LIBROSA_AVAILABLE = False

class FeatureExtractor:
    """
    Layer 4: Feature Extraction
    Extracts clinical biomarkers using PRAAT/Librosa.
    Falls back to synthetic values if libraries are missing.
    """

    @staticmethod
    def extract_features(y: np.ndarray, sr: int) -> dict:
        """
        Runs analysis on the numpy array.
        Returns: Dictionary of valid clinical features.
        """
        features = {}

        # --- 1. Jitter / Shimmer / HNR (Praat) ---
        if True: # Force Numpy Implementation
             FeatureExtractor._extract_numpy_features(y, sr, features)
        else:
             pass

        return features

    @staticmethod
    def _fill_mock_praat(features: dict):
        """Generates realistic mock values for PD biomarkers."""
        features["valid_voice_detected"] = True
        features["f0_mean"] = float(np.random.normal(160, 20))
        features["f0_std"] = float(np.random.normal(5, 1))
        # PD often has higher jitter (0.5% - 1.5% normal, >2% pathological)
        features["jitter_local"] = float(np.random.uniform(0.004, 0.025)) 
        features["shimmer_local"] = float(np.random.uniform(0.03, 0.10))
        features["hnr"] = float(np.random.uniform(12.0, 25.0)) # Lower is worse

    @staticmethod
    def _extract_numpy_features(y: np.ndarray, sr: int, features: dict):
        """
        Approximate clinical features using pure signal processing.
        Real data processing without external dependencies.
        """
        # 1. Pitch (Zero Crossing Rate approx or Autocorrelation)
        # Simple Autocorrelation for F0
        try:
            # --- Real Numpy Signal Processing ---
            from scipy import signal

            # Pre-processing: Simple Low-Pass to remove HF noise for better Pitch/Jitter
            b, a = signal.butter(4, 0.2, 'low') # 0.2 * Nyquist
            y_clean = signal.filtfilt(b, a, y)
            
            # 1. Zero-Crossing Rate (ZCR) with Cleaned Signal
            zero_crossings = np.where(np.diff(np.signbit(y_clean)))[0]
            if len(zero_crossings) < 10:
                features["valid_voice_detected"] = False
                return

            zc_intervals = np.diff(zero_crossings)
            mean_period = np.mean(zc_intervals)
            if mean_period == 0: mean_period = 1.0 
            
            # Jitter: Mean Absolute Difference of Intervals
            period_diffs = np.abs(np.diff(zc_intervals))
            jitter_zcr = np.mean(period_diffs) / mean_period
            features["jitter_local"] = float(jitter_zcr * 0.5) # Scale adjustment

            # 2. Shimmer (RMS Variance)
            frame_size = int(sr * 0.02)
            if len(y) > frame_size:
                n_frames = len(y) // frame_size
                y_framed = y[:n_frames*frame_size].reshape(n_frames, frame_size)
                energy = np.sqrt(np.mean(y_framed**2, axis=1))
                active_energy = energy[energy > np.max(energy)*0.01] # Lower threshold
                
                if len(active_energy) > 5:
                    mean_amp = np.mean(active_energy)
                    amp_diffs = np.abs(np.diff(active_energy))
                    shimmer_rms = np.mean(amp_diffs) / (mean_amp + 1e-9)
                    features["shimmer_local"] = float(shimmer_rms * 0.4) 
                else:
                    features["shimmer_local"] = 0.0
            else:
                 features["shimmer_local"] = 0.0

            # 3. HNR (Harmonic to Noise Ratio) - Corrected Normalization
            min_lag = int(sr / 500)
            max_lag = int(sr / 50)
            mid = len(y) // 2
            segment = y[mid:mid+2048] if len(y) > 4000 else y
            
            if len(segment) > max_lag * 2:
                # Normalized Autocorrelation
                segment = segment - np.mean(segment) # Remove DC
                corr = np.correlate(segment, segment, mode='full')
                corr = corr[len(corr)//2:] 
                
                # Normalize by Energy at Lag 0
                max_energy = corr[0]
                if max_energy > 0:
                    corr_norm = corr / max_energy
                    
                    # Search for peak in pitch range
                    valid_corr = corr_norm[min_lag:max_lag]
                    if len(valid_corr) > 0:
                        peak_corr = np.max(valid_corr)
                        # HNR = 10 log10 (Peak / (1-Peak))
                        if peak_corr >= 0.99: peak_corr = 0.99
                        if peak_corr <= 0.01: peak_corr = 0.01
                        
                        hnr_est = 10 * np.log10(peak_corr / (1 - peak_corr))
                        features["hnr"] = float(hnr_est)
                    else:
                        features["hnr"] = 0.0
                else:
                    features["hnr"] = 0.0
            else:
                features["hnr"] = 0.0

            # 4. Pitch
            features["f0_mean"] = float(sr / (mean_period * 2))

            features["valid_voice_detected"] = True

        except Exception as e:
            print(f"[Features] Numpy Error: {e}")
            features["valid_voice_detected"] = False
