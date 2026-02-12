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
        Extracts clinical features using robust Autocorrelation (ACF) method.
        Replaces flawed Zero-Crossing Rate (ZCR) approach.
        """
        try:
            # --- 1. Pitch Detection (Autocorrelation) ---
            # Frame-based analysis to capture Jitter/Shimmer dynamics
            
            # Constants for Speech Analysis
            frame_dur = 0.04 # 40ms analysis window
            hop_dur = 0.01   # 10ms hop
            min_f0 = 75      # Hz
            max_f0 = 600     # Hz
            
            frame_len = int(sr * frame_dur)
            hop_len = int(sr * hop_dur)
            
            num_frames = (len(y) - frame_len) // hop_len
            
            if num_frames < 3:
                # Signal too short for analysis
                features["valid_voice_detected"] = False
                return

            f0s = []
            peaks = []
            
            # Simple Hanning Window
            window = np.hanning(frame_len)
            
            for i in range(num_frames):
                start = i * hop_len
                frame = y[start : start + frame_len] * window
                
                # ACF
                # Pad to avoid circular convolution separation issues
                n = len(frame)
                pad_frame = np.pad(frame, (0, n), mode='constant')
                f = np.fft.fft(pad_frame)
                acf = np.fft.ifft(f * np.conj(f)).real
                acf = acf[:n]
                
                # Find Peak in Pitch Range
                min_lag = int(sr / max_f0)
                max_lag = int(sr / min_f0)
                
                if max_lag >= len(acf): max_lag = len(acf) - 1
                
                segment = acf[min_lag:max_lag]
                if len(segment) == 0: continue
                    
                peak_idx = np.argmax(segment)
                peak_val = segment[peak_idx]
                true_lag = min_lag + peak_idx
                
                # Voicing Detection (HNR-like check)
                energy = acf[0] # Energy at lag 0
                # STRICTER THRESHOLD: 0.45 (was 0.25) to reject noise/breathiness
                if energy > 0.001 and (peak_val / energy) > 0.45: 
                    f0 = sr / true_lag
                    f0s.append(f0)
                    peaks.append(np.max(np.abs(frame)))

            # --- 2. Jitter & Shimmer Calculation ---
            if len(f0s) < 5:
                # Not enough voiced frames
                features["valid_voice_detected"] = False
                return
                
            features["valid_voice_detected"] = True
            
            # --- 2. Stable Segment Selection (The Fix for Continuous Speech) ---
            # Continuous speech has high pitch variance (intonation). 
            # We must find the "most sustained vowel" segment to calculate valid jitter.
            
            # 2a. Median Filter to remove Octave Jumps (Outliers)
            from scipy.signal import medfilt
            f0_arr = medfilt(np.array(f0s), kernel_size=5)
            amp_arr = np.array(peaks)
            
            # Minimum required duration: 0.30 seconds (30 frames)
            # Find the "Cleanest Vowel"
            min_window_frames = 30 
            
            best_start = 0
            best_end = len(f0_arr)
            
            if len(f0_arr) >= min_window_frames:
                # Sliding window to find LOWEST JITTER directly
                min_jitter = float('inf')
                
                # We check windows of 0.5 second (50 frames) or if shorter, length-15
                window_size = min(50, len(f0_arr)) 
                
                for i in range(len(f0_arr) - window_size):
                    # Get Window
                    w_f0 = f0_arr[i : i+window_size]
                    w_amp = amp_arr[i : i+window_size]
                    
                    # 1. Amplitude Gate (Must be significant part of signal)
                    if np.mean(w_amp) < (0.2 * np.max(amp_arr)):
                        continue
                        
                    # 2. Calculate Jitter for this window
                    w_periods = 1.0 / (w_f0 + 1e-9)
                    avg_per = np.mean(w_periods)
                    per_diff = np.mean(np.abs(np.diff(w_periods)))
                    w_jitter = per_diff / avg_per if avg_per > 0 else 1.0
                    
                    # Minimize Jitter
                    if w_jitter < min_jitter:
                        min_jitter = w_jitter
                        best_start = i
                
                best_end = best_start + window_size
                # print(f"Best Window Jitter: {min_jitter*100:.3f}%")

            # Extract metrics ONLY from the stable window
            f0_stable = f0_arr[best_start:best_end]
            amp_stable = amp_arr[best_start:best_end]
            
            # Re-Calculate Periods for the stable segment
            periods = 1.0 / (f0_stable + 1e-9)
            
            # Jitter (Local)
            avg_period = np.mean(periods)
            period_diffs = np.abs(np.diff(periods))
            jitter = np.mean(period_diffs) / avg_period if avg_period > 0 else 0.0
            
            # Shimmer (Local)
            avg_amp = np.mean(amp_stable)
            amp_diffs = np.abs(np.diff(amp_stable))
            shimmer = np.mean(amp_diffs) / avg_amp if avg_amp > 0 else 0.0
            
            features["jitter_local"] = jitter
            features["shimmer_local"] = shimmer
            
            # Report F0 stats for the Whole file vs Stable
            features["f0_mean"] = float(np.mean(f0_stable))
            features["f0_std"] = float(np.std(f0_stable)) # Use stable std
            features["f0_trace_std"] = float(np.std(f0_arr)) # Full file std (for detection)

            # --- 3. HNR (Harmonoic-to-Noise Ratio) ---
            # Global estimate from middle of signal
            mid = len(y) // 2
            seg_len = min(len(y), 4096)
            mid_segment = y[mid - seg_len//2 : mid + seg_len//2]
            
            # ACF of segment
            n = len(mid_segment)
            pad_seg = np.pad(mid_segment, (0, n), mode='constant')
            f_seg = np.fft.fft(pad_seg)
            acf_seg = np.fft.ifft(f_seg * np.conj(f_seg)).real
            acf_seg = acf_seg[:n]
            
            min_lag = int(sr / 600)
            max_lag = int(sr / 75)
            
            if max_lag < len(acf_seg):
                peak_target = acf_seg[min_lag:max_lag]
                if len(peak_target) > 0:
                    peak_val = np.max(peak_target)
                    total_energy = acf_seg[0]
                    
                    if total_energy > peak_val:
                        # HNR = 10 * log10 (Harmonic / Noise)
                        # Where Harmonic ~ Peak, Noise ~ Total - Peak
                        ratio = peak_val / (total_energy - peak_val + 1e-9)
                        hnr = 10 * np.log10(ratio)
                    else:
                        hnr = 100.0 # Clean
                else:
                    hnr = 0.0
            else:
                hnr = 0.0
                
            features["hnr"] = float(hnr)

        except Exception as e:
            print(f"[Features] Numpy Error: {e}")
            features["valid_voice_detected"] = False
