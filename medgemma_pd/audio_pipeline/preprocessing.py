import numpy as np
import warnings

# Librosa/Soundfile removed due to instability
# Using Pure Scipy/Numpy implementation
LIBROSA_AVAILABLE = False

class AudioPreprocessor:
    """
    Layer 3: Preprocessing
    Standardizes audio for consistent analysis.
    Gracefully falls back to mock processing if dependencies are missing.
    """
    
    TARGET_SR = 16000 # Standard for medical ML
    TARGET_DB = -3.0  # Peak normalization target
    
    @staticmethod
    def process(file_path: str) -> tuple[np.ndarray, int, dict]:
        """
        Loads, Resamples, Mono-mixes, and Normalizes audio.
        Returns: (y_processed, sr, audit_log)
        """
        audit = {}
        
        # if not LIBROSA_AVAILABLE:
        #     # Fallback for environments without librosa
        #     # Return silence - FeatureExtractor will detect this or use its own Mock Mode
        #     return np.zeros(16000), 16000, {"status": "mocked", "reason": "missing_librosa"}

        # 1. Load & Resample (Using Scipy to avoid Librosa crashes)
        try:
            from scipy.io import wavfile
            from scipy import signal
            
            src_sr, y_raw = wavfile.read(file_path)
            
            # Normalize to float (Universal Handling)
            if y_raw.dtype.kind == 'i':
                # Integer type (int16, int32)
                # Check for 24-bit stored as 32-bit? Scipy usually scales to range.
                # Safer: Normalize by the type's max value 
                type_info = np.iinfo(y_raw.dtype)
                y_float = y_raw.astype(float) / max(abs(type_info.min), abs(type_info.max))
            elif y_raw.dtype.kind == 'f':
                # Float type - assumed normalized or requiring peak norm later
                y_float = y_raw
            else:
                 # Unsure (uint8?) - map 0..255 to -1..1
                 y_float = (y_raw.astype(float) - 128.0) / 128.0
                
                
            # Convert to Mono
            if len(y_float.shape) > 1:
                y_mono = np.mean(y_float, axis=1)
            else:
                y_mono = y_float
                
            # --- Smart Trimming (User Requested Check) ---
            # Remove leading/trailing silence to fix "Signal Too Low" false positives
            # --- Peak Normalization (Fix 1: Normalize First) ---
            # Maximize volume before trimming to fix quiet files
            max_val = np.max(np.abs(y_mono))
            if max_val > 0:
                y_norm = y_mono / max_val
            else:
                y_norm = y_mono
                
            # --- Smart Trimming (Fix 2: Lower Threshold) ---
            # Changed top_db from 20 to 60 (Keep almost everything)
            y_trimmed, trim_log = AudioPreprocessor._trim_silence_numpy(y_norm, top_db=60)
            audit.update(trim_log)
            
            # --- Fallback Mode (Fix 3: Never Crash) ---
            if len(y_trimmed) == 0:
                 # If trim removed everything, revert to original normalized signal
                 warnings.warn("Trim removed entire signal. Reverting to original.")
                 y_trimmed = y_norm
                 audit['trim_status'] = "reverted_to_original"

            # Resample if needed
            if src_sr != AudioPreprocessor.TARGET_SR:
                num_samples = int(len(y_trimmed) * AudioPreprocessor.TARGET_SR / src_sr)
                y_resampled = signal.resample(y_trimmed, num_samples)
                audit['resample_rate'] = AudioPreprocessor.TARGET_SR
            else:
                y_resampled = y_trimmed
                audit['resample_rate'] = src_sr

            # 3. Peak Normalization (Already done, but ensures target DB match)
            # We target -3dB (0.707). If we just normalized to 1.0, we scale down slightly.
            current_max = np.max(np.abs(y_resampled))
            if current_max > 0:
                target_amp = 10 ** (AudioPreprocessor.TARGET_DB / 20)
                y_final = y_resampled * (target_amp / current_max)
                audit['normalization_gain'] = target_amp / current_max
            else:
                y_final = y_resampled
                audit['normalization_gain'] = 1.0
                
            return y_final, AudioPreprocessor.TARGET_SR, audit

        except Exception as e:
             # Catch file read errors
             print(f"[AudioPreprocessor] Error reading file: {e}")
             # Return fallback
             return np.zeros(16000), 16000, {"status": "error", "reason": str(e)}

    @staticmethod
    def _trim_silence_numpy(y, top_db=20, frame_length=2048, hop_length=512):
        """
        Numpy implementation of librosa.effects.trim
        """
        if len(y) < frame_length:
            return y, {"trim_skipped": "too_short"}
            
        # 1. Calculate Envelope (RMSE)
        # Pad to ensure frames cover edges
        y_padded = np.pad(y, (0, frame_length), mode='constant')
        
        # Strided slice for efficient windowing (Vectorized RMSE)
        # Shape: (n_frames, frame_length)
        num_frames = (len(y) - 0) // hop_length
        # Simplified: loop (safer than tricky stride tricks in quick implementation)
        # Calculate non-overlapping energy first for speed or use simpler amplitude
        
        # Energy per sample
        energy = y ** 2
        # Convolve with window for smoothing (like RMS)
        window = np.ones(frame_length) / frame_length
        # Use simple moving average as proxy for RMS energy envelope
        mse_env = np.convolve(energy, window, mode='same')
        rmse_env = np.sqrt(mse_env)
        
        # 2. Convert to dB
        # Ref is peak
        ref = np.max(rmse_env)
        if ref <= 0:
            return y, {"trim_status": "silent_ref"}
            
        db_env = 20 * np.log10(rmse_env / ref + 1e-9) # 1e-9 to avoid log(0)
        
        # 3. Find mask
        mask = db_env > -top_db
        
        # Find first and last True
        # np.flatnonzero returns indices of non-zero elements
        active_indices = np.flatnonzero(mask)
        
        if len(active_indices) == 0:
            return np.array([]), {"trim_status": "all_silence"}
            
        start = active_indices[0]
        end = active_indices[-1]
        
        # Map back to samples (approximate since we used convolved window same size)
        # Direct index mapping is sufficiently accurate for trimming
        return y[start:end], {"trim_removed_sec": (len(y) - (end-start))/16000}
