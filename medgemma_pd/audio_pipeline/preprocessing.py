import numpy as np
import warnings

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("Librosa not found. Audio preprocessing will be MOCKED.")

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
        
        if not LIBROSA_AVAILABLE:
            # Fallback for environments without librosa
            # Return silence - FeatureExtractor will detect this or use its own Mock Mode
            return np.zeros(16000), 16000, {"status": "mocked", "reason": "missing_librosa"}

        # 1. Load & Resample
        try:
            # Librosa loads as mono by default (mixes channels)
            y, sr = librosa.load(file_path, sr=AudioPreprocessor.TARGET_SR, mono=True)
            audit['resample_rate'] = AudioPreprocessor.TARGET_SR
            
            # Check length to avoid zero-division
            if len(y) == 0:
                  return np.zeros(16000), 16000, {"status": "empty_file"}

            audit['original_duration'] = len(y)/sr
            
            # 2. Trim Silence (Optional - conservative)
            # We perform a light trim (topdb=60) to remove dead air at ends
            y_trimmed, _ = librosa.effects.trim(y, top_db=60)
            audit['trimmed_duration'] = len(y_trimmed)/sr
            
            # 3. Peak Normalization
            # Normalize to -3dB to prevent clipping but maximize dynamic range
            current_max = np.max(np.abs(y_trimmed))
            if current_max > 0:
                target_amp = 10 ** (AudioPreprocessor.TARGET_DB / 20)
                y_norm = y_trimmed * (target_amp / current_max)
                audit['normalization_gain'] = target_amp / current_max
            else:
                y_norm = y_trimmed
                audit['normalization_gain'] = 1.0
        except Exception as e:
             # Catch file read errors (e.g. if MockGenerator made a bad WAV)
             print(f"[AudioPreprocessor] Error reading file: {e}")
             return np.zeros(16000), 16000, {"status": "error", "reason": str(e)}

        return y_norm, sr, audit
