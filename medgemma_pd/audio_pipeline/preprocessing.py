import numpy as np
import warnings
import io

# Try pydub (uses ffmpeg for m4a/mp3 decoding)
try:
    from pydub import AudioSegment
    # Point pydub to winget-installed ffmpeg explicitly (survives PATH not refreshed)
    import os as _os
    _FFMPEG_BIN = r"C:\Users\Shashwat\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
    if _os.path.isfile(_os.path.join(_FFMPEG_BIN, "ffmpeg.exe")):
        _os.environ["PATH"] += _os.pathsep + _FFMPEG_BIN
        AudioSegment.converter  = _os.path.join(_FFMPEG_BIN, "ffmpeg.exe")
        AudioSegment.ffmpeg     = _os.path.join(_FFMPEG_BIN, "ffmpeg.exe")
        AudioSegment.ffprobe    = _os.path.join(_FFMPEG_BIN, "ffprobe.exe")
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Try librosa as secondary fallback
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

class AudioPreprocessor:
    """
    Layer 3: Preprocessing
    Standardizes audio for consistent analysis.
    Uses librosa for .m4a/.mp3 (requires ffmpeg) and scipy for .wav.
    """
    
    TARGET_SR = 16000 # Standard for medical ML
    TARGET_DB = -3.0  # Peak normalization target
    
    @staticmethod
    def process(file_path: str) -> tuple:
        """
        Loads, Resamples, Mono-mixes, and Normalizes audio.
        Returns: (y_processed, sr, audit_log)
        """
        audit = {}
        
        try:
            ext = file_path.lower().rsplit(".", 1)[-1]

            # --- Load audio ---
            if PYDUB_AVAILABLE and ext in ("m4a", "mp3", "aac", "flac", "ogg"):
                # pydub decodes via ffmpeg -> convert to numpy float array
                seg = AudioSegment.from_file(file_path)
                seg = seg.set_channels(1).set_frame_rate(AudioPreprocessor.TARGET_SR)
                samples = np.array(seg.get_array_of_samples(), dtype=np.float64)
                # Normalize to [-1, 1]
                max_int = float(2 ** (seg.sample_width * 8 - 1))
                y_resampled = samples / max_int
                audit["loader"] = "pydub+ffmpeg"
            else:
                # Pure scipy path for WAV
                from scipy.io import wavfile
                from scipy import signal
                src_sr, y_raw = wavfile.read(file_path)
                audit["loader"] = "scipy"

                if y_raw.dtype.kind == 'i':
                    type_info = np.iinfo(y_raw.dtype)
                    y_float = y_raw.astype(float) / max(abs(type_info.min), abs(type_info.max))
                elif y_raw.dtype.kind == 'f':
                    y_float = y_raw.astype(np.float64)
                else:
                    y_float = (y_raw.astype(float) - 128.0) / 128.0

                if len(y_float.shape) > 1:
                    y_float = np.mean(y_float, axis=1)

                if src_sr != AudioPreprocessor.TARGET_SR:
                    num_samples = int(len(y_float) * AudioPreprocessor.TARGET_SR / src_sr)
                    y_resampled = signal.resample(y_float, num_samples)
                    audit['resample_rate'] = AudioPreprocessor.TARGET_SR
                else:
                    y_resampled = y_float

            # --- Peak normalization ---
            max_val = np.max(np.abs(y_resampled))
            if max_val > 0:
                y_norm = y_resampled / max_val
            else:
                y_norm = y_resampled

            # --- Smart Trimming ---
            y_trimmed, trim_log = AudioPreprocessor._trim_silence_numpy(y_norm, top_db=60)
            audit.update(trim_log)

            if len(y_trimmed) == 0:
                warnings.warn("Trim removed entire signal. Reverting to original.")
                y_trimmed = y_norm
                audit['trim_status'] = "reverted_to_original"

            # --- Final amplitude normalization to target dB ---
            current_max = np.max(np.abs(y_trimmed))
            if current_max > 0:
                target_amp = 10 ** (AudioPreprocessor.TARGET_DB / 20)
                y_final = y_trimmed * (target_amp / current_max)
                audit['normalization_gain'] = target_amp / current_max
            else:
                y_final = y_trimmed
                audit['normalization_gain'] = 1.0
                
            return y_final, AudioPreprocessor.TARGET_SR, audit

        except Exception as e:
            print(f"[AudioPreprocessor] Error reading file: {e}")
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
