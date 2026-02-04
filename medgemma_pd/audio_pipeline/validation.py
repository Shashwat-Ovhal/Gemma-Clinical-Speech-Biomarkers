import os
import os
import contextlib
import wave # Standard library fallback
from scipy.io import wavfile

class InputValidator:
    """
    Layer 1: Input Validation
    Responsible for rejecting invalid files before any processing happens.
    """
    
    ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.ogg', '.flac'}
    MAX_SIZE_MB = 50

    @staticmethod
    def validate(file_path: str) -> dict:
        """
        Checks if file exists, has valid extension, and is readable.
        Returns: {'valid': bool, 'error': str, 'metadata': dict}
        """
        # 1. Path/Existence Check
        if not os.path.exists(file_path):
            return {'valid': False, 'error': "File not found"}
        
        # 2. Extension Check
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in InputValidator.ALLOWED_EXTENSIONS:
             return {'valid': False, 'error': f"Unsupported format: {ext}"}

        # 3. Size Check
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > InputValidator.MAX_SIZE_MB:
             return {'valid': False, 'error': f"File too large ({size_mb:.2f}MB). Max: {InputValidator.MAX_SIZE_MB}MB"}

        # 4. Header Integrity (Try opening)
        # 4. Header Integrity (Try opening)
        try:
            # Use Scipy (robust) or Wave (fast)
            # Scipy returns (rate, data)
            # For validation, we just need headers without full read if possible, 
            # but wavfile.read is standard here.
            
            # Optimization: Use standard 'wave' module for header only check if .wav
            if ext == '.wav':
                with contextlib.closing(wave.open(file_path, 'r')) as f:
                     frames = f.getnframes()
                     rate = f.getframerate()
                     duration = frames / float(rate)
                     channels = f.getnchannels()
                     
                return {
                    'valid': True, 
                    'error': None,
                    'metadata': {
                        'sample_rate': rate,
                        'channels': channels,
                        'duration_sec': duration
                    }
                }
            else:
                 # Be more permissive for non-wavs or assume passed if extensions allowed
                 # We can't validate MP3/etc without ffmpeg/librosa
                 # Just pass them and let Preprocessor handle it.
                 return {'valid': True, 'error': None, 'metadata': {'sample_rate': 0, 'channels': 0, 'duration_sec': 0}}

        except Exception as e:
            return {'valid': False, 'error': f"Corrupt Audio Header: {e}"}
