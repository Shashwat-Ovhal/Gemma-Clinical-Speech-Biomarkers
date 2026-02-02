import os
import soundfile as sf

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
        try:
            with sf.SoundFile(file_path) as f:
                sr = f.samplerate
                ch = f.channels
                dur = len(f) / sr
                
                return {
                    'valid': True, 
                    'error': None,
                    'metadata': {
                        'sample_rate': sr,
                        'channels': ch,
                        'duration_sec': dur
                    }
                }
        except Exception as e:
            return {'valid': False, 'error': f"Corrupt Audio Header: {e}"}
