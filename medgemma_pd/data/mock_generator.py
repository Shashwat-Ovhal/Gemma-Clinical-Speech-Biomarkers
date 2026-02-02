import numpy as np
import pandas as pd
import wave
import struct
import os
import math

class MockDataGenerator:
    """
    Generates synthetic data for the MedGemma-PD demonstration.
    Refactored to use standard 'wave' library (Dependency Free).
    Strictly follows the constraints:
    - Patient P07: Progressing PD (Increasing Jitter, Decreasing HNR)
    - Patient P08: Healthy Control (Stable)
    """

    @staticmethod
    def generate_uci_longitudinal_data(patient_id: str = "P07") -> pd.DataFrame:
        """
        Generates a synthetic UCI Telemonitoring style CSV.
        Columns: subject_id, session_month, motor_updrs, total_updrs, jitter_percent, hnr
        """
        if patient_id == "P07":
            # Progressing PD Case
            data = {
                "subject_id": ["P07"] * 3,
                "session_month": [0, 3, 6],
                "motor_updrs": [15, 18, 22], # Worsening
                "total_updrs": [22, 28, 35], # Worsening
                "jitter_percent": [0.6, 0.9, 1.2], # Worsening (Human < 1.0)
                "hnr": [22.0, 18.0, 15.0] # Worsening (Healthy > 20)
            }
        else:
            # Healthy Control
            data = {
                "subject_id": ["P08"] * 3,
                "session_month": [0, 3, 6],
                "motor_updrs": [2, 2, 3],
                "total_updrs": [5, 4, 6],
                "jitter_percent": [0.3, 0.35, 0.32], # Stable
                "hnr": [25.0, 26.0, 24.5] # Stable
            }
        
        return pd.DataFrame(data)

    @staticmethod
    def generate_synthetic_audio(filename: str, duration: float = 3.0, sr: int = 44100, jitter_level: float = 0.0):
        """
        Generates a synthetic 'Ahhh' vowel sound with optional jitter using standard 'wave' lib.
        Used to mock PC-GITA raw audio.
        """
        # Ensure dir exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        frame_count = int(sr * duration)
        f0 = 140.0 # Hz
        
        # Open wave file
        with wave.open(filename, 'w') as obj:
            obj.setnchannels(1) # Mono
            obj.setsampwidth(2) # 16-bit
            obj.setframerate(sr)
            
            # Generate samples
            phase = 0.0
            data = []
            
            for i in range(frame_count):
                # Apply Simple Jitter (Perturb Frequency)
                current_f0 = f0 + np.random.normal(0, jitter_level * 1000) 
                
                # Sinewave
                phase += 2 * math.pi * current_f0 / sr
                sample = 0.5 * math.sin(phase)
                
                # Add Noise for HNR/Shimmer simulation
                noise = np.random.normal(0, 0.02)
                sample += noise
                
                # Clip
                sample = max(-1.0, min(1.0, sample))
                
                # Convert to 16-bit PCM
                sample_int = int(sample * 32767)
                data.append(struct.pack('<h', sample_int))
                
            obj.writeframes(b''.join(data))
            
        print(f"[MockDataGenerator] Generated {filename}")
        return filename
