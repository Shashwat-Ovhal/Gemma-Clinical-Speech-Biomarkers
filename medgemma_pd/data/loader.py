import os
import pandas as pd
from typing import Dict, Any, List
from .mock_generator import MockDataGenerator

class DataLoader:
    """
    Abstraction layer for loading PC-GITA (Audio) and UCI (Longitudinal) data.
    Automatically handles 'Mock' mode for the demo cases (P07, P08).
    """

    def __init__(self, data_root: str):
        self.data_root = data_root
        self.mock_gen = MockDataGenerator()

    def get_longitudinal_records(self, patient_id: str) -> pd.DataFrame:
        """
        Returns the longitudinal history (UCI style) for a patient.
        If patient_id is P07 or P08, generates synthetic data.
        """
        if patient_id in ["P07", "P08"]:
            print(f"[DataLoader] Generating MOCK longitudinal data for {patient_id}")
            return self.mock_gen.generate_uci_longitudinal_data(patient_id)
        
        # Fallback: Try to load from CSV
        # csv_path = os.path.join(self.data_root, "uci", f"{patient_id}.csv")
        # if os.path.exists(csv_path):
        #     return pd.read_csv(csv_path)
            
        raise FileNotFoundError(f"No records found for {patient_id}")

    def get_audio_session(self, patient_id: str, session_month: int) -> str:
        """
        Returns the path to a .wav file for the specific patient and session.
        Generates synthetic audio if missing/mock.
        """
        file_path = os.path.join(self.data_root, "pc_gita_mock", patient_id, f"session_{session_month}.wav")
        
        if patient_id == "P07":
            # P07 gets worse over time (more jitter)
            jitter_map = {0: 0.005, 3: 0.02, 6: 0.05}
            if not os.path.exists(file_path):
                print(f"[DataLoader] Generating MOCK audio for {patient_id} Session {session_month} (Jitter={jitter_map.get(session_month)})")
                self.mock_gen.generate_synthetic_audio(
                    file_path, 
                    jitter_level=jitter_map.get(session_month, 0.01)
                )
            return file_path
            
        elif patient_id == "P08":
            # P08 is stable (low jitter)
            if not os.path.exists(file_path):
                self.mock_gen.generate_synthetic_audio(file_path, jitter_level=0.002)
            return file_path
            
        # Real file check
        # real_path = os.path.join(self.data_root, "pc_gita", patient_id, f"{session_month}.wav")
        # return real_path
        
        return file_path
