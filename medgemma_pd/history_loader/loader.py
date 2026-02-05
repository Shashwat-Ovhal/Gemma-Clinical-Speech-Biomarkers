import pandas as pd
import os

class HistoryLoader:
    """
    Agent 2: The Historian (Real Implementation)
    Loads longitudinal data from the UCI Parkinson's Telemonitoring Dataset.
    Source: dataset- Parkinsons Telemonitoring/parkinsons_updrs.data
    """
    
    _DATASET_PATH = r"dataset- Parkinsons Telemonitoring/parkinsons_updrs.data"
    _CACHE = None
    
    # --- Mapping Layer (The Frankenstein Fix) ---
    # Maps MDVR Audio IDs to Clinically Equivalent UCI History Subjects
    # MDVR (Audio) -> UCI (History)
    ID_MAPPING = {
        # PD Patients (High/Moderate Severity)
        "ID02": 35, # Severe (UPDRS ~55)
        "ID04": 25, # Severe (UPDRS ~54)
        "ID06": 21, # Moderate-Severe (UPDRS ~50)
        "ID10": 36, # Moderate
        
        # Healthy Controls (Low Severity / Healthy)
        "ID00": 18, # Healthy (UPDRS ~8)
        "ID01": 22, # Mild/Healthy (UPDRS ~15)
        "ID03": 20, # Mild (UPDRS ~18)
    }

    @classmethod
    def load_data(cls, base_path: str = "."):
        """Loads the CSV into memory (Singleton Cache)."""
        if cls._CACHE is not None:
            return cls._CACHE
            
        full_path = os.path.join(base_path, cls._DATASET_PATH)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"UCI Telemonitoring Data not found at: {full_path}")
            
        # The file is CSV format
        try:
            cls._CACHE = pd.read_csv(full_path)
            return cls._CACHE
        except Exception as e:
            raise RuntimeError(f"Failed to parse UCI data: {e}")

    @staticmethod
    def get_patient_history(patient_id: str, base_path: str = ".") -> dict:
        """
        Retrieves history for a given patient ID.
        
        Args:
            patient_id (str): Format "P<number>" (e.g., "P07") or raw ID "1".
                              We will attempt to parse the integer ID from it.
        
        Returns:
            dict: Summary of UPDRS trends and raw history points.
        """
        try:
            df = HistoryLoader.load_data(base_path)
            
            # --- Parsing Logic ---
            # 1. Check if input is a mapped MDVR ID (e.g. "ID02")
            clean_id = str(patient_id).strip()
            
            # Extract "IDxx" pattern from potential filename path
            import re
            id_match = re.search(r"(ID\d+)", clean_id, re.IGNORECASE)
            if id_match:
                key = id_match.group(1).upper() # Normalize to ID02
                if key in HistoryLoader.ID_MAPPING:
                    subject_num = HistoryLoader.ID_MAPPING[key]
                    print(f"   [Mapping Layer] Mapped Audio '{key}' -> History Subject #{subject_num}")
                else:
                    # Fallback for unmapped IDs: extract number
                    digits = re.findall(r'\d+', key)
                    subject_num = int(digits[0]) if digits else 1
            else:
                # 2. Heuristic: extract raw digits (User entered "P07" or "7")
                digits = re.findall(r'\d+', clean_id)
                if not digits:
                    return {"found": False, "error": "Invalid ID format"}
                subject_num = int(digits[0])
            
            # Filter
            mask = df['subject#'] == subject_num
            patient_data = df[mask]
            
            if patient_data.empty:
                return {
                    "found": False, 
                    "error": f"Subject {subject_num} not found in UCI Database",
                    "available_subjects": df['subject#'].unique().tolist()
                }
                
            # Compute Trend
            # Sort by test_time (ascending)
            patient_data = patient_data.sort_values('test_time')
            
            processed_data = {
                "found": True,
                "subject_id": subject_num,
                "record_count": len(patient_data),
                "baseline": {
                    "motor_updrs": float(patient_data.iloc[0]['motor_UPDRS']),
                    "total_updrs": float(patient_data.iloc[0]['total_UPDRS'])
                },
                "latest": {
                    "motor_updrs": float(patient_data.iloc[-1]['motor_UPDRS']),
                    "total_updrs": float(patient_data.iloc[-1]['total_UPDRS']),
                    "time_day": float(patient_data.iloc[-1]['test_time'])
                },
                "trend_analysis": {} 
            }
            
            # Simple Trend Logic
            delta = processed_data['latest']['total_updrs'] - processed_data['baseline']['total_updrs']
            if delta > 3.0:
                processed_data['trend_analysis']['updrs_trend'] = "deteriorating"
            elif delta < -3.0:
                processed_data['trend_analysis']['updrs_trend'] = "improving"
            else:
                processed_data['trend_analysis']['updrs_trend'] = "stable"
                
            processed_data['trend_analysis']['delta_updrs'] = round(delta, 2)
            
            return processed_data

        except Exception as e:
            return {"found": False, "error": str(e)}
