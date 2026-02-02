import pandas as pd
import numpy as np
from ..data.loader import DataLoader
from ..audio_pipeline.features import FeatureExtractor
from ..models.signals import MLSignalGenerator, TrendAnalyzer
from ..audio_pipeline.preprocessing import AudioPreprocessor

class PacketBuilder:
    """
    Constructs the 'Structured Clinical Evidence Packet'.
    This is the bridge between the 'Signal Generators' and 'MedGemma'.
    """

    def __init__(self, data_loader: DataLoader):
        self.loader = data_loader

    def build_packet(self, patient_id: str, current_session_month: int) -> dict:
        """
        Orchestrates the data collection for a specific patient session.
        """
        # 1. Load History (UCI Longitudinal)
        try:
            history = self.loader.get_longitudinal_records(patient_id)
            past_sessions = history[history['session_month'] < current_session_month]
        except Exception as e:
            print(f"Warning: No history found for {patient_id}. {e}")
            history = pd.DataFrame()
            past_sessions = pd.DataFrame()

        # 2. Load Current Audio (PC-GITA)
        audio_path = self.loader.get_audio_session(patient_id, current_session_month)
        
        # 3. Extract Features (Biomarkers)
        # We need to preprocess first
        try:
            y, sr, _ = AudioPreprocessor.process(audio_path)
            features = FeatureExtractor.extract_features(y, sr)
        except Exception as e:
            print(f"Error extracting features: {e}. Using empty.")
            features = {}

        # 4. Generate Weak Signals (ML)
        risk_analysis = MLSignalGenerator.predict_risk_score(features)
        trend_analysis = TrendAnalyzer.analyze_progression(history)

        # 5. Construct Packet
        packet = {
            "meta": {
                "patient_id": patient_id,
                "session_month": current_session_month,
                "data_source": "PC-GITA + UCI Telemonitoring"
            },
            "clinical_biomarkers": {
                "voice_features": features,
                "reference_ranges": { # Context for the LLM
                    "jitter_normal": "< 1.04%",
                    "hnr_normal": "> 20 dB"
                }
            },
            "longitudinal_context": {
                "history_summary": past_sessions.to_dict(orient='records'),
                "trend_analysis": trend_analysis
            },
            "model_signals": {
                "risk_probability": risk_analysis['risk_score'],
                "signal_interpretation": risk_analysis['interpretation']
            },
            "confidence_assessment": {
                "data_quality": "High" if features.get("valid_voice_detected") else "Low",
                "missing_modalities": ["motor_assessment", "cognitive_score"]
            }
        }

        return packet
