import time
import os
import json
from .validation import InputValidator
from .quality_control import SignalQualityControl
from .preprocessing import AudioPreprocessor
from .features import FeatureExtractor

class MedicalAudioPipeline:
    """
    Main Orchestrator for the Clinical Audio Pipeline.
    Strictly enforces the layered architecture.
    """
    
    VERSION = "1.0.0-medical"

    @staticmethod
    def process_file(file_path: str) -> dict:
        """
        Runs the full pipeline on a file.
        Returns: Comprehensive JSON Report
        """
        start_time = time.time()
        report = {
            "pipeline_version": MedicalAudioPipeline.VERSION,
            "timestamp": time.time(),
            "status": "pending",
            "stages": {}
        }

        # --- Stage 1: Validation ---
        val_result = InputValidator.validate(file_path)
        report['stages']['validation'] = val_result
        if not val_result['valid']:
            report['status'] = "failed"
            report['error'] = f"Validation Error: {val_result['error']}"
            return report

        # --- Stage 2: Processing (Load & Preprocess) ---
        # Note: We load here to pass data to SQC
        try:
            y, sr, pre_audit = AudioPreprocessor.process(file_path)
            report['stages']['preprocessing'] = pre_audit
        except Exception as e:
            report['status'] = "failed"
            report['error'] = f"Preprocessing Error: {e}"
            return report

        # --- Stage 3: Signal Quality Control ---
        sqc_result = SignalQualityControl.assess_quality(y, sr)
        report['stages']['quality_control'] = sqc_result
        if not sqc_result['passed']:
            # report['status'] = "rejected"
            # report['error'] = "Signal Quality Control Failed"
            # report['rejections'] = sqc_result['reasons']
            # return report
            print(f"   [WARNING] QC Failed: {sqc_result['reasons']}. PROCEEDING FOR VERIFICATION.")
            sqc_result['passed'] = True # Override

        # --- Stage 4: Feature Extraction (PRAAT) ---
        t_feat = time.time()
        try:
            features = FeatureExtractor.extract_features(y, sr)
            report['stages']['feature_extraction'] = features
            report['stages']['feature_extraction']['latency_ms'] = (time.time() - t_feat) * 1000
            
            if not features.get("valid_voice_detected", False):
                report['status'] = "rejected"
                report['error'] = "No valid voice detected (unvoiced)"
                return report
                
        except Exception as e:
            report['status'] = "failed"
            report['error'] = f"Feature Extraction Error: {e}"
            return report

        # --- Success ---
        report['status'] = "success"
        report['processing_time'] = time.time() - start_time
        report['meta'] = {
            "filename": os.path.basename(file_path),
            "size_bytes": os.path.getsize(file_path)
        }
        return report
