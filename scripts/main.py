import argparse
import json
import os
import sys

# Ensure we can import modules from current dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from medgemma_pd.data.loader import DataLoader
# from medgemma_pd.reasoning.packet_builder import PacketBuilder
# from medgemma_pd.reasoning.engine import MedGemmaEngine

def main():
    with open("status.txt", "w") as f: f.write("STARTING\n")
    parser = argparse.ArgumentParser(description="MedGemma-PD Pipeline (Real Data Mode)")
    parser.add_argument("--file", type=str, required=False, 
                       default=r"dataset- MDVR-KCL Dataset/26_29_09_2017_KCL/26-29_09_2017_KCL/ReadText/PD/ID02_pd_2_0_0.wav",
                       help="Path to input audio WAV file")
    parser.add_argument("--patient_id", type=str, default="P07", help="Patient ID for searching history (if not parsed from filename)")
    
    args = parser.parse_args()
    
    print(f"\n--- MedGemma-PD Pipeline ---")
    
    if not os.path.exists(args.file):
        print(f"ERROR: File not found: {args.file}")
        # Try a fallback search or finding any wav
        import glob
        wavs = glob.glob("**/*.wav", recursive=True)
        if wavs:
            print(f"Suggestion: Try '{wavs[0]}'")
        return
    from medgemma_pd.history_loader import HistoryLoader
    from medgemma_pd.audio_pipeline.pipeline import MedicalAudioPipeline
    from medgemma_pd.reasoning.engine import MedGemmaEngine

    # 1. Pipeline Execution (Audio -> Features)
    print(f"\n[1/3] Processing Audio: {args.file}...")
    with open("status.txt", "a") as f: f.write("STEP 1: AUDIO\n")
    
    pipeline_report = MedicalAudioPipeline.process_file(args.file)
    
    with open("status.txt", "a") as f: f.write(f"STEP 1 DONE: {pipeline_report['status']}\n")
    
    if pipeline_report['status'] != 'success':
        print(f"CRITICAL: Audio Processing Failed: {pipeline_report.get('error')}")
        with open("status.txt", "a") as f: f.write(f"FAIL: {pipeline_report.get('error')}\n")
        sys.exit(1)
        
    features = pipeline_report['stages']['feature_extraction']
    print(f"   > Jitter: {features.get('jitter_local', 0)*100:.4f}%")
    print(f"   > Shimmer: {features.get('shimmer_local', 0)*100:.4f}%")
    print(f"   > HNR: {features.get('hnr', 0):.2f} dB")
    print(f"   > Latency: {features.get('latency_ms', 0):.2f} ms")

    # 2. History Retrieval (The Historian)
    print(f"\n[2/3] Retrieving History for Patient {args.patient_id}...")
    with open("status.txt", "a") as f: f.write("STEP 2: HISTORY\n")
    
    # Map the CLI patient ID (default P07 or derived from filename) to the loader
    # If using regex on filename:
    import re
    pid_match = re.search(r"ID(\d+)", os.path.basename(args.file))
    derived_id = pid_match.group(1) if pid_match else args.patient_id
    
    # UCI dataset IDs are integers (1-42), but user might pass "ID02" or "P07"
    # HistoryLoader handles string conversion internally
    history_report = HistoryLoader.get_patient_history(derived_id)
    
    with open("status.txt", "a") as f: f.write(f"STEP 2 DONE: {history_report['found']}\n")
    
    if not history_report['found']:
        print(f"   > Warning: {history_report.get('error')}. Using Placeholder Context.")
        history_context = {"trend_analysis": {"updrs_trend": "Unknown", "delta_updrs": 0}}
    else:
        print(f"   > Found {history_report['record_count']} records.")
        print(f"   > Baseline UPDRS: {history_report['baseline']['total_updrs']}")
        print(f"   > Latest UPDRS: {history_report['latest']['total_updrs']}")
        print(f"   > Trend: {history_report['trend_analysis']['updrs_trend'].upper()}")
        history_context = history_report

    # 3. Reasoning (The Reasoner)
    print("\n[3/3] Generating Clinical Insight...")
    
    # Construct Packet Manually for now (replacing PacketBuilder's rigid logic)
    packet = {
        "meta": {"patient_id": derived_id, "filename": args.file},
        "clinical_biomarkers": {"voice_features": features},
        "longitudinal_context": history_context,
        "model_signals": {"risk_probability": 0.75 if features.get('jitter_local',0) > 0.0104 else 0.25} # Simple heuristic
    }
    
    insight = MedGemmaEngine.generate_insight(packet)
    
    print("\n" + "="*60)
    print(insight)
    print("="*60 + "\n")
    
    # Generate Web Data
    output_path = os.path.join(os.path.dirname(__file__), "medgemma_pd/ui/data.js")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    web_data = {
        "packet": packet,
        "insight": insight,
        "pipeline_report": pipeline_report
    }
    
    # JSON serialization safety for numpy types
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super(NpEncoder, self).default(obj)
            
    import numpy as np # Import locally if needed
    
    with open(output_path, "w") as f:
        f.write(f"window.medgemmaData = {json.dumps(web_data, cls=NpEncoder, indent=2)};")
    print(f"[UI] Generated dashboard data: {output_path}")
    
    # Validation Output for Agent
    with open("last_run_insight.txt", "w") as f:
        f.write(f"Patient ID: {packet['meta']['patient_id']}\n")
        f.write(f"Mapped History Subject: {packet['longitudinal_context'].get('subject_id', 'Unknown')}\n")
        f.write(f"Jitter: {features.get('jitter_local', 0)*100:.4f}%\n")
        f.write(f"Assessment: {insight}\n")
        
    with open("status.txt", "a") as f: f.write("COMPLETED\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open("status.txt", "a") as f: f.write(f"CRASH: {e}\n")
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR: {e}")
