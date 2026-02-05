print("--- Batch V2 Starting ---", flush=True)
import os
import glob
import csv
import numpy as np
import time

try:
    print("Importing pipeline...", flush=True)
    from medgemma_pd.audio_pipeline.validation import InputValidator
    from medgemma_pd.audio_pipeline.preprocessing import AudioPreprocessor
    from medgemma_pd.audio_pipeline.quality_control import QualityControl
    from medgemma_pd.audio_pipeline.features import FeatureExtractor
    from medgemma_pd.reasoning.history_loader import HistoryLoader
    from medgemma_pd.reasoning.engine import MedGemmaEngine
    print("Imports Success", flush=True)
except Exception as e:
    print(f"Import Error: {e}", flush=True)
    exit(1)

# Configuration
DATASET_ROOT = r"dataset- MDVR-KCL Dataset\26_29_09_2017_KCL\26-29_09_2017_KCL\ReadText"
OUTPUT_FILE = "results.csv"

def main():
    print("Running Main...", flush=True)
    try:
        hc_files = glob.glob(os.path.join(DATASET_ROOT, "HC", "*.wav"))
        pd_files = glob.glob(os.path.join(DATASET_ROOT, "PD", "*.wav"))
        all_files = hc_files + pd_files
        print(f"Found {len(all_files)} files.", flush=True)
        
        # Simple loop for debugging first
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Filename", "Status"])
            for wav in all_files[:3]: # Test first 3
                print(f"Processing {os.path.basename(wav)}", flush=True)
                writer.writerow([os.path.basename(wav), "Processed"])
                
        print("Batch Complete", flush=True)
        
    except Exception as e:
        print(f"Main Error: {e}", flush=True)

if __name__ == "__main__":
    main()
