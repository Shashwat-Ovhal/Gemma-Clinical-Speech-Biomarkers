def log(msg):
    with open("debug_progress.txt", "a") as f:
        f.write(msg + "\n")

log("STARTING SCRIPT")
import sys
import os
sys.path.append(os.getcwd()) # FIX: Force current directory into path

try:
    log("Importing glob...")
    import glob
    log("Importing csv...")
    import csv
    log("Importing numpy...")
    import numpy as np
    import numpy as np
    log("Importing tqdm... SKIPPED")
    # from tqdm import tqdm

    
    log("Importing medgemma...")
    from medgemma_pd.audio_pipeline.validation import InputValidator
    log("Imported InputValidator")
    from medgemma_pd.audio_pipeline.preprocessing import AudioPreprocessor
    log("Imported AudioPreprocessor")
    from medgemma_pd.audio_pipeline.quality_control import QualityControl
    log("Imported QualityControl")
    from medgemma_pd.audio_pipeline.features import FeatureExtractor
    log("Imported FeatureExtractor")
    from medgemma_pd.reasoning.history_loader import HistoryLoader
    log("Imported HistoryLoader")
    from medgemma_pd.reasoning.engine import MedGemmaEngine
    log("Imported Engine")
    
except Exception as e:
    import traceback
    log(f"IMPORT CRASH: {e}")
    log(traceback.format_exc())
    sys.exit(1)

# Configuration
DATASET_ROOT = r"dataset- MDVR-KCL Dataset\26_29_09_2017_KCL\26-29_09_2017_KCL\ReadText"
OUTPUT_FILE = "results.csv"

def run_pipeline(file_path, patient_id):
    try:
        # 1. Validation
        val = InputValidator.validate(file_path)
        if not val['valid']:
            return {"error": "Invalid Header"}

        # 2. Preprocessing
        y_norm, sr, audit = AudioPreprocessor.process(file_path)
        
        # 3. QC
        qc = QualityControl.check(y_norm, sr)

        # 4. Features
        features = FeatureExtractor.extract_features(y_norm, sr)

        # 5. History
        try:
            mapped_subj = HistoryLoader.ID_MAPPING.get(patient_id, None)
            if not mapped_subj:
                 digits = ''.join(filter(str.isdigit, patient_id))
                 if digits:
                     mapped_subj = int(digits) % 42 + 1
                 else:
                     mapped_subj = 1
            
            history = HistoryLoader.get_patient_history(str(mapped_subj))
        except:
            history = {"found": False}
        
        # 6. Metadata
        metrics = {
            "filename": os.path.basename(file_path),
            "group": "PD" if "PD" in file_path else "HC",
            "patient_id": patient_id,
            "mapped_subject_id": mapped_subj,
            "jitter": features.get("jitter_local", 0.0) * 100, 
            "shimmer": features.get("shimmer_local", 0.0) * 100,
            "hnr": features.get("hnr", 0.0),
            "pitch": features.get("f0_mean", 0.0),
            "updrs_total": history.get("latest", {}).get("total_updrs", "N/A"),
            "trim_status": audit.get("trim_status", "ok"),
            "fallback_used": features.get("metadata", {}).get("fallback_used", False)
        }
        return metrics
    except Exception as e:
         log(f"Pipeline Error on {file_path}: {e}")
         return {"error": str(e)}

def main():
    log("Creating Log...")
    try:
        hc_files = glob.glob(os.path.join(DATASET_ROOT, "HC", "*.wav"))
        pd_files = glob.glob(os.path.join(DATASET_ROOT, "PD", "*.wav"))
        all_files = hc_files + pd_files
        log(f"Found {len(all_files)} files.")
        
        results = []
        
        i = 0
        for f in all_files:
            i += 1
            if i % 5 == 0: log(f"Processing {i}/{len(all_files)}...")

            fname = os.path.basename(f)
            pid = fname.split('_')[0] 
            
            # log(f"Processing {fname}...") # Reduced logging
            res = run_pipeline(f, pid)
            results.append(res)
            
        log(f"Saving {len(results)} results...")
        if results:
            keys = results[0].keys()
            with open(OUTPUT_FILE, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
        log("BATCH COMPLETE")
        
    except Exception as e:
        log(f"RUNTIME CRASH: {e}")
        import traceback
        log(traceback.format_exc())

if __name__ == "__main__":
    main()
