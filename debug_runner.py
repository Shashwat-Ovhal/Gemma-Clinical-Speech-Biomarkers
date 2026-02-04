import sys
with open("debug_trace.txt", "w") as f: f.write("STARTING\n")

def log(msg):
    with open("debug_trace.txt", "a") as f: f.write(msg + "\n")

try:
    log("1. Importing DataLoader...")
    from medgemma_pd.data.loader import DataLoader
    log("   OK")
except Exception as e: log(f"   FAIL: {e}")

try:
    log("2. Importing HistoryLoader...")
    from medgemma_pd.reasoning.history_loader import HistoryLoader
    log("   OK")
except Exception as e: log(f"   FAIL: {e}")

try:
    log("3. Importing Preprocessing...")
    from medgemma_pd.audio_pipeline.preprocessing import AudioPreprocessor
    log("   OK")
except Exception as e: log(f"   FAIL: {e}")

try:
    log("4. Importing Features...")
    from medgemma_pd.audio_pipeline.features import FeatureExtractor
    log("   OK")
except Exception as e: log(f"   FAIL: {e}")

try:
    log("5. Importing Pipeline...")
    from medgemma_pd.audio_pipeline.pipeline import MedicalAudioPipeline
    log("   OK")
except Exception as e: log(f"   FAIL: {e}")

try:
    log("6. Importing Engine...")
    from medgemma_pd.reasoning.engine import MedGemmaEngine
    log("   OK")
except Exception as e: log(f"   FAIL: {e}")

log("Debug Runner Finished.")
