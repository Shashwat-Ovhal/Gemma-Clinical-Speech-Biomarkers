import sys
import json
from medgemma_pd.reasoning.history_loader import HistoryLoader
from medgemma_pd.reasoning.engine import MedGemmaEngine

def verify_mapping(audio_id, type_d):
    print(f"\n--- Testing Mapping Logic for {audio_id} ({type_d}) ---")
    
    # 1. Mock Audio Packet (Simulate successful audio stage)
    packet = {
        "meta": {"patient_id": audio_id, "filename": f"{audio_id}_test.wav"},
        "clinical_biomarkers": {
            "voice_features": {
                "jitter_local": 0.005 if type_d == 'HC' else 0.02, # 0.5% vs 2.0%
                "shimmer_local": 0.04,
                "hnr": 25.0 if type_d == 'HC' else 15.0
            }
        },
        "model_signals": {"risk_probability": 0.1 if type_d == 'HC' else 0.8}
    }
    
    # 2. Run History Loader (The Logic we are verifying)
    history = HistoryLoader.get_patient_history(audio_id)
    packet["longitudinal_context"] = history
    
    # 3. Generate Insight
    insight = MedGemmaEngine.generate_insight(packet)
    
    # 4. Validate
    found = history['found']
    mapped_subj = history.get('subject_id', 'None')
    
    print(f"  > History Found: {found}")
    print(f"  > Mapped To Subject: {mapped_subj}")
    print(f"  > Assessment: {insight.splitlines()[3]}") # Extract 'Assessment: ...' line
    
    return mapped_subj

# Test Cases
subj_hc = verify_mapping("ID00", "HC") # Expect Subj 18
subj_pd = verify_mapping("ID02", "PD") # Expect Subj 35

with open("mapping_verification.txt", "w") as f:
    f.write(f"ID00 -> {subj_hc}\n")
    f.write(f"ID02 -> {subj_pd}\n")
    if str(subj_hc) == "18" and str(subj_pd) == "35":
        f.write("PASS")
    else:
        f.write("FAIL")
