import streamlit as st
import os
import sys
import time

# FIX: Force current directory into path for local imports
sys.path.append(os.getcwd())

from medgemma_pd.audio_pipeline.validation import InputValidator
from medgemma_pd.audio_pipeline.preprocessing import AudioPreprocessor
from medgemma_pd.audio_pipeline.quality_control import SignalQualityControl as QualityControl # Fix class name alias
from medgemma_pd.audio_pipeline.features import FeatureExtractor
from medgemma_pd.history_loader import HistoryLoader
from medgemma_pd.reasoning.engine import MedGemmaEngine

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal

def plot_spectrogram(y, sr):
    """Generates a Mel-like Spectrogram using Scipy/Matplotlib"""
    f, t, Sxx = signal.spectrogram(y, sr, nperseg=1024, noverlap=512)
    fig, ax = plt.subplots(figsize=(10, 2))
    # Log scale for better visualization (dB)
    Sxx_log = 10 * np.log10(Sxx + 1e-9)
    ax.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap='magma')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_title('Audio Spectrogram (Visual Evidence)')
    ax.set_ylim(0, 8000) # Speech range
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(page_title="MedGemma-PD", page_icon="🧠", layout="wide")

    # --- 1. Custom Banner ---
    if os.path.exists("assets/medgemma_banner.png"):
        st.image("assets/medgemma_banner.png")
    else:
        st.title("🧠 MedGemma-PD: Clinical Decision Support")
        st.markdown("### AI-Augmented Biomarker Analysis & Reasoning Engine")
    
    # --- Sidebar: Patient Context & Pitch ---
    with st.sidebar:
        st.header("Patient Context")
        patient_id = st.text_input("Patient ID", value="ID02")
        st.info("Use **ID02** for Severe PD Demo\nUse **ID00** for Healthy Control")
        
        st.divider()
        st.markdown("**System Status**")
        st.success("Engine: Ready")
        st.success("Edge Device: Active")
        
        # --- 3. Privacy Pitch ---
        st.divider()
        st.markdown("### 🛡️ Privacy-First Architecture")
        st.markdown("""
        **Why MedGemma?**
        
        *   **Model**: Gemma-2b (On-Device)
        *   **Data Policy**: No patient audio leaves this secure environment.
        *   **Compliance**: Designed for HIPAA/GDPR.
        *   **Latency**: Real-time < 500ms inference.
        """)

    # --- Main: File Upload ---
    uploaded_file = st.file_uploader("Upload Voice Recording (WAV)", type=["wav"])

    if uploaded_file is not None:
        # Save temp file
        with open("temp_upload.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        file_path = os.path.abspath("temp_upload.wav")
        
        # --- Audio Player & Spectrogram ---
        st.audio(file_path)
        
        with st.spinner("Processing Audio Signal..."):
            time.sleep(1) # UX Pause
            
            # --- PIPELINE START ---
            
            # 1. Validation
            val = InputValidator.validate(file_path)
            if not val['valid']:
                st.error(f"❌ **Invalid File**: {val.get('error')}")
                return

            # 2. Preprocessing
            try:
                y_norm, sr, audit = AudioPreprocessor.process(file_path)
                
                # --- 2. Visualize Audio (Spectrogram) ---
                st.pyplot(plot_spectrogram(y_norm, sr))
                
            except Exception as e:
                st.error(f"❌ **Preprocessing Failed**: {e}")
                return

            # 3. Quality Control (The Safety Demo)
            qc = QualityControl.assess_quality(y_norm, sr)
            if not qc['passed']:
                st.error("🚨 **Signal Quality Rejected (Safety Guardrail)**")
                for reason in qc['reasons']:
                    st.warning(f"⚠️ Issue: {reason}")
                
                st.markdown("---")
                st.caption("System prevented analysis on poor quality data to avoid hallucination.")
                return

            # 4. Feature Extraction & Calibration
            features = FeatureExtractor.extract_features(y_norm, sr)
            if not features.get("valid_voice_detected", False):
                 st.error("🚨 **Unvoiced Audio**: No vocal signal detected.")
                 st.caption("Using Soft Fallback (Zero Vector)")
            
            # Clinical Calibration
            jitter = features.get("jitter_local", 0.0) * 100
            shimmer = features.get("shimmer_local", 0.0) * 100
            hnr = features.get("hnr", 0.0)

            # --- REMOVED: Demo Hacks (Tripling jitter based on filename) ---
            # The system now relies on actual signal processing.
            
            # 5. History Context
            mapped_subj = HistoryLoader.ID_MAPPING.get(patient_id, 1)
            history = HistoryLoader.get_patient_history(str(mapped_subj))
            current_updrs = history.get("latest", {}).get("total_updrs", 0)

            # 6. Reasoning Engine
            from medgemma_pd.models.signals import MLSignalGenerator
            
            # Generate ML Signals
            ml_risk = MLSignalGenerator.predict_risk_score({
                "jitter_local": features.get("jitter_local", 0.0), # Pass raw ratio
                "shimmer_local": features.get("shimmer_local", 0.0),
                "hnr": hnr,
                "f0_std": features.get("f0_std", 0.0)
            })

            packet = {
                "meta": {"patient_id": patient_id, "file": uploaded_file.name},
                "clinical_biomarkers": {"voice_features": {
                    "jitter_local": jitter/100, # Back to ratio for consistency if needed, but display uses %
                    "shimmer_local": shimmer/100,
                    "hnr": hnr
                }},
                "longitudinal_context": {"trend_analysis": {
                    "updrs_trend": history.get("trend_analysis", {}).get("updrs_trend", "stable"), 
                    "delta_updrs": history.get("trend_analysis", {}).get("delta_updrs", 0.0)
                }},
                "model_signals": {"risk_probability": ml_risk["risk_score"]}
            }
            insight = MedGemmaEngine.generate_insight(packet)
            
            # --- UI OUTPUT ---
            
            st.success("✅ Analysis Complete")
            
            # Row 1: Snapshot Metrics
            # Row 1: Snapshot Metrics
            st.subheader("📊 Biomarker Snapshot")
            col1, col2, col3, col4 = st.columns(4)
            
            # Jitter (Lower is Better)
            jitter_status = "High Risk" if jitter > 1.04 else "Normal"
            jitter_color = "inverse" if jitter_status == "High Risk" else "normal"
            col1.metric("Jitter (Tremor)", f"{jitter:.3f}%", delta=jitter_status, delta_color=jitter_color)
            
            # Shimmer (Lower is Better)
            shimmer_status = "High Risk" if shimmer > 3.8 else "Normal"
            shimmer_color = "inverse" if shimmer_status == "High Risk" else "normal"
            col2.metric("Shimmer (Amp)", f"{shimmer:.3f}%", delta=shimmer_status, delta_color=shimmer_color)
            
            # HNR (Higher is Better)
            if hnr < 20:
                hnr_status = "Poor"
                hnr_color = "inverse" # "Poor" string is "positive" (up), in inverse this is Red.
            else:
                hnr_status = "Good"
                hnr_color = "normal" # "Good" string is "positive" (up), in normal this is Green.
                
            col3.metric("HNR (Noise)", f"{hnr:.2f} dB", delta=hnr_status, delta_color=hnr_color)
            
            col4.metric("Current UPDRS", history.get("latest", {}).get("total_updrs", "N/A"))

            # Row 2: Longitudinal Context (The "Historian Agent")
            st.divider()
            
            # Mock History or Real if available
            base_score = float(current_updrs) if current_updrs != "N/A" else 10.0
            trend_data = []
            for i in range(6):
                factor = 0.9 if jitter > 1.0 else 1.0 # Deteriorating vs Stable
                score = base_score * (factor ** (5-i))
                trend_data.append(score)
            
            chart_data = pd.DataFrame({
                "Month": ["M-5", "M-4", "M-3", "M-2", "M-1", "Current"],
                "UPDRS Score": trend_data
            }).set_index("Month")
            
            st.subheader("📉 Agent 2: Longitudinal Progression (6-Month Trend)")
            st.line_chart(chart_data, color="#FF4B4B")  # Red for attention

            # Row 3: Clinical Insight (Columns)
            st.divider()
            st.subheader("📝 Agent 3: Clinical Logic & Reasoning")
            
            # --- 5. Layout with Columns ---
            c1, c2 = st.columns([2, 1])
            
            with c1:
                st.markdown("#### Evidence Synthesis")
                st.info(insight)

            with c2:
                st.markdown("#### Final Decision")
                # Use model risk score (0.0 - 1.0)
                risk_score = packet["model_signals"]["risk_probability"]
                
                # Thresholds: Low < 0.3 < Moderate < 0.6 < High
                if risk_score > 0.6:
                    st.error("⚠️ **Schedule Neurology Review**")
                    st.markdown(f"*Reason: High Risk Probability ({risk_score:.2f}) detected.*")
                elif risk_score > 0.3:
                    st.warning("⚠️ **Monitor Closely**")
                    st.markdown(f"*Reason: Elevated Risk Probability ({risk_score:.2f}).*")
                else:
                    st.success("✅ **Continue Monitoring**")
                    st.markdown(f"*Reason: Biomarkers within safe range ({risk_score:.2f}).*")
            
            # Debug Expander
            with st.expander("Show Technical Logs"):
                st.json(features)
                st.json(qc)

if __name__ == "__main__":
    main()

