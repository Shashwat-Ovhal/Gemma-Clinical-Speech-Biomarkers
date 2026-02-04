# MedGemma-PD: Clinical Speech Biomarker Pipeline

## Overview
MedGemma-PD is an automated clinical assessment system that analyzes speech biomarkers (Jitter, Shimmer, HNR) and correlates them with patient history (UPDRS scores) to generate logic-driven clinical insights for Parkinson's Disease monitoring.

## Key Features
*   **Robust Audio Pipeline**: Pure Numpy/Scipy implementation (no unstable dependencies like Librosa/Parselmouth).
*   **Smart Trimming**: Automatically strips leading/trailing silence (20-60dB dynamic threshold).
*   **Soft Fallback**: Gracefully handles silent/unvoiced signals without crashing.
*   **Data Consistency**: "Frankenstein Fix" mapping layer links disparate audio IDs to longitudinal patient history.

## Quick Start
1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Pipeline**
    To analyze a patient audio file:
    ```bash
    python main.py --file "path/to/audio.wav" --patient_id ID02
    ```
    *   `--patient_id`: ID02 (Severe PD) or P07/ID00 (Healthy Control).

3.  **View Results**
    *   Console output shows real-time biomarker extraction and reasoning.
    *   Dashboard data generated at `medgemma_pd/ui/data.js`.

## Project Structure
*   `medgemma_pd/`: Core logic packages.
    *   `audio_pipeline/`: Scipy-based signal processing.
    *   `reasoning/`: Clinical logic engine and history loader.
*   `tests/`: Verification scripts and tools.
*   `scripts/`: Mapping and analysis utilities.
