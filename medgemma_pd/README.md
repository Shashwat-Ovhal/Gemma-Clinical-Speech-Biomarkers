# MedGemma-PD: Clinical Reasoning over Speech Biomarkers

**A Clinical Decision Support System (CDSS) that uses MedGemma (HAI-DEF) to reason over longitudinal speech biomarkers for Parkinson's Disease.**

Unlike traditional classifiers that output a black-box probability (e.g., "Risk: 0.87"), MedGemma-PD aggregates scattered biomarkers (Jitter, Shimmer, HNR) and patient history into a **Structured Clinical Evidence Packet**. MedGemma then acts as a reasoning engine to synthesize this evidence into a clinician-grade narrative, explaining *why* the signals matter in the context of the patient's progression.

## 🏗️ System Architecture

```ascii
[ Patient Audio ] 
       ⬇
[ 1. Audio Pipeline ] --> (Features: Jitter, Shimmer, HNR, MFCC)
       ⬇
[ 2. Signal Generators ] --> (Weak Models: Risk Score, Trend Slope)
       ⬇
[ 3. PACKET BUILDER ] --> 📦 { Evidence Packet (JSON) }
                                    ⬇
                            [ 4. MEDGEMMA (Reasoning) ] 🧠
                                    ⬇
                            📄 "Clinical Note: High Risk..."
                                    ⬇
                            [ 5. Dashboard (UI) ]
```

## 🧠 Why MedGemma is Essential?
**This system fails without MedGemma.**
If you remove the reasoning layer, the output is just a raw table of numbers (e.g., "Jitter: 1.2%"). These numbers are clinically ambiguous on their own. MedGemma is required to:
1.  **Contextualize** the biomarkers against the patient's specific history.
2.  **Weigh** conflicting signals (e.g., "Good HNR but bad Jitter").
3.  **Generate** the human-readable assessment that doctors trust.

## 🚀 Quick Start
This repo includes an **Auto-Mock Mode** for high-reliability demos. It runs even if you lack audio libraries.

1.  **Run the Analysis**:
    ```bash
    python main.py --patient P07 --session 6
    ```
2.  **View the Dashboard**:
    Open `medgemma_pd/ui/index.html` in your browser.

## 📂 Structure
*   `audio_pipeline/`: Robust feature extraction (Praat/Librosa).
*   `models/`: Baseline ML "weak learners" (Signal Generators).
*   `reasoning/`: Packet Builder & MedGemma Prompts.
*   `ui/`: Premium static dashboard.
