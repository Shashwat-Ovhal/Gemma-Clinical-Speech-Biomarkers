# MedGemma-PD: Clinical Reasoning over Speech Biomarkers for Parkinson’s Disease
**MedGemma Impact Challenge Submission**

## 1. Introduction: The "Black Box" Problem in Digital Biomarkers
Parkinson's Disease (PD) is characterized by subtle motor degradation that often manifests in voice (dysarthria) years before clinical diagnosis. While ML models can classify PD from speech with high accuracy (>90%), they fail clinically because they act as "Black Boxes." A neurologist cannot treat a patient based on a score of `0.87`. They need **reasoning**.

**MedGemma-PD** is not a classifier. It is a **Clinical Decision Support System (CDSS)** that leverages the HAI-DEF MedGemma architecture to reason over scattered speech biomarkers (Jitter, Shimmer, HNR) and longitudinal patient history.

## 2. System Architecture: The MedGemma Advantage
The system is designed to **fail without MedGemma**. It does not perform end-to-end classification. Instead, it follows a strict reasoning pipeline:

### Layer 1: Speech Biomarker Extraction
We process raw audio (PC-GITA dataset) to extract clinically validated features:
- **Jitter/Shimmer**: Measures of vocal stability.
- **HNR (Harmonics-to-Noise Ratio)**: Measure of breathiness/hoarseness.
- **MFCCs**: Mel-frequency cepstral coefficients for spectral shape.
*Critically, these are just numbers. A 2% increase in Jitter is meaningless without context.*

### Layer 2: Baseline Signal Generators ("Weak Learners")
We use interpretable ML models (Logistic Regression) to generate "Risk Signals" and "Trend Analysis" (Slope of UPDRS). These models provide probability masses, NOT diagnoses.

### Layer 3: The Structured Clinical Evidence Packet
All upstream data is aggregated into a JSON packet containing:
- Current Session Metrics
- Longitudinal History (UCI Telemonitoring)
- Confidence Intervals & Missing Data Flags

### Layer 4: MedGemma (The Core)
MedGemma consumes the Evidence Packet. Using a specialized clinical system prompt, it:
1.  **Synthesizes**: Correlates the rising Jitter with the worsening UPDRS score.
2.  **Contextualizes**: Compares values against age-matched norms.
3.  **Reasons**: Determines that the *rate of change* is more concerning than the absolute value.
4.  **Communicates**: Generates a neurologist-grade clinical note.

Without MedGemma, the user sees a table of numbers. With MedGemma, they see a coherent clinical narrative.

## 3. Demonstration Case Study: Patient P07
**Scenario**: Patient P07 is undergoing telemonitoring.
- **Month 0**: Baseline. Mild vocal tremor.
- **Month 3**: Slight increase in Jitter.
- **Month 6**: Significant degradation in HNR.

**System Output (Before MedGemma)**:
`{ "jitter": 1.2%, "risk_score": 0.72 }` -> **Clinically Useless.**

**System Output (After MedGemma)**:
*"Assessment: High Risk of Motor Progression. The analysis indicates a significant deterioration in motor control stability. Key Evidence: Current Jitter (1.2%) has deviated 20% from baseline, concordant with the upward trend in UPDRS scores. Recommendation: Prioritize adjustment of dopaminergic therapy."* -> **Clinically Actionable.**

## 4. Ethics & Limitations
- **Decision Support Only**: The system explicitly disclaims diagnostic authority.
- **Data Privacy**: All processing can occur locally or via secure Enclaves.
- **Bias Mitigation**: MedGemma is prompted to flag uncertainty when data is sparse (e.g., missing modalities).

## 5. Conclusion
MedGemma-PD demonstrates that the future of medical AI is not higher accuracy classification, but **deeper clinical reasoning**. By anchoring Large Language Models in validated biomarker physics, we bridge the gap between signal processing and patient care.

## 6. Supporting Resources
*   **Project Repository**: [Link to Public GitHub/Kaggle Repo]
*   **Demo Video**: [Link to YouTube/Loom Demo]
*   **Dataset (PC-GITA)**: [Link to Dataset Source]
*   **Presentation Slides**: [Link to Slides PDF]
