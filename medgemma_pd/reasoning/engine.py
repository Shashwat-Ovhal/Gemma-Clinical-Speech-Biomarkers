import json

class MedGemmaEngine:
    """
    The Core HAI-DEF Reasoning Engine.
    Constraint: MUST NOT be a classifier. MUST be a reasoning support system.
    """

    SYSTEM_PROMPT = """
    You are MedGemma (HAI-DEF), an advanced Clinical Decision Support System for Neurology.
    Your goal is to assist a neurologist in assessing Parkinson's Disease progression based on speech biomarkers.

    GUIDELINES:
    1. DO NOT DIAGNOSE. You generate "Risk Trajectories" and "Clinical Insights".
    2. SYNTHESIZE evidence from (A) Current Biomarkers, (B) Longitudinal History, and (C) ML Signals.
    3. EXPLAIN your reasoning. Why is the jitter increase significant? How does it compare to the UPDRS trend?
    4. HANDLE UNCERTAINTY. If data is sparse or signals conflict, state it clearly.
    5. OUTPUT FORMAT: Structured Clinical Note (Markdown).
    """

    @staticmethod
    def generate_insight(packet: dict, mock_mode: bool = True) -> str:
        """
        Generates the clinical narrative.
        If mock_mode is True, returns a pre-written template based on the packet data
        to ensure high-quality demo output without live LLM calls.
        """
        patient_id = packet["meta"]["patient_id"]
        jitter = packet["clinical_biomarkers"]["voice_features"].get("jitter_local", 0) * 100
        risk = packet["model_signals"]["risk_probability"]
        trend = packet["longitudinal_context"]["trend_analysis"].get("updrs_trend", "unknown")

        # In a real system, we would do:
        # prompt = f"{MedGemmaEngine.SYSTEM_PROMPT}\nData: {json.dumps(packet)}"
        # response = model.generate(prompt)
        
        if mock_mode:
            return MedGemmaEngine._mock_response(patient_id, jitter, risk, trend)
        
        return "LLM Integration Not Configured"

    @staticmethod
    def _mock_response(pid: str, jitter: float, risk: float, trend: str) -> str:
        """
        Deterministically produces a 'Clinical Note' based on the input signals.
        This ensures the demo is consistent.
        """
        if pid == "P07":
            # Progressing Case logic
            return f"""
### MedGemma Clinical Insight
**Patient {pid} | Session Month 6**

**Assessment**: High Risk of Motor Progression
The analysis of current speech biomarkers combined with longitudinal history indicates a significant deterioration in motor control stability.

**Key Evidence**:
1.  **Biomarker Deviation**: Current Jitter is **{jitter:.2f}%** (Norm: <1.04%), which is a marked increase from Month 3.
2.  **Longitudinal Trend**: UPDRS scores show a consistent **{trend}** trajectory over the last 6 months.
3.  **Signal Concordance**: The ML risk signal ({risk:.2f}) aligns with the observed degradation in Harmonic-to-Noise Ratio (HNR).

**Uncertainty & Gaps**:
This assessment relies solely on speech telemonitoring. Absence of recent clinical motor exams limits definitive staging.

**Recommendation**:
Schedule in-person neurology review within 4 weeks. Prioritize adjustment of dopaminergic therapy.
            """
        
        elif pid == "P08":
            # Stable Control logic
            return f"""
### MedGemma Clinical Insight
**Patient {pid} | Session Month 6**

**Assessment**: Stable / Low Risk
Speech biomarkers remain within the normative range for the patient's age group. No significant deviation from baseline.

**Key Evidence**:
1.  **Biomarker Stability**: Jitter ({jitter:.2f}%) and Shimmer are stable compared to prior sessions.
2.  **Longitudinal Trend**: No significant upward trend in UPDRS or feature variance.
3.  **Signal Concordance**: ML Signal ({risk:.2f}) is negligible.

**Recommendation**:
Continue routine telemonitoring. Next review in 6 months.
            """
        
        else:
            return "Insufficient data for clinical insight."
