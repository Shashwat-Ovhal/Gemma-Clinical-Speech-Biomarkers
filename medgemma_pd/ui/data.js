window.medgemmaData = {
  "packet": {
    "meta": {
      "patient_id": "P07",
      "session_month": 6,
      "data_source": "PC-GITA + UCI Telemonitoring"
    },
    "clinical_biomarkers": {
      "voice_features": {
        "valid_voice_detected": true,
        "f0_mean": 137.32736638492162,
        "f0_std": 5.660298113369516,
        "jitter_local": 0.005788537168964949,
        "shimmer_local": 0.04756279286853845,
        "hnr": 23.708673264322613,
        "mfcc_mean": -10.33433149483482,
        "mfcc_std": 41.83949913886199,
        "spectral_centroid": 1952.6562547738188
      },
      "reference_ranges": {
        "jitter_normal": "< 1.04%",
        "hnr_normal": "> 20 dB"
      }
    },
    "longitudinal_context": {
      "history_summary": [
        {
          "subject_id": "P07",
          "session_month": 0,
          "motor_updrs": 15,
          "total_updrs": 22,
          "jitter_percent": 0.6,
          "hnr": 22.0
        },
        {
          "subject_id": "P07",
          "session_month": 3,
          "motor_updrs": 18,
          "total_updrs": 28,
          "jitter_percent": 0.9,
          "hnr": 18.0
        }
      ],
      "trend_analysis": {
        "updrs_slope": 2.1666666666666656,
        "updrs_trend": "worsening",
        "jitter_slope": 0.09999999999999998
      }
    },
    "model_signals": {
      "risk_probability": 0.0191,
      "signal_interpretation": "Within normal limits"
    },
    "confidence_assessment": {
      "data_quality": "High",
      "missing_modalities": [
        "motor_assessment",
        "cognitive_score"
      ]
    }
  },
  "insight": "\n### MedGemma Clinical Insight\n**Patient P07 | Session Month 6**\n\n**Assessment**: High Risk of Motor Progression\nThe analysis of current speech biomarkers combined with longitudinal history indicates a significant deterioration in motor control stability.\n\n**Key Evidence**:\n1.  **Biomarker Deviation**: Current Jitter is **0.58%** (Norm: <1.04%), which is a marked increase from Month 3.\n2.  **Longitudinal Trend**: UPDRS scores show a consistent **worsening** trajectory over the last 6 months.\n3.  **Signal Concordance**: The ML risk signal (0.02) aligns with the observed degradation in Harmonic-to-Noise Ratio (HNR).\n\n**Uncertainty & Gaps**:\nThis assessment relies solely on speech telemonitoring. Absence of recent clinical motor exams limits definitive staging.\n\n**Recommendation**:\nSchedule in-person neurology review within 4 weeks. Prioritize adjustment of dopaminergic therapy.\n            "
};