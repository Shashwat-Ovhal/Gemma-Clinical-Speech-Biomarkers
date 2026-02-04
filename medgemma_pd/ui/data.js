window.medgemmaData = {
  "packet": {
    "meta": {
      "patient_id": "ID02",
      "filename": "C:\\Users\\Shashwat\\OneDrive\\Desktop\\test.wav"
    },
    "clinical_biomarkers": {
      "voice_features": {
        "valid_voice_detected": true,
        "f0_mean": 160.0,
        "f0_std": 10.0,
        "jitter_local": 0.015116000000000001,
        "shimmer_local": 0.1638848315845658,
        "hnr": 24.942,
        "latency_ms": 327.0423412322998
      }
    },
    "longitudinal_context": {
      "found": true,
      "subject_id": 35,
      "record_count": 165,
      "baseline": {
        "motor_updrs": 36.073,
        "total_updrs": 54.073
      },
      "latest": {
        "motor_updrs": 34.163,
        "total_updrs": 53.109,
        "time_day": 202.43
      },
      "trend_analysis": {
        "updrs_trend": "stable",
        "delta_updrs": -0.96
      }
    },
    "model_signals": {
      "risk_probability": 0.75
    }
  },
  "insight": "### MedGemma Clinical Insight\n**Patient ID02 | Automated Assessment**\n\n**Assessment**: At Risk (Risk Signal: 0.75)\nAnalysis of speech biomarkers suggests at risk motor control.\n\n**Evidence Integration**:\n1.  **Speech Biomarkers**:\n    *   Jitter: **1.512%** (Norm: <1.04%) - Elevated\n    *   Shimmer: **16.388%** (Norm: <3.8%)\n    *   HNR: **24.94dB** (Norm: >20dB)\n\n2.  **Longitudinal Context (UCI History)**:\n    *   UPDRS Trend: **Stable**\n    *   Change from Baseline: -0.96 points\n    \n3.  **Synthesis**:\n    The acoustic features (specifically Jitter=1.51%) are divergent with the historical UPDRS trend.\n    \n**Recommendation**:\nSchedule Neurology Review",
  "pipeline_report": {
    "pipeline_version": "1.0.0-medical",
    "timestamp": 1770228298.4422805,
    "status": "success",
    "stages": {
      "validation": {
        "valid": true,
        "error": null,
        "metadata": {
          "sample_rate": 44100,
          "channels": 1,
          "duration_sec": 157.55666666666667
        }
      },
      "preprocessing": {
        "trim_removed_sec": 6.25e-05,
        "resample_rate": 16000,
        "normalization_gain": 0.7065517288781947
      },
      "quality_control": {
        "passed": true,
        "metrics": {
          "duration": 157.556625,
          "clipping_ratio": 0.0,
          "rms_energy": 0.04397392554152084
        },
        "reasons": []
      },
      "feature_extraction": {
        "valid_voice_detected": true,
        "f0_mean": 160.0,
        "f0_std": 10.0,
        "jitter_local": 0.015116000000000001,
        "shimmer_local": 0.1638848315845658,
        "hnr": 24.942,
        "latency_ms": 327.0423412322998
      }
    },
    "processing_time": 27.042312383651733,
    "meta": {
      "filename": "test.wav",
      "size_bytes": 20844791
    }
  }
};