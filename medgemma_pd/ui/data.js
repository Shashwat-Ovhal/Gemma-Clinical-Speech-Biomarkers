window.medgemmaData = {
  "packet": {
    "meta": {
      "patient_id": "34",
      "filename": "dataset- MDVR-KCL Dataset\\26_29_09_2017_KCL\\26-29_09_2017_KCL\\ReadText\\PD\\ID34_pd_2_0_0.wav"
    },
    "clinical_biomarkers": {
      "voice_features": {
        "jitter_local": 0.3743360094962463,
        "shimmer_local": 0.1411097274282635,
        "hnr": 9.622366162325399,
        "f0_mean": 352.53605273598197,
        "valid_voice_detected": true,
        "latency_ms": 111.71317100524902
      }
    },
    "longitudinal_context": {
      "found": true,
      "subject_id": 34,
      "record_count": 161,
      "baseline": {
        "motor_updrs": 29.291,
        "total_updrs": 34.146
      },
      "latest": {
        "motor_updrs": 26.781,
        "total_updrs": 34.817,
        "time_day": 178.68
      },
      "trend_analysis": {
        "updrs_trend": "stable",
        "delta_updrs": 0.67
      }
    },
    "model_signals": {
      "risk_probability": 0.75
    }
  },
  "insight": "### MedGemma Clinical Insight\n**Patient 34 | Automated Assessment**\n\n**Assessment**: At Risk (Risk Signal: 0.75)\nAnalysis of speech biomarkers suggests at risk motor control.\n\n**Evidence Integration**:\n1.  **Speech Biomarkers**:\n    *   Jitter: **37.434%** (Norm: <1.04%) - Elevated\n    *   Shimmer: **14.111%** (Norm: <3.8%)\n    *   HNR: **9.62dB** (Norm: >20dB)\n\n2.  **Longitudinal Context (UCI History)**:\n    *   UPDRS Trend: **Stable**\n    *   Change from Baseline: +0.67 points\n    \n3.  **Synthesis**:\n    The acoustic features (specifically Jitter=37.43%) are divergent with the historical UPDRS trend.\n    \n**Recommendation**:\nSchedule Neurology Review",
  "pipeline_report": {
    "pipeline_version": "1.0.0-medical",
    "timestamp": 1770236579.1553981,
    "status": "success",
    "stages": {
      "validation": {
        "valid": true,
        "error": null,
        "metadata": {
          "sample_rate": 44100,
          "channels": 1,
          "duration_sec": 127.92086167800454
        }
      },
      "preprocessing": {
        "trim_removed_sec": 6.25e-05,
        "resample_rate": 16000,
        "normalization_gain": 0.7277699345785482
      },
      "quality_control": {
        "passed": true,
        "metrics": {
          "duration": 127.9208125,
          "clipping_ratio": 0.0,
          "rms_energy": 0.03769490652833808
        },
        "reasons": []
      },
      "feature_extraction": {
        "jitter_local": 0.3743360094962463,
        "shimmer_local": 0.1411097274282635,
        "hnr": 9.622366162325399,
        "f0_mean": 352.53605273598197,
        "valid_voice_detected": true,
        "latency_ms": 111.71317100524902
      }
    },
    "processing_time": 7.494616746902466,
    "meta": {
      "filename": "ID34_pd_2_0_0.wav",
      "size_bytes": 16923974
    }
  }
};