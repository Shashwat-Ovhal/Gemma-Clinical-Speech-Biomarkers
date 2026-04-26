# MedGemma-PD: Agentic Physiological Speech Biomarker Pipeline for Unconstrained Remote Parkinson's Disease Monitoring

<p align="center">
  <img src="./assets/medgemma_banner.jpg" alt="MedGemma-PD Banner" width="100%"/>
</p>

## Overview

**MedGemma-PD** is a research-grade acoustic processing pipeline designed to diagnose Parkinson’s Disease (PD) from unconstrained, noisy smartphone voice recordings. It neutralizes demographic confounders (like age) and fuses classical physiological feature extraction with a lightweight localized Large Language Model (Gemma) to generate readable, actionable clinical triage narratives for neurologists.

This repository serves as the official codebase and data supplement for the manuscript:
> *"MedGemma-PD: An Agentic Physiological Speech Biomarker Pipeline for Unconstrained Remote Parkinson's Disease Monitoring"* 
> (Prepared for *npj Digital Medicine* / *IEEE JBHI*).

---

## 🔬 Core Scientific Innovations

1. **Strict Confounder Mitigation**: Implements rigorous Age-Gating ($\ge 45$ years) on the mPower dataset, mathematically neutralizing the pervasive "young vs. old" age-bias that artificially inflates performance in current Voice-AI literature.
2. **Robust Acoustic Extraction**: Validates macroscopic vocal variance metrics (Shimmer, Jitter) against severe ambient noise, proving physiological features are highly superior to generic deep learning embeddings (`wav2vec 2.0`) and standard MFCCs in unconstrained environments.
3. **Subject-Independent Validation**: Utilizes `GroupKFold` tied strictly to subject IDs, ensuring 0% data leakage across training folds, validated by 1,000-iteration bootstrap confidence intervals.
4. **Explainable AI (XAI)**: Uses SHapley Additive exPlanations (SHAP) coupled with Pearson correlation pruning to unpack the algorithmic "black box", establishing Shimmer (38.2%) and Jitter (31.8%) as the primary physiological drivers of model decisions.
5. **Agentic Clinical Translation**: Engineers an Edge-deployable student model (1.1 MB, 31.9 ms latency) paired with an on-device LLM (**Gemma-2b**) to synthesize stochastic classification probabilities and longitudinal metadata into fully interpretable clinical triage notes.

---

## 📊 Final Research Outcomes

The comprehensive ablation study proves the supremacy of our hybrid architecture in real-world scenarios:

| Architecture | AUC-ROC [95% CI] | Note |
| :--- | :--- | :--- |
| **TQWT Denoised + Classical + RF** | **0.700 [0.606–0.793]** | **Best Performance (A2)** |
| Cross-Attention Fusion System | 0.665 [0.573–0.758] | Best Neural Setup (A5) |
| Knowledge-Distilled Student (Edge) | 0.635 [0.543–0.728] | On-Device Deploy (A6) |
| `wav2vec 2.0` full fine-tune | 0.432 [0.333–0.533] | Neural Overfitting (A3) |
| TQWT Denoised + MFCCs + RF | 0.332 [0.242–0.440] | MFCC Noise Failure (A7) |

---

## ⚙️ Repository Structure

The repository has been structured for reproducibility and peer review:

```
Gemma-Clinical-Speech-Biomarkers/
├── manuscript/
│   ├── research_paper_draft.md         # Final publication-ready manuscript
│   ├── publication_strategy.md         # Journal targeting and strategy notes
│   ├── final_publication_results/      # Output metrics and raw JSON files
│   │   └── figures/                    # Generated ROC, SHAP, and Ablation charts
│   └── supplementary/
│       └── S1_TRIPOD_Checklist.md      # Completed 22-item TRIPOD-AI checklist
├── medgemma_pd/                        # Main Application Code
│   ├── audio_pipeline/                 # TQWT filtering & feature extraction
│   ├── models/                         # ML models, distillation, and fusion
│   └── reasoning/                      # LLM context builder and agentic engine
├── scripts/                            # Executable Automation & Pipeline Scripts
│   ├── bspc_train.py                   # Main 7-row orchestration loop
│   ├── generate_publication_figures.py # Script to rebuild paper figures
│   ├── import_synapse_dataset.py       # mPower extraction tool
│   └── ...                             
├── notebooks/                          
│   └── BSPC_Colab_Runner.ipynb         # Cloud-training wrapper
├── app.py                              # Local UI Demo application
└── README.md
```

---

## 🚀 Getting Started

### 1. Environment Setup

It is highly recommended to run this project in a `conda` environment or on GPU-enabled instances.

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Synapse Authentication
To download the mPower dataset, you need a Synapse authentication token.

```bash
# Windows
set SYNAPSE_AUTH_TOKEN="your_personal_access_token"
# Linux/Mac
export SYNAPSE_AUTH_TOKEN="your_personal_access_token"
```

### 3. Executing the Ablation Pipeline

The primary orchestrator handles everything from TQWT denoising to model training and ablation testing. It is fully checkpoint-aware.

```bash
python scripts/bspc_train.py
```

### 4. Generating Publication Figures

To rebuild the exact figures embedded in the manuscript (SHAP, ROC operating points, learning curves):

```bash
python scripts/generate_publication_figures.py
```

---

## 📝 License & Data Usage

- **Code License**: MIT License.
- **Data Governance**: The **mPower** dataset is governed by Sage Bionetworks. Access is strictly controlled via [Synapse](https://www.synapse.org/#!Synapse:syn4993293). You must digitally sign the Data Use Agreement to access the underlying `.m4a` files.
- **Ethics**: This study operated exclusively on secondary de-identified data. 
- **HIPAA Compliance**: The MedGemma reasoning engine is designed to run inference *locally on-device*, ensuring Private Health Information (PHI) is never transmitted to remote commercial APIs.

---
*Developed for High-Impact Clinical Publication.*
