# Supplementary File S1: TRIPOD Checklist
## Transparent Reporting of a Multivariable Prediction Model for Individual Prognosis or Diagnosis

**Study:** MedGemma-PD: An Agentic Physiological Speech Biomarker Pipeline for Unconstrained Remote Parkinson's Disease Monitoring  
**Checklist Version:** TRIPOD — Prediction Model Development and Validation (BMJ 2015;350:g7594)  
**Completed by:** [Author Name]  
**Date:** April 2026

> **D** = Applicable to Development data | **V** = Applicable to Validation data | **D;V** = Both

---

## Section 1: Title and Abstract

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 1 | Title (D;V) | Identify the study as developing **and** validating a multivariable prediction model, the target population, and the outcome to be predicted. | Title Page |
| 2 | Abstract (D;V) | Provide a summary of objectives, study design, setting, participants, sample size, predictors, outcome, statistical analysis, results, and conclusions. | p. 1 (Abstract) |

**Item 1 — Evidence:** The title identifies (a) pipeline development ("An Agentic Physiological Speech Biomarker Pipeline"), (b) the target population ("Remote Parkinson's Disease Monitoring"), and (c) the outcome (PD vs. Healthy Control classification).

**Item 2 — Evidence:** The Abstract reports: objectives (remote PD monitoring), design (retrospective mPower cohort), setting (unconstrained mHealth), participants (N=120 age-gated), predictors (Shimmer, Jitter, HNR), outcome (binary PD classification), statistical method (LOSO GroupKFold, bootstrap CIs), results (AUC 0.700), and conclusions (physiological superiority over deep learning).

---

## Section 2: Introduction

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 3a | Background (D;V) | Explain the medical context and rationale for developing or validating the multivariable prediction model, including references to existing models. | p. 2 (§1 Introduction) |
| 3b | Objectives (D;V) | Specify the objectives, including whether the study describes the development or validation of the model or both. | p. 2 (§1, final paragraph) |

**Item 3a — Evidence:** Section 1 describes: (i) the clinical burden of episodic UPDRS assessment, (ii) the failure of existing lab-grade voice AI models in unconstrained environments, and (iii) the unmet need for clinical translation via LLM agentic synthesis. References to prior models (Tsanas et al. [4], Wroge et al. [3]) are included.

**Item 3b — Evidence:** Section 1 (Contributions) explicitly states four objectives: confounder disentanglement, robust acoustic extraction, XAI mapping, and agentic clinical translation. Section 2.6 further specifies the zero-shot **external validation** on MDVR-KCL as a separate objective.

---

## Section 3: Methods

### Source of Data

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 4a | Source of data (D;V) | Describe the study design or source of data, separately for the development and validation datasets. | p. X (§4.1 Methods) |
| 4b | Key dates (D;V) | Specify key study dates including start and end of accrual, and end of follow-up if applicable. | p. X (§4.1 Methods) |

**Item 4a — Evidence:**
- **Development:** Retrospective observational cohort from the mPower longitudinal study (Synapse ID: syn4993293, Bot et al. 2016 [19]). Unconstrained smartphone audio recordings collected via the ResearchKit iOS application.
- **Validation:** Independent lab-recorded dataset — MDVR-KCL (Klumpp et al. 2022 [24]), N=37, collected under controlled acoustic conditions at King's College London.

**Item 4b — Evidence:** The mPower study ran from 2015 onwards. Data were accessed in 2024–2026 for this analysis. MDVR-KCL recordings were collected 2014–2016. No prospective follow-up was conducted; this is a cross-sectional classification study.

---

### Participants

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 5a | Setting (D;V) | Specify key elements of study setting (e.g., primary vs. secondary care, general population). | p. X (§4.1) |
| 5b | Eligibility criteria (D;V) | Describe eligibility criteria for participants. | p. X (§4.1) |
| 5c | Treatments (D;V) | Give details of treatments received, if relevant. | N/A |

**Item 5a — Evidence:** Both datasets are general-population mHealth/clinical research settings. mPower: US general public via smartphone app. MDVR-KCL: outpatient neurology clinic (King's College Hospital, London).

**Item 5b — Evidence:**
- **Development inclusion:** (i) Professional PD diagnosis field populated, (ii) age ≥ 45 years (to neutralize laryngeal aging confounders), (iii) successful audio extraction yielding ≥ 1 valid voiced segment.
- **Development exclusion:** Age < 45 years (HCs), failed or corrupt audio files (b'\x00\x00\x00\x1c' format errors).
- **Validation:** All 37 MDVR-KCL subjects used; no additional age-gating applied (mean age comparable to development cohort).

**Item 5c — Evidence:** Not applicable. This is a diagnostic/classification study; no intervention or treatment was administered. Medication status was not controlled for (acknowledged as a limitation in §3).

---

### Outcome

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 6 | Outcome (D;V) | Clearly define the outcome predicted by the prediction model, including how and when assessed. | p. X (§4.2) |

**Item 6 — Evidence:** The outcome is binary: **Parkinson's Disease (PD = 1) vs. Healthy Control (HC = 0)**. In the mPower development set, this was determined by the `professional-diagnosis` field from the Synapse demographics table (syn5511429) — a self-reported field indicating whether the participant received a professional PD diagnosis. In the MDVR-KCL validation set, labels were provided by the dataset curators based on clinical diagnosis by a consultant neurologist.

---

### Predictors

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 7a | Predictor definition (D;V) | Clearly define all predictors used, including how and when they were measured. | p. X (§4.3) |
| 7b | Missing data (D;V) | Report the number of participants with missing data for predictors and outcome. | p. X (§4.1, Table 1) |
| 7c | Validation predictions (V) | For validation, describe how predictions were made (same model, updated model, or re-estimated model). | p. X (§2.6) |

**Item 7a — Evidence:** Four acoustic predictors were retained after Pearson correlation pruning (r ≤ 0.9):
1. **Shimmer (local)** — Cycle-to-cycle amplitude perturbation, extracted from voiced frames via Praat-equivalent signal processing on 16 kHz resampled audio.
2. **Jitter (local)** — Cycle-to-cycle fundamental frequency perturbation.
3. **Harmonics-to-Noise Ratio (HNR)** — Signal-to-noise ratio in voiced speech.
4. **F0 standard deviation (F0\_std)** — Prosodic variance of fundamental frequency.

All features were extracted per-recording after TQWT denoising (Row A2) or raw signal (Row A1), then z-score normalized within each training fold to prevent data leakage.

**Item 7b — Evidence:** Subjects with completely failed audio extraction (corrupt format) were excluded pre-modelling. No partial feature missingness was observed in successfully processed files. The final cohort (N=120) represents complete-case analysis.

**Item 7c — Evidence:** For zero-shot MDVR-KCL validation (§2.6), the **identical frozen model** trained on mPower (N=120) was applied directly. No recalibration, fine-tuning, or threshold adjustment was performed. Predictions used the same decision threshold (0.5) and the same pre-processing pipeline.

---

### Sample Size

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 8 | Sample size (D;V) | Explain how the study size was arrived at. | p. X (§4.1) |

**Item 8 — Evidence:** The development cohort (N=120, 60 PD / 60 HC) was determined by the availability of valid audio extractions from mPower after applying the age ≥ 45 eligibility gate. The gate was set to balance (a) class balance (1:1 PD:HC ratio achieved via age-gating) and (b) maximizing cohort size within the age-matched stratum. No formal a priori power calculation was conducted; the sample size is consistent with prior mHealth PD voice studies (e.g., Wroge et al. [3]: N=65; Sakar et al. [1]: N=195 after preprocessing). The validation set (N=37) represents the complete MDVR-KCL corpus.

---

### Missing Data

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 9 | Missing data (D;V) | Describe how missing data were handled. | p. X (§4.1) |

**Item 9 — Evidence:** Missing data were handled via **complete-case exclusion** at two stages: (i) subjects with missing `professional-diagnosis` fields were excluded from the Synapse query, and (ii) subjects whose audio files yielded decoding errors were excluded from feature extraction. No imputation was applied. Age was missing for a subset of mPower participants; these were excluded from demographic reporting (Table 1) but subjects with valid audio and diagnosis were retained if audio features could be extracted regardless.

---

### Statistical Analysis Methods

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 10a | Predictor handling (D;V) | Describe how predictors were handled in the analyses. | p. X (§4.4) |
| 10b | Model building (D) | Specify type of model, model-building procedures (including predictor selection), and method for internal validation. | p. X (§4.4) |
| 10c | Model updating for validation (V) | Describe any model updating done before performance assessment in the validation set. | p. X (§2.6) |
| 10d | Performance measures (D;V) | Specify all measures used to assess model performance, and to compare multiple models. | p. X (§4.5) |
| 10e | Model updating results (V) | Describe results of any model updating prior to performance assessment. | N/A |

**Item 10a — Evidence:** Predictors were: (i) extracted per recording, (ii) subjected to Pearson correlation pruning (r > 0.9 → feature dropped) computed on the **training fold only** to prevent leakage, (iii) z-score standardized using training-fold mean/std, and (iv) SMOTE [20] oversampling applied to the training fold only to address within-fold class imbalance. Test fold data were never used in any preprocessing computation.

**Item 10b — Evidence:**
- **Model type:** Random Forest Classifier [21] (primary); Logistic Regression, XGBoost, SVM-RBF (comparators).
- **Predictor selection:** Pearson pruning (r ≤ 0.9 threshold, applied per fold on training data only).
- **Hyperparameter optimisation:** GridSearchCV with 3-fold inner CV on training data; parameters searched: `n_estimators` ∈ {100, 200, 300}, `max_depth` ∈ {3, 5, None}, `min_samples_split` ∈ {2, 5}.
- **Internal validation:** **Leave-One-Subject-Out (LOSO) GroupKFold** — subjects never split across folds, preventing subject-level data leakage. Aggregated predictions across all folds used for final AUC estimation.
- **Confidence intervals:** 1,000-iteration non-parametric bootstrap on the aggregated LOSO predictions.

**Item 10c — Evidence:** No model updating was performed for validation. The mPower-trained model was applied zero-shot to MDVR-KCL. This was an intentional methodological choice to test generalizability without domain adaptation.

**Item 10d — Evidence:** Primary performance measure: **AUC-ROC** (reported with 95% bootstrap CI). Secondary: Accuracy, Sensitivity (Recall), Specificity. Statistical significance of RF vs. SVM baseline: **DeLong's Test** [2] (exact, paired). Ablation comparison: 7-row ablation study (Table 3) using the same LOSO-bootstrap framework.

**Item 10e — Evidence:** Not applicable. No model updating was performed.

---

### Risk Groups

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 11 | Risk groups (D;V) | Provide details on how risk groups were created, if done. | N/A |

**Item 11 — Evidence:** Not applicable. The model outputs a continuous probability score (0–1) converted to binary classification at threshold 0.5. No discrete risk strata (e.g., low/medium/high risk) were created. The agentic LLM layer generates narrative risk language ("At Risk" / "Monitor") but this is based on the continuous probability, not predefined groups.

---

### Development vs. Validation

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 12 | Validation differences (V) | Identify any differences from the development data in setting, eligibility criteria, outcome, and predictors. | p. X (§2.6) |

**Item 12 — Evidence (Development vs. Validation dataset comparison):**

| Dimension | Development (mPower) | Validation (MDVR-KCL) |
|:--|:--|:--|
| Recording environment | Unconstrained smartphone (ambient noise) | Controlled lab (quiet room) |
| Device | Consumer smartphones (iOS) | Professional microphone |
| Audio format | .m4a (AAC encoded) | .wav (PCM) |
| Sample rate | Variable (resampled to 16 kHz) | 16 kHz native |
| N | 120 (60 PD, 60 HC) | 37 (16 PD, 21 HC) |
| Age gate | ≥ 45 years applied | No additional gating |
| Geography | USA | UK (London) |
| Diagnosis basis | Self-report professional diagnosis | Clinical neurologist confirmation |

These differences are explicitly discussed in §2.6 as potential sources of the performance gap between development (AUC 0.700) and zero-shot validation performance.

---

## Section 4: Results

### Participants

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 13a | Participant flow (D;V) | Describe the flow of participants through the study, including the number with and without the outcome. | p. X (§2.1) |
| 13b | Participant characteristics (D;V) | Report characteristics of participants including demographics, clinical features, and missing data. | p. X (Table 1) |
| 13c | Development vs. validation comparison (V) | Show a comparison of important variables between development and validation data. | p. X (§2.6, Item 12 above) |

**Item 13a — Evidence:**
- mPower: All participants with `professional-diagnosis` field present → age-gate (≥ 45) → valid audio extraction → **N=120** (60 PD, 60 HC).
- MDVR-KCL: All 37 subjects used (no additional exclusions). 21 HC, 16 PD.

**Item 13b — Evidence:** Table 1 reports: group (HC/PD), N, Age (mean ± std), Female (%). Age missingness and gender missingness percentages are noted in §2.1.

**Item 13c — Evidence:** Section 2.6 and Item 12 (above) compare development vs. validation data across recording environment, device, format, N, age policy, geography, and diagnosis method.

---

### Model Development

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 14a | Development analysis (D) | Specify the number of participants and outcome events in each analysis. | p. X (§2.1, §2.2) |
| 14b | Unadjusted associations (D) | If done, report the unadjusted association between each candidate predictor and the outcome. | p. X (§2.4, Figure 2) |

**Item 14a — Evidence:** Development analysis: N=120, 60 PD outcome events, 60 HC (non-event). All 120 participants contributed to the LOSO cross-validation. Per-fold sample sizes varied (N≈108 training, N≈12 test per fold).

**Item 14b — Evidence:** SHAP (SHapley Additive exPlanations [9]) values provide model-consistent feature-outcome associations (Figure 2): Shimmer 38.2%, Jitter 31.8%, HNR 19.7%, F0\_std 10.3%. Traditional univariate associations (OR, χ²) are not separately reported — SHAP importance is used as the interpretability framework per current XAI best practice [9].

---

### Model Specification

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 15a | Full model (D) | Present the full prediction model to allow predictions for individuals. | §4.4 + Code Availability §6.4 |
| 15b | Explanation of use (D) | Explain how to use the prediction model. | §2.5, §6.4 |

**Item 15a — Evidence:** The full model specification (Random Forest hyperparameters, feature list, preprocessing steps) is documented in §4.4 of the paper and in full in the publicly available source code (§6.4 Code Availability). The four retained predictors (Shimmer, Jitter, HNR, F0\_std) and their extraction methodology are defined in §4.3. Model weights are serialized as a `.pkl` file in the repository.

**Item 15b — Evidence:** Section 2.5 describes the on-device inference pipeline and the agentic LLM integration. The README.md in the code repository provides step-by-step instructions for applying the model to new audio recordings.

---

### Model Performance

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 16 | Model performance (D;V) | Report performance measures with confidence intervals. | p. X (Table 2, Table 3, §2.2, §2.6) |

**Item 16 — Evidence:**
- **Development performance (primary):** AUC-ROC = **0.700 [95% CI: 0.606–0.793]** (Table 2, §2.2). Accuracy = 0.650, Sensitivity = 0.683, Specificity = 0.583.
- **Ablation performance:** Full 7-row comparison in Table 3 with individual 95% bootstrap CIs for each architecture (A1–A7).
- **Validation performance (zero-shot):** Reported in §2.6 for MDVR-KCL cross-corpus application.
- **Calibration:** Not formally assessed (Brier Score / calibration plot not included — acknowledged as a limitation).

---

### Model Updating

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 17 | Validation updating results (V) | If done, report results from any model updating. | N/A |

**Item 17 — Evidence:** Not applicable. No model updating, recalibration, or threshold adjustment was performed on the MDVR-KCL validation set. This was by design to test true zero-shot cross-corpus generalizability.

---

## Section 5: Discussion

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 18 | Limitations (D;V) | Discuss any study limitations, including impact on predictive performance. | p. X (§3 Discussion) |
| 19a | Validation interpretation (V) | Discuss validation results with reference to development performance and other validation data. | p. X (§2.6, §3) |
| 19b | Overall interpretation (D;V) | Give an overall interpretation considering objectives, limitations, and results from similar studies. | p. X (§3 Discussion) |
| 20 | Implications (D;V) | Discuss the potential clinical use of the model and implications for future research. | p. X (§2.5, §3) |

**Item 18 — Evidence (Limitations reported in §3):**
1. **Small cohort (N=120):** Limits statistical power for subgroup analyses; future work should target N>500.
2. **Sex imbalance:** mPower convenience sampling yields disproportionate female representation in the HC group (10.0% female HC vs. 30.0% female PD), which may bias sex-specific vocal feature distributions.
3. **Self-reported diagnosis:** mPower PD labels are based on professional diagnosis self-report, not verified clinical records. Misclassification cannot be excluded.
4. **No calibration assessment:** Brier Score and calibration plots were not computed; clinical deployment would require formal calibration evaluation.
5. **Medication status not controlled:** Dopaminergic medication (L-DOPA) is known to partially normalise vocal biomarkers in treated PD patients, potentially attenuating the diagnostic signal.
6. **LLM subjectivity:** Gemma-2b clinical narrative outputs are not validated against expert neurologist triage standards; a formal Randomized Controlled Trial (RCT) against board-certified neurologist assessments is required before clinical deployment.

**Item 19a — Evidence:** Section 2.6 compares development AUC (0.700, mPower) against zero-shot validation on MDVR-KCL. The performance gap is attributed to domain shift (unconstrained vs. lab audio), dataset size difference (N=120 vs. N=37), and the absence of any model adaptation to the target domain.

**Item 19b — Evidence:** Section 3 discusses results in the context of: Tsanas et al. [4] (Oxford lab benchmark: AUC >0.99), Wroge et al. [3] (mHealth benchmark: AUC ≈0.72), and this study's position as a rigorously confounded real-world result. The discussion contextualises AUC 0.700 as a strong result for unconstrained telemetry.

**Item 20 — Evidence:** Section 2.5 describes the clinical use pathway: Edge Student model (1.1 MB, 31.9 ms) deployed on consumer smartphones, outputting structured triage narratives via Gemma LLM. Section 3 calls for RCT validation comparing agentic triage outputs vs. neurologist assessments as the required next step before clinical translation.

---

## Section 6: Other Information

| Item | Topic | Checklist Item | Reported on Page |
|:----:|:------|:---------------|:----------------:|
| 21 | Supplementary information (D;V) | Provide information about availability of supplementary resources (protocol, raw data, code). | §6.3, §6.4 |
| 22 | Funding (D;V) | Give the source of funding and the role of funders. | §6.2 |

**Item 21 — Evidence:**
- **Raw data:** Available via Sage Bionetworks Synapse platform (ID: syn4993293) under the mPower governance agreement (§6.3).
- **Source code:** Publicly available at [PLACEHOLDER — GitHub/Zenodo URL] (§6.4).
- **Pre-processed features and model weights:** Available from the corresponding author upon reasonable request.
- **MDVR-KCL validation dataset:** Publicly available per Klumpp et al. [24].

**Item 22 — Evidence:** This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors (§6.2).

---

## TRIPOD Summary: Compliance Overview

| Item | Reported | Notes |
|:----:|:--------:|:------|
| 1 | ✅ | Title identifies development + validation, population, outcome |
| 2 | ✅ | Abstract covers all required elements |
| 3a | ✅ | Medical context and prior model references provided |
| 3b | ✅ | Objectives clearly state both development and validation |
| 4a | ✅ | mPower (development) and MDVR-KCL (validation) described separately |
| 4b | ✅ | Data collection dates noted |
| 5a | ✅ | mHealth general population + outpatient clinic settings described |
| 5b | ✅ | Age ≥ 45 eligibility criterion clearly stated |
| 5c | ✅ | N/A — no treatments |
| 6 | ✅ | Binary outcome (PD vs HC) clearly defined with source of labels |
| 7a | ✅ | All 4 predictors defined with extraction methodology |
| 7b | ✅ | Complete-case approach; no partial missingness after extraction |
| 7c | ✅ | Same frozen model applied zero-shot — explicitly stated |
| 8 | ✅ | Sample size rationale provided; consistent with literature |
| 9 | ✅ | Complete-case exclusion described |
| 10a | ✅ | Pearson pruning + SMOTE + z-score (training fold only) |
| 10b | ✅ | RandomForest + GridSearchCV + LOSO GroupKFold specified |
| 10c | ✅ | No updating for validation — explicitly stated |
| 10d | ✅ | AUC, Accuracy, Sensitivity, Specificity, DeLong's Test |
| 10e | ✅ | N/A — no updating done |
| 11 | ✅ | N/A — binary output, no risk strata |
| 12 | ✅ | Table comparing development vs. validation data dimensions |
| 13a | ✅ | Participant flow described (mPower + MDVR-KCL) |
| 13b | ✅ | Table 1: demographics with missingness noted |
| 13c | ✅ | Development vs. validation data comparison in §2.6 |
| 14a | ✅ | N=120, 60 PD / 60 HC specified |
| 14b | ✅ | SHAP importance as feature-outcome association |
| 15a | ✅ | Full model in §4.4 + code repository |
| 15b | ✅ | Usage described in §2.5 and README |
| 16 | ✅ | AUC 0.700 [0.606–0.793] + full ablation Table 3 with CIs |
| 17 | ✅ | N/A — no model updating done |
| 18 | ✅ | 6 limitations discussed in §3 |
| 19a | ✅ | Validation vs. development gap discussed in §2.6 |
| 19b | ✅ | Overall interpretation in §3 vs. Tsanas et al., Wroge et al. |
| 20 | ✅ | Clinical use pathway + RCT call in §2.5 and §3 |
| 21 | ✅ | Data (Synapse), code (GitHub), features (on request) |
| 22 | ✅ | No funding |

**Overall TRIPOD compliance: 22/22 items addressed ✅**

> *Note: Items 5c, 10e, 11, and 17 are reported as "N/A" because they are not applicable to this study design. N/A counts as compliant per TRIPOD guidelines.*

---

*This checklist should be submitted as a supplementary document accompanying the manuscript. Page numbers in the "Reported on Page" column should be updated to match the final formatted manuscript before journal submission.*
