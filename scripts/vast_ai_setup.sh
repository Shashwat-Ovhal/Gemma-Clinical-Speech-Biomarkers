#!/bin/bash
# vast_ai_setup.sh — One-shot setup script for vast.ai GPU instance
# ================================================================
# Run this on your vast.ai machine right after it boots.
# 
# BEFORE RUNNING:
#   1. On vast.ai, select an instance with:
#      - Template: "PyTorch 2.x + CUDA 12" (pre-installed)
#      - GPU: RTX 3090 or A5000 ($0.20–0.40/hr)
#      - Storage: 50GB minimum
#      - Port 22 exposed (for SSH)
#
#   2. SCP your project zip to the instance:
#      scp -P <port> Gemma-Clinical-Speech-Biomarkers.zip root@<ip>:/workspace/
#
#   3. SSH in and run this script:
#      bash vast_ai_setup.sh

set -e
echo "=============================================="
echo "  vast.ai BSPC Training Setup"
echo "=============================================="

# Unzip project
cd /workspace
if [ -f "Gemma-Clinical-Speech-Biomarkers.zip" ]; then
    echo "Unzipping project..."
    unzip -q Gemma-Clinical-Speech-Biomarkers.zip
    cd Gemma-Clinical-Speech-Biomarkers
else
    echo "ERROR: Gemma-Clinical-Speech-Biomarkers.zip not found in /workspace"
    exit 1
fi

# Install Python deps
echo "Installing dependencies..."
pip install -q -r requirements.txt
pip install -q transformers torch torchvision torchaudio \
               huggingface_hub soundfile librosa \
               imbalanced-learn shap scikit-learn xgboost

# Verify GPU
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Pre-seed checkpoint with known results to skip completed ablation rows
python - <<'EOF'
import json, os
os.makedirs("./outputs/bspc", exist_ok=True)
checkpoint = "./outputs/bspc/ablation_results.json"

# Only pre-seed if no checkpoint exists yet
if not os.path.exists(checkpoint):
    known = [
        {"row":"A1","desc":"Classical + RF (Level 1 baseline)","auc":0.688,"ci_lo":0.590,"ci_hi":0.783},
        {"row":"A2","desc":"TQWT Denoised + Classical + RF",   "auc":0.691,"ci_lo":0.589,"ci_hi":0.786},
        {"row":"A3","desc":"wav2vec full fine-tune",            "auc":0.502,"ci_lo":0.396,"ci_hi":0.604},
        {"row":"A4","desc":"wav2vec frozen encoder + adapters", "auc":0.605,"ci_lo":0.498,"ci_hi":0.705},
    ]
    with open(checkpoint,"w") as f:
        json.dump(known, f, indent=2)
    print("Checkpoint pre-seeded. Will start from A5.")
else:
    print("Existing checkpoint found:", checkpoint)
EOF

echo ""
echo "Setup complete. Starting training..."
echo "=============================================="
python bspc_train.py

echo ""
echo "Running Phase 7 — MDVR-KCL Zero-Shot Validation..."
python bspc_pipeline/cross_corpus_validation.py

echo ""
echo "=============================================="
echo "  Training Complete!"
echo "  Push results to Hugging Face Hub:"
echo "  python hf_push.py --token YOUR_HF_TOKEN --repo YOUR_USERNAME/pd-cross-attention"
echo "=============================================="
