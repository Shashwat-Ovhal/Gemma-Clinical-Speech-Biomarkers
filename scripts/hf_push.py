"""
hf_push.py — Push trained model weights to Hugging Face Hub
============================================================
Run this after training completes on vast.ai.

Usage:
    pip install huggingface_hub
    python hf_push.py --token YOUR_HF_TOKEN --repo your-username/pd-cross-attention

Your HF token: https://huggingface.co/settings/tokens
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo


def push_to_hub(token: str, repo_id: str, local_dir: str = "."):
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=token, private=True, exist_ok=True)
        print(f"  ✓ Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"  [WARN] Repo creation: {e}")

    # Files to push
    files_to_push = [
        ("medgemma_pd/models/bspc_model_teacher.pt", "bspc_model_teacher.pt"),
        ("outputs/bspc/ablation_results.json",        "results/ablation_results.json"),
        ("outputs/bspc/edge_profile.json",            "results/edge_profile.json"),
        ("outputs/bspc/phase7_kcl_results.json",      "results/phase7_kcl_results.json"),
        ("bspc_pipeline/cross_attention_fusion.py",   "model_card/cross_attention_fusion.py"),
    ]

    for local_path, hf_path in files_to_push:
        full_local = os.path.join(local_dir, local_path)
        if not os.path.exists(full_local):
            print(f"  [SKIP] Not found: {full_local}")
            continue
        api.upload_file(
            path_or_fileobj=full_local,
            path_in_repo=hf_path,
            repo_id=repo_id,
            token=token,
        )
        print(f"  ✓ Uploaded: {hf_path}")

    print(f"\n  All artifacts pushed → https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token",  required=True, help="HF access token")
    parser.add_argument("--repo",   required=True, help="e.g. your-username/pd-cross-attention")
    parser.add_argument("--dir",    default=".",   help="Project root directory")
    args = parser.parse_args()
    push_to_hub(args.token, args.repo, args.dir)
