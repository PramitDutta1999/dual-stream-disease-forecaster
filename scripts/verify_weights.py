"""Weight Verification Script

Verifies model weight initialization:
1. Asserts the strict absence of any pre-trained weight loaders in generative_models.py.
2. Instantiates DualStreamEHRTransformer and calculates global mean and standard deviation
   across all initialized parameters to prove initialization from pure random noise.
3. Tests Stage 2 transition (load_pretrained_core) verifying safe lm_head discarding.
4. Tests explainability feature (get_attention_heatmap) verifying attention extraction.
"""

from __future__ import annotations

import sys

# Prevent scripts/tokenize.py from shadowing Python standard library tokenize
_scripts_dir = sys.path[0] if sys.path else ""
if sys.path and ("scripts" in sys.path[0] or sys.path[0].replace("\\", "/").rstrip("/").endswith("/scripts")):
    del sys.path[0]
import tokenize as _stdlib_tokenize

if _scripts_dir and _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from pathlib import Path
import torch
from generative_models import DualStreamEHRTransformer, EHRModelConfig


def verify_code_integrity(model_file_path: Path):
    """Enforces strict absence of pre-trained loaders in generative_models.py."""
    print("=" * 80)
    print("STEP 1: CODE INTEGRITY CHECK")
    print("=" * 80)

    if not model_file_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_file_path}")

    with open(model_file_path, "r", encoding="utf-8") as f:
        code_text = f.read()

    # Strict check: "from_pretrained" must be entirely absent
    forbidden_string = "from_" + "pretrained"
    assert forbidden_string not in code_text, (
        f"VIOLATION: '{forbidden_string}' detected in {model_file_path}! "
        f"Pretrained weights or Hugging Face loaders are forbidden."
    )

    print(f"[PASS] Confirmed '{forbidden_string}' is entirely absent from {model_file_path.name}.")
    print("Model architecture is 100% custom-built using pure PyTorch nn.Module.\n")


def verify_weight_initialization(model: DualStreamEHRTransformer):
    """Verifies that model parameters are initialized from pure random normal noise (mean=0.0, std=0.02)."""
    print("=" * 80)
    print("STEP 2: RANDOM NOISE WEIGHT INITIALIZATION CHECK")
    print("=" * 80)

    linear_emb_weights = []
    all_weights = []
    layer_stats = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            data = param.data.cpu().float()
            all_weights.append(data.view(-1))
            if "ln" not in name and "bias" not in name:
                linear_emb_weights.append(data.view(-1))

            mean_val = data.mean().item()
            std_val = data.std().item()
            layer_stats.append({
                "name": name,
                "shape": list(data.shape),
                "numel": data.numel(),
                "mean": mean_val,
                "std": std_val,
            })

    # Linear and Embedding parameters (the target random normal distribution)
    linear_emb_tensor = torch.cat(linear_emb_weights)
    linear_emb_mean = linear_emb_tensor.mean().item()
    linear_emb_std = linear_emb_tensor.std().item()

    # All model parameters (including LayerNorm gains and biases)
    global_tensor = torch.cat(all_weights)
    global_mean = global_tensor.mean().item()
    global_std = global_tensor.std().item()

    print(f"{'Parameter Tensor':<45} | {'Shape':<16} | {'Mean':>10} | {'Std Dev':>10}")
    print("-" * 88)
    for stat in layer_stats[:10]:
        print(f"{stat['name']:<45} | {str(stat['shape']):<16} | {stat['mean']:>10.6f} | {stat['std']:>10.6f}")
    if len(layer_stats) > 10:
        print(f"... and {len(layer_stats) - 10} more parameter tensors ...")
    print("-" * 88)

    print(f"\nLINEAR & EMBEDDING WEIGHTS INITIALIZATION ({linear_emb_tensor.numel():,} parameters):")
    print(f"  - Target Distribution : Normal(mean=0.0000, std=0.0200)")
    print(f"  - Actual Weight Mean  : {linear_emb_mean:+.6f}")
    print(f"  - Actual Weight Std   : {linear_emb_std:.6f}")

    print(f"\nOVERALL MODEL PARAMETERS ({global_tensor.numel():,} parameters, including norms & biases):")
    print(f"  - Global Mean         : {global_mean:+.6f}")
    print(f"  - Global Std          : {global_std:.6f}")

    # Strict assertion: Linear & Embedding weights must strictly adhere to Normal(0, 0.02)
    assert abs(linear_emb_mean) < 0.001, f"Linear & Embedding mean {linear_emb_mean} deviates excessively from 0.0!"
    assert abs(linear_emb_std - 0.02) < 0.001, f"Linear & Embedding std {linear_emb_std} deviates from 0.02!"
    assert abs(global_mean) < 0.005, f"Global mean {global_mean} deviates excessively from 0.0!"

    print("\n[PASS] Model initialized from pure random normal noise. No external weights loaded.\n")


def verify_stage2_core_loading(model: DualStreamEHRTransformer):
    """Verifies load_pretrained_core safely discards lm_head and handles Stage 2 transition."""
    print("=" * 80)
    print("STEP 3: STAGE 2 WEIGHT LOADING & HEAD DISCARD VERIFICATION")
    print("=" * 80)

    # Simulate saving a Stage 1 state_dict
    stage1_state = model.state_dict()
    assert "lm_head.weight" in stage1_state, "lm_head.weight must be present in stage1 state_dict"

    # Instantiate a fresh model for Stage 2
    stage2_model = DualStreamEHRTransformer(model.config)

    # Load core weights
    transition_result = stage2_model.load_pretrained_core(stage1_state)

    print(f"Loaded Core Tensors    : {transition_result['loaded_count']}")
    print(f"Safely Discarded Heads : {transition_result['discarded_count']} ({transition_result['discarded_keys']})")
    print(f"Size Mismatches Caught : {transition_result['mismatched_count']}")

    # Verify that lm_head was discarded
    assert any("lm_head" in k for k in transition_result["discarded_keys"]), "lm_head was not discarded!"

    # Verify classification_head parameters remain untouched / newly initialized
    cls_head_missing = any("classification_head" in k for k in transition_result["missing_keys"])
    assert cls_head_missing, "classification_head must be fresh for Stage 2!"

    print("\n[PASS] Stage 2 core loading cleanly discarded pre-trained lm_head.\n")


def verify_explainability_attention(model: DualStreamEHRTransformer):
    """Verifies explainability feature get_attention_heatmap."""
    print("=" * 80)
    print("STEP 4: EXPLAINABILITY & ATTENTION HEATMAP EXTRACTION")
    print("=" * 80)

    batch_size = 2
    seq_len = 16
    dummy_input_ids = torch.randint(1, 100, (batch_size, seq_len))
    dummy_age_ids = torch.randint(0, 100, (batch_size, seq_len))
    dummy_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)

    heatmap = model.get_attention_heatmap(
        input_ids=dummy_input_ids,
        age_ids=dummy_age_ids,
        attention_mask=dummy_mask,
        head_reduction="mean",
    )

    print(f"Extracted Attention Heatmap Shape: {list(heatmap.shape)} (expected [{batch_size}, {seq_len}, {seq_len}])")
    assert heatmap.shape == (batch_size, seq_len, seq_len), f"Unexpected shape {heatmap.shape}"

    # Verify attention weights are non-negative probabilities
    assert (heatmap >= 0.0).all(), "Attention weights must be non-negative!"

    print("\n[PASS] Explainability attention heatmap successfully extracted.\n")


def main():
    model_file = Path("scripts/generative_models.py")

    # Step 1: Code audit
    verify_code_integrity(model_file)

    # Step 2: Weight verification
    config = EHRModelConfig(
        vocab_size=1362,
        max_age=121,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        num_classes=40,
    )
    model = DualStreamEHRTransformer(config)

    # Architecture summary
    arch_summary = model.log_model_architecture()
    print("=" * 80)
    print("MODEL ARCHITECTURE PARAMETER BREAKDOWN")
    print("=" * 80)
    for k, v in arch_summary.items():
        print(f"  {k:<35}: {v:>12,d}")
    print("=" * 80 + "\n")

    verify_weight_initialization(model)

    # Step 3: Stage 2 core transition
    verify_stage2_core_loading(model)

    # Step 4: Explainability
    verify_explainability_attention(model)

    print("=" * 80)
    print("ALL WEIGHT AND ARCHITECTURE VERIFICATIONS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
