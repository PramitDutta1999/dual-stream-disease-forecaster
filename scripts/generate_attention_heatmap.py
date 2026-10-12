"""Standalone Transformer Attention Explainability Heatmap Generator

Extracts final-layer attention weights from the fine-tuned DualStreamEHRTransformer,
isolates the top-predicted target condition for a held-out test patient,
extracts and decodes the Top 10 attended clinical/genomic tokens,
and renders a clean heatmap visualization saved to reports/heatmaps/.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Prevent scripts/tokenize.py from shadowing Python standard library tokenize
_scripts_dir = sys.path[0] if sys.path else ""
if sys.path and ("scripts" in sys.path[0] or sys.path[0].replace("\\", "/").rstrip("/").endswith("/scripts")):
    del sys.path[0]
import tokenize as _stdlib_tokenize

if _scripts_dir and _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import argparse
import json
import logging
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from ehr_dataset import get_dataloaders
from generative_models import DualStreamEHRTransformer, EHRModelConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("explainability")


def get_project_root() -> Path:
    """Returns project root directory assuming this script is in scripts/."""
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "scripts":
        return script_dir.parent
    return Path.cwd()


def load_code_descriptions(project_root: Path) -> Dict[str, str]:
    """Builds a comprehensive dictionary mapping clinical codes to medical descriptions."""
    code_to_desc: Dict[str, str] = {}

    # 1. Target conditions
    tc_path = project_root / "data" / "target_conditions.csv"
    if not tc_path.exists():
        tc_path = project_root / "target_conditions.csv"
    if tc_path.exists():
        df_tc = pd.read_csv(tc_path)
        for _, row in df_tc.iterrows():
            code = str(row["CODE"]).strip()
            desc = str(row["DESCRIPTION"]).strip()
            code_to_desc[code] = desc

    # 2. Extract descriptions from clinical tables across splits
    tables = [
        "conditions.csv",
        "medications.csv",
        "procedures.csv",
        "observations.csv",
        "careplans.csv",
        "allergies.csv",
    ]
    candidate_dirs = [
        project_root / "test",
        project_root / "train_val",
        project_root / "truncated_train_val",
    ]

    for c_dir in candidate_dirs:
        if not c_dir.exists():
            continue
        for t_file in tables:
            fpath = c_dir / t_file
            if not fpath.exists():
                continue
            try:
                sample_cols = pd.read_csv(fpath, nrows=2).columns.tolist()
                code_col = None
                desc_col = None
                for c in ["CODE", "Code", "code"]:
                    if c in sample_cols:
                        code_col = c
                        break
                for d in ["DESCRIPTION", "Description", "desc"]:
                    if d in sample_cols:
                        desc_col = d
                        break
                if code_col and desc_col:
                    chunk = pd.read_csv(fpath, usecols=[code_col, desc_col], nrows=10000, dtype=str)
                    for _, row in chunk.dropna().iterrows():
                        code = str(row[code_col]).strip()
                        desc = str(row[desc_col]).strip()
                        if code not in code_to_desc:
                            code_to_desc[code] = desc
            except Exception:
                continue

    logger.info(f"Loaded {len(code_to_desc):,} medical code descriptions.")
    return code_to_desc


def decode_token_label(
    token_id: int,
    inv_vocab: Dict[int, str],
    code_to_desc: Dict[str, str],
    max_desc_len: int = 24,
) -> str:
    """Decodes a token ID into a human-readable medical description with its clinical code."""
    token_str = inv_vocab.get(token_id, str(token_id))

    if token_str in ("<PAD>", "<UNK>", "<EOS>"):
        return token_str

    # Handle binned observation tokens (e.g. 10230-1_Q2)
    base_code = token_str.split("_")[0] if ("_" in token_str and not token_str.startswith("IMG_")) else token_str
    desc = code_to_desc.get(token_str) or code_to_desc.get(base_code)

    if desc:
        if len(desc) > max_desc_len:
            short_desc = desc[:max_desc_len].rstrip() + ".."
        else:
            short_desc = desc
        if "_" in token_str and not token_str.startswith("IMG_"):
            suffix = token_str.split("_", 1)[1]
            return f"{short_desc} [{suffix}]\n({base_code})"
        return f"{short_desc}\n({token_str})"

    return token_str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate standalone transformer attention explainability heatmap for test set."
    )
    project_root = get_project_root()

    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=project_root / "checkpoints" / "best_model",
        help="Path to best model checkpoint directory",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=project_root / "reports" / "heatmaps" / "test_1.png",
        help="Target output image path (default: reports/heatmaps/test_1.png, overwriting existing)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top attended tokens to display (default: 10)",
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        default=None,
        help="Optional specific patient ID from test set (defaults to highest-confidence test patient)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI resolution for saved plot (default: 300)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # 1. Load best fine-tuned model and config
    config_path = args.checkpoint_dir / "config.json"
    weights_path = args.checkpoint_dir / "pytorch_model.bin"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    config = EHRModelConfig(**cfg_dict)

    model = DualStreamEHRTransformer(config).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Successfully loaded model from {args.checkpoint_dir}")

    # Load vocabulary
    vocab_path = project_root / "data" / "vocab.json"
    if not vocab_path.exists():
        vocab_path = project_root / "vocab.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    inv_vocab = {int(v): str(k) for k, v in vocab.items()}

    # Load Target Conditions
    tc_path = project_root / "data" / "target_conditions.csv"
    if not tc_path.exists():
        tc_path = project_root / "target_conditions.csv"
    tc_df = pd.read_csv(tc_path)
    target_codes = tc_df["CODE"].astype(str).tolist()
    target_names = tc_df["DESCRIPTION"].astype(str).tolist()
    target_token_ids = [vocab.get(code, None) for code in target_codes]

    code_to_desc = load_code_descriptions(project_root)

    # 2. Load test dataset using DataLoader
    logger.info("Loading held-out test cohort DataLoader...")
    _, _, test_loader = get_dataloaders(
        batch_size=16,
        mode="finetune",
        max_len=config.max_seq_len,
        include_test=True,
    )

    # 3. Find target patient and condition
    logger.info("Scanning test set to isolate target patient and condition...")
    selected_patient_data = None
    best_candidate_prob = -1.0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            age_ids = batch["age_ids"].to(device)
            genomic_ids = batch["genomic_ids"].to(device) if "genomic_ids" in batch else None
            attention_mask = batch["attention_mask"].to(device)

            out = model(
                input_ids=input_ids,
                age_ids=age_ids,
                attention_mask=attention_mask,
                genomic_ids=genomic_ids,
                output_attentions=True,
            )

            cls_logits = out["cls_logits"]
            # Build pre-existing condition mask
            batch_size, num_targets = input_ids.size(0), len(target_token_ids)
            mask = torch.ones((batch_size, num_targets), dtype=torch.float32, device=device)
            for j, tid in enumerate(target_token_ids):
                if tid is not None:
                    has_tok = (input_ids == tid).any(dim=1)
                    mask[has_tok, j] = 0.0

            probs = torch.sigmoid(cls_logits) * mask  # [B, 40]
            attentions = out["attentions"]  # List of [B, num_heads, L, L]
            final_layer_attn = attentions[-1]  # [B, num_heads, L, L]

            for i, pid in enumerate(batch["patient_ids"]):
                p_probs = probs[i].cpu().numpy()
                p_best_idx = int(p_probs.argmax())
                p_best_prob = float(p_probs[p_best_idx])

                # Check if this matches user requested patient or best candidate
                is_match = (args.patient_id is not None and pid == args.patient_id) or (
                    args.patient_id is None and p_best_prob > best_candidate_prob
                )

                if is_match:
                    best_candidate_prob = p_best_prob
                    valid_len = int(attention_mask[i].sum().item())
                    last_token_idx = valid_len - 1

                    # Attention from classification token (last valid token) to all input sequence tokens
                    # Shape: [num_heads, valid_len]
                    heads_attn = final_layer_attn[i, :, last_token_idx, :valid_len].cpu().numpy()
                    mean_attn = heads_attn.mean(axis=0)  # [valid_len]

                    selected_patient_data = {
                        "patient_id": str(pid),
                        "condition_idx": p_best_idx,
                        "condition_code": target_codes[p_best_idx],
                        "condition_name": target_names[p_best_idx],
                        "probability": p_best_prob,
                        "input_ids": input_ids[i, :valid_len].cpu().tolist(),
                        "heads_attention": heads_attn,
                        "token_importance": mean_attn,
                        "valid_len": valid_len,
                    }

                    if args.patient_id is not None and pid == args.patient_id:
                        break
            if args.patient_id is not None and selected_patient_data is not None:
                break

    if selected_patient_data is None:
        raise RuntimeError("Failed to find suitable test patient for attention visualization.")

    p_info = selected_patient_data
    pid = p_info["patient_id"]
    cond_name = p_info["condition_name"]
    cond_code = p_info["condition_code"]
    prob = p_info["probability"]
    logger.info(
        f"Selected Test Patient: {pid} | Isolated Condition: {cond_name} ({cond_code}) with Predicted Probability: {prob:.2%}"
    )

    # 4. Filter sequence to select Top 10 tokens with highest attention weights
    token_importance = p_info["token_importance"]
    heads_attention = p_info["heads_attention"]
    seq_input_ids = p_info["input_ids"]
    valid_len = len(seq_input_ids)

    k = min(args.top_k, valid_len)
    top_indices = np.argsort(token_importance)[::-1][:k]

    top_importance = token_importance[top_indices]
    top_heads = heads_attention[:, top_indices]
    top_tokens = [seq_input_ids[idx] for idx in top_indices]

    # Decode tokens
    decoded_labels = []
    print("\n" + "=" * 80)
    print(f"TOP {k} ATTENDED TOKENS FOR PATIENT {pid}")
    print(f"Predicted Condition: {cond_name} (Code: {cond_code}, Prob: {prob:.2%})")
    print("=" * 80)
    print(f"{'Rank':<6} | {'Token ID':<10} | {'Mean Attention':<16} | {'Decoded Event Label':<40}")
    print("-" * 80)

    for rank, (tid, score) in enumerate(zip(top_tokens, top_importance), start=1):
        label = decode_token_label(tid, inv_vocab, code_to_desc, max_desc_len=26)
        clean_label = label.replace("\n", " ")
        print(f"#{rank:<5} | {tid:<10} | {score:>14.4f} | {clean_label}")
        decoded_labels.append(f"#{rank}: {label}")
    print("=" * 80 + "\n")

    # 5. Generate clean heatmap plot mapping Top 10 tokens against attention scores
    mean_row = top_importance.reshape(1, -1)
    matrix = np.vstack([mean_row, top_heads])
    y_labels = ["Mean Importance", "Head 1", "Head 2", "Head 3", "Head 4"]

    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=args.dpi)

    sns.heatmap(
        matrix,
        cmap="magma",
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 9.5, "fontweight": "bold"},
        cbar_kws={"label": "Attention Weight", "shrink": 0.82, "pad": 0.02},
        xticklabels=decoded_labels,
        yticklabels=y_labels,
        ax=ax,
        linewidths=1.0,
        linecolor="#E2E8F0",
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=35,
        ha="right",
        fontsize=10.0,
        fontweight="bold",
        color="#0F172A",
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color="#1E293B",
    )
    ax.set_xlabel(
        f"Top {k} Attended Tokens in Longitudinal Timeline (Ranked by Mean Attention)",
        fontsize=12,
        fontweight="bold",
        labelpad=14,
        color="#0F172A",
    )
    ax.set_title(
        f"Patient ID: {pid}\nPredicted Condition: {cond_name} (Confidence: {prob:.1%})  |  Top {k} Attention Drivers",
        fontsize=13.0,
        fontweight="bold",
        pad=16,
        color="#0F172A",
    )

    plt.tight_layout()

    # 6. Save visualization directly into reports/heatmaps/ overwriting target file
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_file, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Successfully saved and overwritten explainability heatmap at: {args.output_file}")
    print(f"[SUCCESS] Heatmap saved to: {args.output_file}")


if __name__ == "__main__":
    main()

