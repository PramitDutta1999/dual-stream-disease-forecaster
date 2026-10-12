"""Generate Explainability Attention Heatmaps & Prediction vs Ground Truth Report

Generates Top-10 token attention heatmaps for 9 patients (3 Train, 3 Validation, 3 Test)
using the fine-tuned DualStreamEHRTransformer model, and compiles a comprehensive
prediction vs ground truth verification report for the Train and Validation cohorts.
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
logger = logging.getLogger("explainability_heatmaps")


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
                    chunk = pd.read_csv(fpath, usecols=[code_col, desc_col], nrows=15000, dtype=str)
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


def plot_top10_heatmap(
    patient_info: Dict[str, Any],
    inv_vocab: Dict[int, str],
    code_to_desc: Dict[str, str],
    output_path: Path,
    dpi: int = 300,
) -> None:
    """Renders and saves a clean, legible heatmap mapping the Top-10 attended tokens."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_ids = patient_info["input_ids"]
    token_importance = patient_info["token_importance"]
    heads_attention = patient_info["heads_attention"]
    valid_len = len(input_ids)

    k = min(10, valid_len)
    top_indices = np.argsort(token_importance)[::-1][:k]

    top_importance = token_importance[top_indices]
    top_heads = heads_attention[:, top_indices]
    top_tokens = [input_ids[idx] for idx in top_indices]

    decoded_labels = []
    for rank, tid in enumerate(top_tokens, start=1):
        raw_label = decode_token_label(tid, inv_vocab, code_to_desc, max_desc_len=24)
        decoded_labels.append(f"#{rank}: {raw_label}")

    mean_row = top_importance.reshape(1, -1)
    matrix = np.vstack([mean_row, top_heads])
    y_labels = ["Mean Importance", "Head 1", "Head 2", "Head 3", "Head 4"]

    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=dpi)

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
        f"Top {k} Attended Tokens in Patient History (Ranked by Attention Score)",
        fontsize=12,
        fontweight="bold",
        labelpad=14,
        color="#0F172A",
    )

    pid = patient_info["patient_id"]
    cond = patient_info["top_condition_name"]
    code = patient_info["top_condition_code"]
    prob = patient_info["top_condition_prob"]
    split = patient_info["split"].upper()

    ax.set_title(
        f"[{split}] Patient ID: {pid}\nPredicted Condition: {cond} ({code})  |  Confidence: {prob:.1%}  |  Top {k} Attention Drivers",
        fontsize=13.0,
        fontweight="bold",
        pad=16,
        color="#0F172A",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved explainability heatmap to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate explainability attention heatmaps for 9 patients across Train, Val, and Test cohorts."
    )
    project_root = get_project_root()

    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=project_root / "checkpoints" / "best_model",
        help="Path to best model checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root / "reports" / "heatmaps",
        help="Directory to save heatmaps and report (default: reports/heatmaps)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution DPI for heatmaps (default: 300)",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # 1. Load best model and configuration
    config_path = args.checkpoint_dir / "config.json"
    weights_path = args.checkpoint_dir / "pytorch_model.bin"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    config = EHRModelConfig(**cfg_dict)

    model = DualStreamEHRTransformer(config).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded trained DualStreamEHRTransformer from: {args.checkpoint_dir}")

    # Load vocab
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

    # 2. Initialize DataLoaders for Train, Val, and Test
    logger.info("Initializing DataLoaders across Train, Validation, and Test cohorts...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=16,
        mode="finetune",
        max_len=config.max_seq_len,
        include_test=True,
    )

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    selected_patients_by_split: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    # 3. Process each split and extract 3 patients per split
    for split_name in ["train", "val", "test"]:
        loader = dataloaders[split_name]
        logger.info(f"\nProcessing '{split_name}' cohort to select 3 distinct patients...")

        found_patients = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                age_ids = batch["age_ids"].to(device)
                genomic_ids = batch["genomic_ids"].to(device) if "genomic_ids" in batch else None
                attention_mask = batch["attention_mask"].to(device)
                targets = batch.get("targets")

                out = model(
                    input_ids=input_ids,
                    age_ids=age_ids,
                    attention_mask=attention_mask,
                    genomic_ids=genomic_ids,
                    output_attentions=True,
                )

                cls_logits = out["cls_logits"]
                # Dynamic pre-existing mask
                batch_size = input_ids.size(0)
                mask = torch.ones((batch_size, len(target_token_ids)), dtype=torch.float32, device=device)
                for j, tid in enumerate(target_token_ids):
                    if tid is not None:
                        has_tok = (input_ids == tid).any(dim=1)
                        mask[has_tok, j] = 0.0

                probs = torch.sigmoid(cls_logits) * mask  # [B, 40]
                final_layer_attns = out["attentions"][-1]  # [B, num_heads, L, L]

                for i, pid in enumerate(batch["patient_ids"]):
                    p_probs = probs[i].cpu().numpy()
                    # Sort conditions by predicted probability descending
                    ranked_cond_indices = np.argsort(p_probs)[::-1]
                    top_cond_idx = int(ranked_cond_indices[0])
                    top_cond_prob = float(p_probs[top_cond_idx])

                    # Require reasonable confidence or take available
                    if top_cond_prob > 0.0:
                        valid_len = int(attention_mask[i].sum().item())
                        last_token_idx = valid_len - 1

                        # Extract attention weights for classification token
                        heads_attn = final_layer_attns[i, :, last_token_idx, :valid_len].cpu().numpy()
                        token_importance = heads_attn.mean(axis=0)

                        top_3_indices = [int(idx) for idx in ranked_cond_indices[:3]]
                        top_3_info = []
                        for c_idx in top_3_indices:
                            c_code = target_codes[c_idx]
                            c_name = target_names[c_idx]
                            c_prob = float(p_probs[c_idx])
                            gt_val = int(targets[i, c_idx].item()) if (targets is not None and targets.ndim == 2) else None
                            top_3_info.append({
                                "code": c_code,
                                "name": c_name,
                                "prob": c_prob,
                                "ground_truth": gt_val,
                            })

                        found_patients.append({
                            "patient_id": str(pid),
                            "split": split_name,
                            "top_condition_code": target_codes[top_cond_idx],
                            "top_condition_name": target_names[top_cond_idx],
                            "top_condition_prob": top_cond_prob,
                            "top_3_conditions": top_3_info,
                            "input_ids": input_ids[i, :valid_len].cpu().tolist(),
                            "heads_attention": heads_attn,
                            "token_importance": token_importance,
                            "valid_len": valid_len,
                        })

                        if len(found_patients) >= 3:
                            break
                if len(found_patients) >= 3:
                    break

        selected_patients_by_split[split_name] = found_patients
        logger.info(f"Selected {len(found_patients)} patients for '{split_name}' cohort.")

    # 4. Generate 9 Heatmap Plots
    print("\n" + "=" * 90)
    print("                     GENERATING 9 ATTENTION HEATMAP PLOTS                   ")
    print("=" * 90)

    for split_name, p_list in selected_patients_by_split.items():
        for idx, p_info in enumerate(p_list, start=1):
            filename = f"{split_name}_{idx}.png"
            file_path = args.output_dir / filename
            plot_top10_heatmap(
                patient_info=p_info,
                inv_vocab=inv_vocab,
                code_to_desc=code_to_desc,
                output_path=file_path,
                dpi=args.dpi,
            )
            print(f"[{split_name.upper()} #{idx}] Saved: {filename} (Patient: {p_info['patient_id']} | Condition: {p_info['top_condition_name']} | Prob: {p_info['top_condition_prob']:.1%})")

    # 5. Generate patient_prediction_report.txt (Train & Val cohorts ONLY)
    report_path = args.output_dir / "patient_prediction_report.txt"
    logger.info(f"\nWriting Prediction vs Ground Truth report to: {report_path}...")

    report_lines = []
    report_lines.append("=" * 90)
    report_lines.append("        PATIENT PREDICTION VS. GROUND TRUTH VERIFICATION REPORT        ")
    report_lines.append("=" * 90)
    report_lines.append("Evaluation Cohorts : Train (3 patients) and Validation (3 patients)")
    report_lines.append("Note               : Test set ground truth outcomes are withheld by design.")
    report_lines.append("Model Architecture : DualStreamEHRTransformer (Clinical + Genomic NanoGPT)")
    report_lines.append("-" * 90)

    for split_name in ["train", "val"]:
        report_lines.append(f"\n[{split_name.upper()} COHORT]")
        report_lines.append("-" * 90)

        for idx, p_info in enumerate(selected_patients_by_split[split_name], start=1):
            pid = p_info["patient_id"]
            seq_len = p_info["valid_len"]
            report_lines.append(f"Patient #{idx} (ID: {pid}) | Clinical Sequence Length: {seq_len} events")
            report_lines.append("  Top 3 Predicted Incident Conditions:")

            for rank, c_info in enumerate(p_info["top_3_conditions"], start=1):
                cond_name = c_info["name"]
                cond_code = c_info["code"]
                prob_pct = f"{c_info['prob']:.2%}"
                gt_label = c_info["ground_truth"]
                gt_str = f"Outcome: {gt_label}" if gt_label is not None else "Outcome: Withheld"
                match_status = " [MATCH - True Positive]" if gt_label == 1 else " [Non-Incident / True Negative]" if gt_label == 0 else ""
                report_lines.append(
                    f"    {rank}. {cond_name} (Code: {cond_code}) -> Predicted Probability: {prob_pct} | Ground Truth {gt_str}{match_status}"
                )
            report_lines.append("")

    report_lines.append("=" * 90)
    report_lines.append("End of Verification Report.")
    report_lines.append("=" * 90)

    report_content = "\n".join(report_lines)
    report_path.write_text(report_content, encoding="utf-8")
    logger.info(f"Successfully saved prediction report to: {report_path}")

    print("\n" + "=" * 90)
    print("                    PREDICTION VS. GROUND TRUTH REPORT                     ")
    print("=" * 90)
    print(report_content)
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()

