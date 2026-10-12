"""Transformer Explainability & Attention Heatmap Generation Script

Extracts final-layer attention weights from the trained DualStreamEHRTransformer,
computes token-level importance scores averaged across attention heads,
decodes chronological clinical sequences into readable medical text,
and generates publication-ready heatmaps for 3 Train, 3 Validation, and 3 Test cases.
"""

from __future__ import annotations

import sys

# Prevent scripts/tokenize.py from shadowing Python's standard library tokenize
_scripts_dir = sys.path[0] if sys.path else ""
if sys.path and ("scripts" in sys.path[0] or sys.path[0].replace("\\", "/").rstrip("/").endswith("/scripts")):
    del sys.path[0]
import tokenize as _stdlib_tokenize

if _scripts_dir and _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from ehr_dataset import EHRDataset, ehr_collate_fn
from generative_models import DualStreamEHRTransformer, EHRModelConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("explainability")


def get_default_project_root() -> Path:
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
        try:
            tc_df = pd.read_csv(tc_path)
            for _, r in tc_df.iterrows():
                code_to_desc[str(r["CODE"]).strip()] = str(r["DESCRIPTION"]).strip()
        except Exception as e:
            logger.warning(f"Failed loading target conditions descriptions: {e}")

    # 2. Historical clinical tables
    search_dirs = [project_root / "train_val", project_root / "test"]
    tables = [
        "conditions.csv",
        "medications.csv",
        "procedures.csv",
        "observations.csv",
        "encounters.csv",
        "careplans.csv",
        "allergies.csv",
        "devices.csv",
    ]

    for d in search_dirs:
        for fname in tables:
            fpath = d / fname
            if not fpath.exists():
                continue
            try:
                # Read CODE and DESCRIPTION columns
                df = pd.read_csv(fpath, usecols=["CODE", "DESCRIPTION"], nrows=10000)
                df = df.dropna(subset=["CODE", "DESCRIPTION"])
                for _, r in df.iterrows():
                    c = str(r["CODE"]).strip()
                    if c not in code_to_desc:
                        code_to_desc[c] = str(r["DESCRIPTION"]).strip()
            except Exception:
                continue

    logger.info(f"Loaded {len(code_to_desc):,} medical code descriptions into lookup table.")
    return code_to_desc


def build_pre_existing_mask(
    input_ids: torch.Tensor,
    target_token_ids: List[int | None],
) -> torch.Tensor:
    """Constructs dynamic pre-existing diagnosis mask [batch_size, 40].

    1.0 = incident / at-risk (condition token not in patient history)
    0.0 = pre-existing (condition token found anywhere in patient history)
    """
    batch_size = input_ids.size(0)
    num_targets = len(target_token_ids)
    mask = torch.ones((batch_size, num_targets), dtype=torch.float32, device=input_ids.device)
    for j, tid in enumerate(target_token_ids):
        if tid is not None:
            has_token = (input_ids == tid).any(dim=1)
            mask[has_token, j] = 0.0
    return mask


def load_model_and_config(
    checkpoint_dir: Path,
    device: torch.device,
) -> Tuple[DualStreamEHRTransformer, EHRModelConfig]:
    """Loads trained DualStreamEHRTransformer from the checkpoint directory."""
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "pytorch_model.bin"

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
    logger.info(f"Successfully loaded model checkpoint from: {weights_path}")
    return model, config


def get_deterministic_dataloaders(
    data_path: Path,
    splits_path: Path,
    batch_size: int = 16,
    max_len: int = 256,
) -> Dict[str, DataLoader]:
    """Constructs deterministic (shuffle=False) DataLoaders for train, val, and test splits."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")

    splits_df = pd.read_csv(splits_path)
    split_map = dict(zip(splits_df["Id"], splits_df["split"]))

    raw_data = torch.load(data_path)
    all_records = []
    if isinstance(raw_data, dict):
        for split_key in ["train", "val", "test"]:
            if split_key in raw_data and isinstance(raw_data[split_key], list):
                all_records.extend(raw_data[split_key])
            elif split_key in raw_data and isinstance(raw_data[split_key], dict):
                all_records.extend(raw_data[split_key].values())
        if not all_records and "data" in raw_data:
            for split_key in ["train", "val", "test"]:
                all_records.extend(raw_data["data"].get(split_key, []))

    split_records: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for rec in all_records:
        pid = rec["patient_id"]
        assigned = split_map.get(pid)
        if assigned in split_records:
            split_records[assigned].append(rec)

    loaders = {}
    for s in ["train", "val", "test"]:
        dataset = EHRDataset(split_records[s], mode="finetune", max_len=max_len)
        loaders[s] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # strictly deterministic
            collate_fn=ehr_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
        logger.info(f"Created deterministic DataLoader for '{s}' split with {len(dataset):,} patients.")

    return loaders


def decode_token_label(
    token_id: int,
    inv_vocab: Dict[int, str],
    code_to_desc: Dict[str, str],
    max_desc_len: int = 22,
) -> str:
    """Decodes a token ID into a clean medical description with clinical code."""
    token_str = inv_vocab.get(token_id, str(token_id))

    if token_str in ("<PAD>", "<UNK>", "<EOS>"):
        return token_str

    # Check if observation quintile (e.g., 10230-1_Q2)
    base_code = token_str.split("_")[0] if "_" in token_str and not token_str.startswith("IMG_") else token_str
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


def select_confident_patients(
    model: DualStreamEHRTransformer,
    dataloader: DataLoader,
    target_codes: List[str],
    target_token_ids: List[int | None],
    code_to_desc: Dict[str, str],
    device: torch.device,
    target_count: int = 3,
    confidence_threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    """Identifies the first target_count patients with confident incident condition predictions."""
    selected: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            age_ids = batch["age_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            out = model(
                input_ids=input_ids,
                age_ids=age_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

            cls_logits = out["cls_logits"]
            pre_mask = build_pre_existing_mask(input_ids, target_token_ids)
            probs = torch.sigmoid(cls_logits) * pre_mask  # zero out pre-existing conditions

            final_layer_attns = out["attentions"][-1]  # [B, num_heads, seq_len, seq_len]

            for i, pid in enumerate(batch["patient_ids"]):
                patient_probs = probs[i].cpu().numpy()
                best_idx = int(patient_probs.argmax())
                best_prob = float(patient_probs[best_idx])

                if best_prob >= confidence_threshold:
                    valid_len = int(attention_mask[i].sum().item())
                    last_token_idx = valid_len - 1

                    # Extract attention weights for classification decision token
                    # [num_heads, valid_len]
                    patient_heads_attn = (
                        final_layer_attns[i, :, last_token_idx, :valid_len].cpu().numpy()
                    )
                    # Average across heads to obtain 1D token importance scores: [valid_len]
                    token_importance = patient_heads_attn.mean(axis=0)

                    cond_code = target_codes[best_idx]
                    cond_name = code_to_desc.get(cond_code, f"Condition {cond_code}")

                    patient_input_ids = input_ids[i, :valid_len].cpu().tolist()

                    selected.append({
                        "patient_id": str(pid),
                        "condition_code": cond_code,
                        "condition_name": cond_name,
                        "confidence": best_prob,
                        "valid_len": valid_len,
                        "input_ids": patient_input_ids,
                        "token_importance": token_importance,
                        "heads_attention": patient_heads_attn,
                    })

                    if len(selected) >= target_count:
                        return selected

    return selected


def plot_patient_heatmap(
    patient_info: Dict[str, Any],
    inv_vocab: Dict[int, str],
    code_to_desc: Dict[str, str],
    output_path: Path,
    include_individual_heads: bool = True,
    top_k: int = 10,
    dpi: int = 300,
) -> None:
    """Generates and saves a clean, interpretable explainability heatmap displaying ONLY the Top-K most attended tokens."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_ids = patient_info["input_ids"]
    token_importance = patient_info["token_importance"]  # [valid_len]
    heads_attention = patient_info["heads_attention"]      # [num_heads, valid_len]
    valid_len = len(input_ids)

    # Top K Selection: sort descending by mean attention score
    k = min(top_k, valid_len)
    top_indices = np.argsort(token_importance)[::-1][:k]

    top_importance = token_importance[top_indices]
    top_heads = heads_attention[:, top_indices]
    top_tids = [input_ids[idx] for idx in top_indices]

    # Decode strictly the Top-K tokens into formatted medical labels with ranks
    decoded_labels = []
    for rank, tid in enumerate(top_tids, start=1):
        raw_desc = decode_token_label(tid, inv_vocab, code_to_desc, max_desc_len=24)
        decoded_labels.append(f"#{rank}: {raw_desc}")

    # Prepare matrix
    if include_individual_heads:
        mean_row = top_importance.reshape(1, -1)
        matrix = np.vstack([mean_row, top_heads])
        y_labels = [
            "Mean (Importance)",
            "Head 1",
            "Head 2",
            "Head 3",
            "Head 4",
        ]
        fig_height = 6.5
    else:
        matrix = top_importance.reshape(1, -1)
        y_labels = ["Mean (Importance)"]
        fig_height = 4.0

    # Fixed comfortable figure width for 10 tokens ensuring large legible text
    fig_width = 14.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    # Plot seaborn heatmap with annotated numerical attention values
    sns.heatmap(
        matrix,
        cmap="magma",
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 9.5, "fontweight": "bold"},
        cbar_kws={
            "label": "Attention Weight",
            "shrink": 0.82,
            "pad": 0.02,
        },
        xticklabels=decoded_labels,
        yticklabels=y_labels,
        ax=ax,
        linewidths=1.0,
        linecolor="#E2E8F0",
    )

    # High-legibility styling for ticks and labels
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=35,
        ha="right",
        fontsize=10.5,
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
        f"Top {k} Highly Attended Clinical Tokens (Ranked by Mean Attention)",
        fontsize=12,
        fontweight="bold",
        labelpad=14,
        color="#0F172A",
    )

    pid = patient_info["patient_id"]
    cond = patient_info["condition_name"]
    prob = patient_info["confidence"]
    ax.set_title(
        f"Patient ID: {pid}\nPredicted Incident Condition: {cond} (Confidence: {prob:.1%})  |  Top {k} Attention Drivers",
        fontsize=13.0,
        fontweight="bold",
        pad=16,
        color="#0F172A",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Top-{k} explainability heatmap to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract transformer attention weights and generate explainability heatmaps."
    )
    project_root = get_default_project_root()

    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=project_root / "checkpoints" / "best_model",
        help="Path to best model checkpoint directory",
    )
    def resolve_data_file(filename: str) -> Path:
        p_data = project_root / "data" / filename
        if p_data.exists():
            return p_data
        return project_root / filename

    parser.add_argument(
        "--data_path",
        type=Path,
        default=resolve_data_file("tokenized_data.pt"),
        help="Path to tokenized_data.pt",
    )
    parser.add_argument(
        "--splits_path",
        type=Path,
        default=resolve_data_file("patient_splits.csv"),
        help="Path to patient_splits.csv",
    )
    parser.add_argument(
        "--vocab_path",
        type=Path,
        default=resolve_data_file("vocab.json"),
        help="Path to vocab.json",
    )
    parser.add_argument(
        "--target_conditions_path",
        type=Path,
        default=resolve_data_file("target_conditions.csv"),
        help="Path to target_conditions.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root / "reports" / "heatmaps",
        help="Destination directory for heatmaps (default: reports/heatmaps/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Minimum incident probability threshold for positive cases (default: 0.50)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top most attended tokens to display on heatmap (default: 10)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for saved heatmaps (default: 300)",
    )

    args = parser.parse_args()

    for attr, name in [
        ("data_path", "tokenized_data.pt"),
        ("splits_path", "patient_splits.csv"),
        ("vocab_path", "vocab.json"),
        ("target_conditions_path", "target_conditions.csv"),
    ]:
        val = getattr(args, attr)
        if not val.exists():
            if (project_root / "data" / name).exists():
                setattr(args, attr, project_root / "data" / name)
            elif (project_root / name).exists():
                setattr(args, attr, project_root / name)

    # Step 1: Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # Load Model & Weights
    model, config = load_model_and_config(args.checkpoint_dir, device)

    # Load Vocab
    with open(args.vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    inv_vocab = {int(v): str(k) for k, v in vocab.items()}

    # Load Medical Code Descriptions
    code_to_desc = load_code_descriptions(project_root)

    # Load Target Conditions
    tc_df = pd.read_csv(args.target_conditions_path)
    target_codes = tc_df["CODE"].astype(str).tolist()
    target_token_ids = [vocab.get(code, None) for code in target_codes]

    # DataLoaders
    dataloaders = get_deterministic_dataloaders(
        args.data_path,
        args.splits_path,
        batch_size=16,
        max_len=config.max_seq_len,
    )

    # Step 2 & 3: Extract and generate heatmaps for 3 train, 3 val, 3 test cases
    summary_records = []
    splits = ["train", "val", "test"]

    for split in splits:
        logger.info(f"\n--- Extracting confident patients for split: {split.upper()} ---")
        selected_patients = select_confident_patients(
            model=model,
            dataloader=dataloaders[split],
            target_codes=target_codes,
            target_token_ids=target_token_ids,
            code_to_desc=code_to_desc,
            device=device,
            target_count=3,
            confidence_threshold=args.threshold,
        )

        if len(selected_patients) < 3:
            logger.warning(
                f"Found only {len(selected_patients)} patients with probability >= {args.threshold} in '{split}' split."
            )

        for idx, p_info in enumerate(selected_patients, start=1):
            filename = f"{split}_{idx}.png"
            file_path = args.output_dir / filename

            plot_patient_heatmap(
                patient_info=p_info,
                inv_vocab=inv_vocab,
                code_to_desc=code_to_desc,
                output_path=file_path,
                include_individual_heads=True,
                top_k=args.top_k,
                dpi=args.dpi,
            )

            # Top attended clinical token
            top_token_idx = int(p_info["token_importance"].argmax())
            top_token_id = p_info["input_ids"][top_token_idx]
            top_token_str = decode_token_label(top_token_id, inv_vocab, code_to_desc, max_desc_len=30)
            top_token_clean = top_token_str.replace("\n", " ")

            summary_records.append({
                "Split": split.capitalize(),
                "File": filename,
                "Patient ID": p_info["patient_id"],
                "Predicted Condition": p_info["condition_name"],
                "Confidence": f"{p_info['confidence']:.1%}",
                "Seq Len": p_info["valid_len"],
                "Top Attended Event": top_token_clean,
            })

    # Step 4: Executive Summary
    summary_df = pd.DataFrame(summary_records)
    sep = "=" * 90
    print("\n" + sep)
    print("           TRANSFORMER ATTENTION EXPLAINABILITY HEATMAPS SUMMARY           ")
    print(sep)
    print(summary_df.to_string(index=False))
    print(sep)
    print(f"All 9 heatmaps successfully saved into: {args.output_dir}\n")


if __name__ == "__main__":
    main()

