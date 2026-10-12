"""Comprehensive Training Pipeline for DualStreamEHRTransformer

Includes:
1. Pre-run verification & single-batch overfitting sanity check.
2. Stage 1: Autoregressive Next-Token Pre-Training (CrossEntropyLoss).
3. Stage 1 -> Stage 2 Transition via load_pretrained_core().
4. Stage 2: Fine-Tuning Grid Search (BCEWithLogitsLoss) with wandb tracking.
5. Multi-label evaluation: Macro AUROC, mAP, BCE Loss, and Global Accuracy.
6. Hugging Face Hub ready checkpointing (pytorch_model.bin, config.json, vocab.json).
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

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import os
import shutil
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb

from ehr_dataset import PAD_ID, get_dataloaders
from generative_models import DualStreamEHRTransformer, EHRModelConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def load_secrets(project_root: Path | None = None) -> None:
    """Dynamically loads secrets from secrets.txt into os.environ if not already present."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()
    for candidate in [project_root / "secrets.txt", project_root.parent / "secrets.txt", Path("secrets.txt")]:
        if candidate.exists():
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip()
                        if k and v and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass
            break


# ==============================================================================
# 1. Pre-Run Verification & Dummy Overfitting Sanity Check
# ==============================================================================
def run_sanity_check(
    config: EHRModelConfig,
    device: torch.device,
    dummy_iterations: int = 50,
):
    """Verifies output tensor shapes and trains on a single dummy batch for 50-100 iterations

    to prove that the model can backpropagate gradients and overfit.
    """
    print("\n" + "=" * 80)
    print("PRE-RUN VERIFICATION & DUMMY OVERFITTING SANITY CHECK")
    print("=" * 80)

    model = DualStreamEHRTransformer(config).to(device)
    model.train()

    # Create dummy batch
    batch_size = 4
    seq_len = 32
    dummy_input_ids = torch.randint(3, config.vocab_size, (batch_size, seq_len), device=device)
    dummy_age_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
    dummy_genomic_ids = torch.randint(0, config.genomic_vocab_size, (batch_size, seq_len), device=device)
    dummy_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    dummy_pretrain_targets = torch.randint(3, config.vocab_size, (batch_size, seq_len), device=device)
    dummy_cls_targets = torch.randint(0, 2, (batch_size, config.num_classes), device=device).float()

    # 1. Shape Verification
    with torch.no_grad():
        out = model(
            input_ids=dummy_input_ids,
            age_ids=dummy_age_ids,
            attention_mask=dummy_mask,
            genomic_ids=dummy_genomic_ids,
        )

    expected_lm_shape = (batch_size, seq_len, config.vocab_size)
    expected_cls_shape = (batch_size, config.num_classes)

    assert out["lm_logits"].shape == expected_lm_shape, (
        f"LM logits shape mismatch: expected {expected_lm_shape}, got {out['lm_logits'].shape}"
    )
    assert out["cls_logits"].shape == expected_cls_shape, (
        f"Classification logits shape mismatch: expected {expected_cls_shape}, got {out['cls_logits'].shape}"
    )
    print(f"[PASS] Shape check passed: LM logits {out['lm_logits'].shape}, CLS logits {out['cls_logits'].shape}")

    # 2. Single-Batch Overfitting Run
    print(f"\nRunning single-batch overfitting sanity check for {dummy_iterations} iterations...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    initial_loss = None
    final_loss = None

    for step in range(1, dummy_iterations + 1):
        optimizer.zero_grad()
        out = model(
            input_ids=dummy_input_ids,
            age_ids=dummy_age_ids,
            attention_mask=dummy_mask,
            genomic_ids=dummy_genomic_ids,
        )

        lm_loss = F.cross_entropy(
            out["lm_logits"].view(-1, config.vocab_size),
            dummy_pretrain_targets.view(-1),
            ignore_index=PAD_ID,
        )
        cls_loss = F.binary_cross_entropy_with_logits(
            out["cls_logits"],
            dummy_cls_targets,
        )
        total_loss = lm_loss + cls_loss

        total_loss.backward()
        optimizer.step()

        loss_val = total_loss.item()
        if step == 1:
            initial_loss = loss_val
        final_loss = loss_val

        if step == 1 or step % 10 == 0 or step == dummy_iterations:
            print(f"  Step {step:3d}/{dummy_iterations:3d} | Total Loss: {loss_val:.4f} (LM: {lm_loss.item():.4f}, CLS: {cls_loss.item():.4f})")

    assert final_loss < initial_loss * 0.5, (
        f"Overfitting check failed: final loss {final_loss:.4f} did not drop below 50% of initial loss {initial_loss:.4f}"
    )

    print(f"\n[PASS] Sanity check successful: Loss dropped from {initial_loss:.4f} -> {final_loss:.4f}.")
    print("Model architecture successfully verified. Proceeding to main training.\n" + "=" * 80 + "\n")


# ==============================================================================
# 2. Stage 1: Autoregressive Next-Token Pre-Training
# ==============================================================================
def train_stage1_pretrain(
    config: EHRModelConfig,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    max_len: int = 256,
    max_batches: int | None = None,
    save_path: Path = Path("checkpoints/pretrain_core.pt"),
) -> DualStreamEHRTransformer:
    """Pre-trains DualStreamEHRTransformer using CrossEntropyLoss on next-token prediction."""
    logger.info("Initializing Stage 1: Next-Token Pre-Training...")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl = get_dataloaders(
        batch_size=batch_size,
        mode="pretrain",
        max_len=max_len,
    )

    model = DualStreamEHRTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_tokens = 0
        t0 = time.time()

        for step, batch in enumerate(train_dl):
            if max_batches is not None and step >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            age_ids = batch["age_ids"].to(device)
            genomic_ids = batch["genomic_ids"].to(device) if "genomic_ids" in batch else None
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            out = model(
                input_ids=input_ids,
                age_ids=age_ids,
                attention_mask=attention_mask,
                genomic_ids=genomic_ids,
            )

            loss = F.cross_entropy(
                out["lm_logits"].view(-1, config.vocab_size),
                targets.view(-1),
                ignore_index=PAD_ID,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            non_pad = (targets != PAD_ID).sum().item()
            train_loss_sum += loss.item() * non_pad
            train_tokens += non_pad

        train_loss = train_loss_sum / max(1, train_tokens)

        # Validation evaluation
        model.eval()
        val_loss_sum = 0.0
        val_tokens = 0
        with torch.no_grad():
            for step, batch in enumerate(val_dl):
                if max_batches is not None and step >= max_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                age_ids = batch["age_ids"].to(device)
                genomic_ids = batch["genomic_ids"].to(device) if "genomic_ids" in batch else None
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["targets"].to(device)

                out = model(
                    input_ids=input_ids,
                    age_ids=age_ids,
                    attention_mask=attention_mask,
                    genomic_ids=genomic_ids,
                )
                loss = F.cross_entropy(
                    out["lm_logits"].view(-1, config.vocab_size),
                    targets.view(-1),
                    ignore_index=PAD_ID,
                )
                non_pad = (targets != PAD_ID).sum().item()
                val_loss_sum += loss.item() * non_pad
                val_tokens += non_pad

        val_loss = val_loss_sum / max(1, val_tokens)
        elapsed = time.time() - t0

        # Learning rate scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Stage 1 Epoch {epoch}/{epochs} | LR: {current_lr:.6f} | "
            f"Train CE: {train_loss:.4f} | Val CE: {val_loss:.4f} | Time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best Stage 1 checkpoint to {save_path} (Val CE: {val_loss:.4f})")

    return model


# ==============================================================================
# 3. Dynamic On-The-Fly Pre-Existing Diagnosis Mask & Multi-Label Metrics
# ==============================================================================
DEFAULT_TARGET_CODES = [
    "444814009", "195662009", "10509002", "88805009", "230690007", "22298006",
    "68496003", "44465007", "53741008", "233604007", "43878008", "65966004",
    "126906006", "271737000", "64859006", "443165006", "49436004", "92691004",
    "36971009", "75498004", "162573006", "424132000", "254637007", "39848009",
    "26929004", "263102004", "72892002", "713197008", "62106007", "15777000",
    "58150001", "162864005", "16114001", "70704007", "283371005", "314994000",
    "284551006", "22325002", "723857007", "156073000",
]

DEFAULT_TARGET_TOKEN_IDS = [
    885, 193, 24, 1301, 313, 298, 1090, 884, 929, None, 875, 1065, 54, 408,
    1058, 878, 919, 1317, 778, 1180, 90, 849, 373, 821, 398, 382, 1146, 1109,
    1027, 84, 977, 93, 88, 1106, 423, None, 426, 299, 1134, 81,
]


def load_target_token_ids(
    vocab_path: Path | str = "data/vocab.json",
    targets_path: Path | str = "data/target_conditions.csv",
) -> tuple[list[str], list[int | None]]:
    """Loads target condition codes and token IDs with dynamic file resolution and embedded fallback."""
    target_codes = list(DEFAULT_TARGET_CODES)
    search_target_paths = [
        Path(targets_path),
        Path("data") / Path(targets_path).name,
        Path(__file__).resolve().parent.parent / "data" / "target_conditions.csv",
        Path(__file__).resolve().parent / "data" / "target_conditions.csv",
        Path(__file__).resolve().parent / "target_conditions.csv",
        Path(__file__).resolve().parent.parent / "target_conditions.csv",
    ]
    for p in search_target_paths:
        if p.exists():
            try:
                df = pd.read_csv(p)
                if "CODE" in df.columns:
                    target_codes = df["CODE"].astype(str).tolist()
                    break
            except Exception:
                pass

    token_ids = list(DEFAULT_TARGET_TOKEN_IDS)
    search_vocab_paths = [
        Path(vocab_path),
        Path("data") / Path(vocab_path).name,
        Path(__file__).resolve().parent.parent / "data" / "vocab.json",
        Path(__file__).resolve().parent / "data" / "vocab.json",
        Path(__file__).resolve().parent / "vocab.json",
        Path(__file__).resolve().parent.parent / "vocab.json",
    ]
    for p in search_vocab_paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    vocab = json.load(f)
                token_ids = [vocab.get(code, None) for code in target_codes]
                break
            except Exception:
                pass

    return target_codes, token_ids


def build_pre_existing_mask(
    input_ids: torch.Tensor,
    target_token_ids: list[int | None],
) -> torch.Tensor:
    """Dynamically constructs pre-existing diagnosis mask [batch_size, 40] on the same device.

    1.0 = incident / at-risk (condition token not found in patient history)
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


def evaluate_multilabel(
    model: DualStreamEHRTransformer,
    val_dl: torch.utils.data.DataLoader,
    device: torch.device,
    target_token_ids: list[int | None],
    pos_weight: torch.Tensor | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Computes BCE loss with pre_existing_mask, zeroed Sigmoid probabilities, Macro AUROC, mAP, and Accuracy."""
    model.eval()
    all_targets = []
    all_probs = []
    val_loss_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for step, batch in enumerate(val_dl):
            if max_batches is not None and step >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            age_ids = batch["age_ids"].to(device)
            genomic_ids = batch["genomic_ids"].to(device) if "genomic_ids" in batch else None
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            out = model(
                input_ids=input_ids,
                age_ids=age_ids,
                attention_mask=attention_mask,
                genomic_ids=genomic_ids,
            )
            logits = out["cls_logits"]

            # Dynamic on-the-fly pre-existing diagnosis mask [batch_size, 40]
            mask = build_pre_existing_mask(input_ids, target_token_ids)

            # Apply mask to validation loss calculation
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, weight=mask, pos_weight=pos_weight
            )

            # Method A: Multiply output probabilities by mask (zero out pre-existing conditions)
            probs = torch.sigmoid(logits) * mask

            val_loss_sum += loss.item() * len(targets)
            total_samples += len(targets)

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_probs)

    mean_bce = val_loss_sum / max(1, total_samples)

    # 1. Macro-averaged AUROC
    auc_scores = []
    for col in range(y_true.shape[1]):
        col_true = y_true[:, col]
        col_pred = y_pred[:, col]
        if len(np.unique(col_true)) > 1:
            try:
                score = roc_auc_score(col_true, col_pred)
                auc_scores.append(score)
            except Exception:
                pass
    macro_auroc = float(np.mean(auc_scores)) if auc_scores else 0.5

    # 2. Mean Average Precision (mAP)
    ap_scores = []
    for col in range(y_true.shape[1]):
        col_true = y_true[:, col]
        col_pred = y_pred[:, col]
        if np.sum(col_true) > 0:
            try:
                ap = average_precision_score(col_true, col_pred)
                ap_scores.append(ap)
            except Exception:
                pass
    map_score = float(np.mean(ap_scores)) if ap_scores else 0.0

    # 3. Global Accuracy at 0.5 threshold
    y_bin = (y_pred >= 0.5).astype(int)
    acc = float(accuracy_score(y_true.ravel(), y_bin.ravel()))

    return {
        "val_loss": mean_bce,
        "val_macro_auroc": macro_auroc,
        "val_map": map_score,
        "val_accuracy": acc,
    }


def predict_probabilities(
    model: DualStreamEHRTransformer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_token_ids: list[int | None],
) -> tuple[list[str], np.ndarray]:
    """Generates multi-label probabilities [N, 40] with Method A pre-existing mask zeroing out."""
    model.eval()
    patient_ids = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            age_ids = batch["age_ids"].to(device)
            genomic_ids = batch["genomic_ids"].to(device) if "genomic_ids" in batch else None
            attention_mask = batch["attention_mask"].to(device)

            out = model(
                input_ids=input_ids,
                age_ids=age_ids,
                attention_mask=attention_mask,
                genomic_ids=genomic_ids,
            )

            # Apply Method A: Multiply probabilities by mask to force pre-existing conditions to 0.0
            mask = build_pre_existing_mask(input_ids, target_token_ids)
            probs = torch.sigmoid(out["cls_logits"]) * mask

            patient_ids.extend(batch["patient_ids"])
            all_probs.append(probs.cpu().numpy())

    return patient_ids, np.vstack(all_probs)


# ==============================================================================
# 4. Stage 2: Fine-Tuning Grid Search with wandb
# ==============================================================================
def run_finetune_grid_search(
    config: EHRModelConfig,
    device: torch.device,
    pretrained_checkpoint_path: Path,
    learning_rates: list[float] = [1e-4, 5e-5, 1e-5],
    batch_sizes: list[int] = [16, 32],
    epochs: int = 30,
    patience: int = 5,
    max_len: int = 256,
    max_batches: int | None = None,
    wandb_mode: str = "offline",
    checkpoints_dir: Path = Path("checkpoints/best_model"),
    vocab_source_path: Path = Path("data/vocab.json"),
) -> dict[str, Any]:
    """Performs grid search fine-tuning over hyperparameter combinations, logs to wandb,
    and saves the best model for Hugging Face Hub upload.
    """
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if not vocab_source_path.exists():
        fallback_vocab = Path(__file__).resolve().parent.parent / "data" / "vocab.json"
        if fallback_vocab.exists():
            vocab_source_path = fallback_vocab
    target_codes, target_token_ids = load_target_token_ids(vocab_source_path)
    best_overall_map = -1.0
    best_run_info: dict[str, Any] = {}

    total_runs = len(learning_rates) * len(batch_sizes)
    run_idx = 0

    print("\n" + "=" * 80)
    print("STAGE 2: FINE-TUNING GRID SEARCH")
    print(f"Total Combinations: {total_runs} (LRs: {learning_rates}, Batch Sizes: {batch_sizes})")
    print("=" * 80 + "\n")

    for lr in learning_rates:
        for bs in batch_sizes:
            run_idx += 1
            run_name = f"ehr_ft_lr{lr}_bs{bs}"
            print("-" * 80)
            print(f"[{run_idx}/{total_runs}] STARTING RUN: {run_name} (LR={lr}, BS={bs})")
            print("-" * 80)

            # HPC-friendly wandb initialization (Compute Canada Slurm cluster standard)
            # CRITICAL CLUSTER RULE: wandb.save() is NEVER used; compute nodes block outbound uploads
            try:
                run = wandb.init(
                    project="prediction-patient-history",
                    settings=wandb.Settings(start_method="fork"),
                    name=run_name,
                    mode=wandb_mode,
                    config={
                        "learning_rate": lr,
                        "batch_size": bs,
                        "epochs": epochs,
                        "patience": patience,
                        "max_len": max_len,
                        "model_config": asdict(config),
                        "stage": "finetune",
                    },
                )
            except Exception:
                # Fallback if start_method='fork' is not supported on host OS (e.g. Windows)
                run = wandb.init(
                    project="prediction-patient-history",
                    name=run_name,
                    mode=wandb_mode,
                    config={
                        "learning_rate": lr,
                        "batch_size": bs,
                        "epochs": epochs,
                        "patience": patience,
                        "max_len": max_len,
                        "model_config": asdict(config),
                        "stage": "finetune",
                    },
                )

            # 1. Instantiate fresh model & transition from Stage 1 using load_pretrained_core
            model = DualStreamEHRTransformer(config).to(device)
            if pretrained_checkpoint_path.exists():
                pretrain_weights = torch.load(pretrained_checkpoint_path, map_location=device)
                trans_result = model.load_pretrained_core(pretrain_weights)
                logger.info(
                    f"Stage 2 Transition: Loaded {trans_result['loaded_count']} core tensors. "
                    f"Discarded {trans_result['discarded_count']} head tensors."
                )

            # Log architecture parameter breakdown to wandb
            model.log_model_architecture(run)

            # DataLoaders for this batch size
            train_dl, val_dl = get_dataloaders(
                batch_size=bs,
                mode="finetune",
                max_len=max_len,
            )

            # Address class imbalance: compute pos_weight from training set targets
            train_targets = torch.stack([
                rec["target"] if isinstance(rec["target"], torch.Tensor) else torch.tensor(rec["target"])
                for rec in train_dl.dataset.records
            ])
            num_pos = train_targets.sum(dim=0)
            num_neg = train_targets.size(0) - num_pos
            pos_weight = (num_neg / num_pos.clamp(min=1.0)).float().to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
            )

            best_run_map = -1.0
            epochs_no_improve = 0

            # Train epochs
            for epoch in range(1, epochs + 1):
                model.train()
                train_loss_sum = 0.0
                total_samples = 0
                t0 = time.time()

                for step, batch in enumerate(train_dl):
                    if max_batches is not None and step >= max_batches:
                        break
                    input_ids = batch["input_ids"].to(device)
                    age_ids = batch["age_ids"].to(device)
                    genomic_ids = batch["genomic_ids"].to(device) if "genomic_ids" in batch else None
                    attention_mask = batch["attention_mask"].to(device)
                    targets = batch["targets"].to(device)

                    optimizer.zero_grad()
                    out = model(
                        input_ids=input_ids,
                        age_ids=age_ids,
                        attention_mask=attention_mask,
                        genomic_ids=genomic_ids,
                    )

                    # Dynamic On-The-Fly Pre-Existing Diagnosis Mask [batch_size, 40]
                    mask = build_pre_existing_mask(input_ids, target_token_ids)

                    # Runtime shape verification on the first batch
                    if step == 0 and epoch == 1:
                        print(
                            f"[Verification] Batch 0 pre_existing_mask shape: {tuple(mask.shape)} "
                            f"exactly matches targets shape: {tuple(targets.shape)} (device: {mask.device})"
                        )
                        assert mask.shape == targets.shape, (
                            f"Mask shape {mask.shape} does not match targets shape {targets.shape}"
                        )

                    # Method A: Pass mask as sample/element weights into BCEWithLogits loss
                    loss = F.binary_cross_entropy_with_logits(
                        out["cls_logits"], targets, weight=mask, pos_weight=pos_weight
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_loss_sum += loss.item() * len(targets)
                    total_samples += len(targets)

                train_bce = train_loss_sum / max(1, total_samples)

                # Validation Evaluation with Method A masking
                metrics = evaluate_multilabel(
                    model,
                    val_dl,
                    device,
                    target_token_ids=target_token_ids,
                    pos_weight=pos_weight,
                    max_batches=max_batches,
                )
                elapsed = time.time() - t0

                # Learning rate scheduler step tracking Validation mAP
                scheduler.step(metrics["val_map"])
                current_lr = optimizer.param_groups[0]["lr"]

                # Log metrics to wandb (Accuracy, Train/Val BCE Loss, Val macro AUROC, Val mAP, lr)
                log_data = {
                    "epoch": epoch,
                    "lr": current_lr,
                    "train_loss": train_bce,
                    **metrics,
                }
                if run is not None and hasattr(run, "log"):
                    run.log(log_data)

                print(
                    f"  Epoch {epoch}/{epochs} | LR: {current_lr:.6f} | "
                    f"Train BCE: {train_bce:.4f} | "
                    f"Val BCE: {metrics['val_loss']:.4f} | "
                    f"Val AUROC: {metrics['val_macro_auroc']:.4f} | "
                    f"Val mAP: {metrics['val_map']:.4f} | "
                    f"Acc: {metrics['val_accuracy']:.4f} ({elapsed:.1f}s)"
                )

                # Check if new best model overall
                if metrics["val_map"] > best_overall_map:
                    best_overall_map = metrics["val_map"]
                    best_run_info = {
                        "run_name": run_name,
                        "learning_rate": lr,
                        "batch_size": bs,
                        "best_epoch": epoch,
                        "best_metrics": metrics,
                    }

                    # Save checkpoint ready for Hugging Face Hub strictly to local disk
                    # CRITICAL CLUSTER RULE: wandb.save() is NEVER used
                    torch.save(model.state_dict(), checkpoints_dir / "pytorch_model.bin")
                    with open(checkpoints_dir / "config.json", "w", encoding="utf-8") as f:
                        json.dump(asdict(config), f, indent=2)
                    if vocab_source_path.exists():
                        shutil.copy2(vocab_source_path, checkpoints_dir / "vocab.json")
                    with open(checkpoints_dir / "eval_summary.json", "w", encoding="utf-8") as f:
                        json.dump(best_run_info, f, indent=2)

                    logger.info(
                        f"*** NEW BEST OVERALL MODEL SAVED to {checkpoints_dir} (mAP: {metrics['val_map']:.4f}) ***"
                    )

                # Early Stopping within current run: halt if val mAP does not improve for `patience` epochs
                if metrics["val_map"] > best_run_map:
                    best_run_map = metrics["val_map"]
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        logger.info(
                            f"Early stopping triggered for run {run_name} at epoch {epoch} "
                            f"(Val mAP did not improve for {patience} consecutive epochs. Best run mAP: {best_run_map:.4f})."
                        )
                        break

            if run is not None and hasattr(run, "finish"):
                run.finish()

    return best_run_info


# ==============================================================================
# Main CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="DualStreamEHRTransformer Training & Fine-Tuning Pipeline")
    parser.add_argument("--pretrain_epochs", type=int, default=100, help="Pre-training epochs (default: 100)")
    parser.add_argument("--finetune_epochs", type=int, default=30, help="Fine-tuning epochs (default: 30)")
    parser.add_argument("--learning_rates", nargs="+", type=float, default=[1e-4, 5e-5, 1e-5], help="Grid search LRs (default: [1e-4, 5e-5, 1e-5])")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[16, 32], help="Grid search batch sizes (default: [16, 32])")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience for Stage 2 (default: 5)")
    parser.add_argument("--dummy_iterations", type=int, default=50, help="Sanity check iterations (default: 50)")
    parser.add_argument("--max_len", type=int, default=256, help="Max sequence length (default: 256)")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional max batches per epoch for quick test")
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--checkpoints_dir", type=Path, default=Path("checkpoints/best_model"))
    parser.add_argument("--vocab_path", type=Path, default=Path("data/vocab.json"))

    parser.add_argument("--genomics_dir", type=Path, default=Path("genomics"), help="Directory containing genomics files (default: genomics)")
    parser.add_argument("--export_predictions", action="store_true", default=True, help="Export predictions.csv for test cohort (default: True)")

    args = parser.parse_args()

    # Dynamically inject credentials from secrets.txt if not already in environment
    load_secrets()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # Model configuration
    config = EHRModelConfig(
        vocab_size=1362,
        max_age=121,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        max_seq_len=1024,
        num_classes=40,
    )

    t_start = time.time()

    # Step 1: Pre-Run Verification & Single-Batch Overfitting Sanity Check
    run_sanity_check(config, device, dummy_iterations=args.dummy_iterations)

    # Step 2: Stage 1 Next-Token Pre-Training
    pretrain_ckpt = Path("checkpoints/pretrain_core.pt")
    train_stage1_pretrain(
        config=config,
        device=device,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_sizes[0],
        max_len=args.max_len,
        max_batches=args.max_batches,
        save_path=pretrain_ckpt,
    )

    # Step 3: Stage 2 Fine-Tuning Grid Search
    best_results = run_finetune_grid_search(
        config=config,
        device=device,
        pretrained_checkpoint_path=pretrain_ckpt,
        learning_rates=args.learning_rates,
        batch_sizes=args.batch_sizes,
        epochs=args.finetune_epochs,
        patience=args.patience,
        max_len=args.max_len,
        max_batches=args.max_batches,
        wandb_mode=args.wandb_mode,
        checkpoints_dir=args.checkpoints_dir,
        vocab_source_path=args.vocab_path,
    )

    # Step 4: Export predictions.csv for held-out test cohort
    if args.export_predictions:
        best_model_path = args.checkpoints_dir / "pytorch_model.bin"
        if best_model_path.exists():
            logger.info("Generating predictions.csv on held-out test cohort (with Method A pre-existing mask applied)...")
            best_model = DualStreamEHRTransformer(config).to(device)
            best_model.load_state_dict(torch.load(best_model_path, map_location=device))

            _, _, test_dl = get_dataloaders(
                batch_size=args.batch_sizes[0],
                mode="finetune",
                max_len=args.max_len,
                include_test=True,
            )

            target_codes, target_token_ids = load_target_token_ids(args.vocab_path, "data/target_conditions.csv")
            pids, test_probs = predict_probabilities(
                best_model, test_dl, device, target_token_ids=target_token_ids
            )

            pred_df = pd.DataFrame(test_probs, columns=target_codes)
            pred_df.insert(0, "patient_id", pids)
            pred_df.to_csv("predictions.csv", index=False)
            logger.info(f"Saved predictions.csv with shape {pred_df.shape} for submission (pre-existing forced to 0.0).")

    total_time = time.time() - t_start

    # Print Final Summary
    print("\n" + "=" * 80)
    print("TRAINING & EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total Pipeline Runtime : {total_time:.1f} seconds")
    print(f"Best Configuration     : {best_results.get('run_name')} (LR: {best_results.get('learning_rate')}, BS: {best_results.get('batch_size')})")
    if "best_metrics" in best_results:
        m = best_results["best_metrics"]
        print(f"  - Best Val mAP       : {m['val_map']:.4f}")
        print(f"  - Best Val AUROC     : {m['val_macro_auroc']:.4f}")
        print(f"  - Best Val BCE Loss  : {m['val_loss']:.4f}")
        print(f"  - Accuracy (0.5 th)  : {m['val_accuracy']:.4f}")
    print(f"Exported Hub Directory : {args.checkpoints_dir}")
    print("Exported Hub Files     :")
    for f in args.checkpoints_dir.glob("*"):
        print(f"  - {f.name} ({f.stat().st_size:,} bytes)")
    if Path("predictions.csv").exists():
        print(f"Submission Prediction  : predictions.csv ({Path('predictions.csv').stat().st_size:,} bytes)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

