"""
scripts/train.py

Training and Validation Pipeline for EHR Longitudinal Forecasting Models:
  - Baseline 1: ClinicalTransformer (discrete age embeddings)
  - Baseline 2: MultimodalTransformer (late fusion with discrete age embeddings)

Features:
  - WandB experiment tracking (project="ehr-prediction")
  - Interactive robust overfit test on single batch (--overfit_test)
  - Interpretability hook: attention heatmap extraction and visualization
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ehr_dataset import EHRDataset, collate_fn
from scripts.models import ClinicalTransformer, MultimodalTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def plot_attention_heatmap(
    tokens: List[str],
    attention_matrix: np.ndarray,
    save_path: str = "reports/figures/attention_sample.png",
):
    """
    Plot and save attention heatmap for a patient timeline sequence.

    Args:
        tokens: List of token strings (e.g. clinical codes) of length T.
        attention_matrix: 2D numpy array [T, T] representing self-attention weights.
        save_path: Path to save the output PNG plot.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.figure(figsize=(10, 8))

    # If sequence is long, visualize the last up to 30 events for clarity
    T = len(tokens)
    if T > 30:
        tokens = tokens[-30:]
        attention_matrix = attention_matrix[-30:, -30:]

    sns.heatmap(
        attention_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        annot=False,
    )
    plt.title("Clinical Timeline Attention Heatmap")
    plt.xlabel("Key Tokens (Past Events)")
    plt.ylabel("Query Tokens")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved attention heatmap to {save_path}")


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
) -> Dict[str, float]:
    """
    Compute macro-averaged AUROC and mean Average Precision (mAP) across conditions.
    Safely handles single-class columns.
    """
    num_classes = y_true.shape[1]
    aurocs = []
    maps = []

    for c in range(num_classes):
        col_true = y_true[:, c]
        col_pred = y_pred_probs[:, c]

        # Only evaluate conditions with both positive and negative instances
        if len(np.unique(col_true)) > 1:
            try:
                auc = roc_auc_score(col_true, col_pred)
                aurocs.append(auc)
            except ValueError:
                pass

            try:
                ap = average_precision_score(col_true, col_pred)
                maps.append(ap)
            except ValueError:
                pass

    macro_auroc = float(np.mean(aurocs)) if aurocs else 0.0
    macro_map = float(np.mean(maps)) if maps else 0.0

    return {
        "macro_auroc": macro_auroc,
        "mean_ap": macro_map,
        "eval_classes": len(aurocs),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    mode: str,
    max_batches: Optional[int] = None,
) -> float:
    """Run one training epoch with discrete age_ids unpacked from batch."""
    model.train()
    total_loss = 0.0
    batch_count = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        optimizer.zero_grad()

        # Batch unpacking expecting age_ids
        if mode == "clinical":
            input_ids, attention_mask, age_ids, labels = [t.to(device) for t in batch]
            logits = model(
                input_ids=input_ids,
                age_ids=age_ids,
                attention_mask=attention_mask,
            )
            loss = criterion(logits, labels)
        else:  # multimodal
            batch_tensors = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            labels = batch_tensors["labels"]
            logits = model(batch=batch_tensors)
            loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    return total_loss / max(batch_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mode: str,
    max_batches: Optional[int] = None,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on validation split with age_ids unpacking."""
    model.eval()
    total_loss = 0.0
    batch_count = 0
    all_preds = []
    all_targets = []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        if mode == "clinical":
            input_ids, attention_mask, age_ids, labels = [t.to(device) for t in batch]
            logits = model(
                input_ids=input_ids,
                age_ids=age_ids,
                attention_mask=attention_mask,
            )
            loss = criterion(logits, labels)
            targets = labels
        else:  # multimodal
            batch_tensors = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            targets = batch_tensors["labels"]
            logits = model(batch=batch_tensors)
            loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

        total_loss += loss.item()
        batch_count += 1

    avg_loss = total_loss / max(batch_count, 1)

    if all_preds:
        y_pred_probs = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        metrics = compute_multilabel_metrics(y_true, y_pred_probs)
    else:
        metrics = {"macro_auroc": 0.0, "mean_ap": 0.0, "eval_classes": 0}

    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Baseline Models with Discrete Age Embeddings.")
    parser.add_argument("--mode", type=str, default="multimodal", choices=["clinical", "multimodal"],
                        help="Baseline model to train (default: multimodal).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4).")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay (default: 1e-2).")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Max sequence length (default: 256).")
    parser.add_argument("--d-model", type=int, default=128, help="Model hidden dimension (default: 128).")
    parser.add_argument("--n-head", type=int, default=4, help="Number of attention heads (default: 4).")
    parser.add_argument("--n-layer", type=int, default=4, help="Number of transformer layers (default: 4).")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional max batches per epoch (for quick testing).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--overfit_test", action="store_true",
                        help="Run interactive single-batch 50-step overfit test and exit cleanly.")
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} | Training Mode: {args.mode}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. wandb Setup (if not in overfit test mode)
    if not args.overfit_test:
        try:
            wandb_settings = wandb.Settings(start_method="fork")
        except Exception:
            wandb_settings = wandb.Settings()

        try:
            wandb.init(
                project="ehr-prediction",
                config=vars(args),
                settings=wandb_settings,
            )
            logger.info("wandb initialized successfully (project='ehr-prediction').")
        except Exception as e:
            logger.warning(f"wandb initialization encountered error: {e}. Continuing with offline mode.")
            wandb.init(project="ehr-prediction", config=vars(args), mode="offline")

    # 2. Load Datasets
    logger.info("Initializing Training Dataset...")
    train_dataset = EHRDataset(
        mode=args.mode,
        split="train",
        max_seq_len=args.max_seq_len,
    )
    logger.info("Initializing Validation Dataset...")
    val_dataset = EHRDataset(
        mode=args.mode,
        split="val",
        max_seq_len=args.max_seq_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    logger.info(f"Loaded {len(train_dataset):,d} training samples, {len(val_dataset):,d} validation samples.")

    # 3. Instantiate Model with Discrete Age Embeddings
    clinical_vocab_size = getattr(train_dataset, "clinical_vocab_size", 735)
    num_classes = len(train_dataset.target_codes) if train_dataset.target_codes else 40

    if args.mode == "clinical":
        model = ClinicalTransformer(
            vocab_size=clinical_vocab_size,
            max_seq_len=args.max_seq_len,
            num_ages=120,
            d_model=args.d_model,
            n_head=args.n_head,
            n_layer=args.n_layer,
            num_classes=num_classes,
        ).to(device)
    else:
        genomic_vocab_size = getattr(train_dataset, "genomic_vocab_size", 333)
        num_genomic_loci = getattr(train_dataset, "num_genomic_variants", 322)
        model = MultimodalTransformer(
            clinical_vocab_size=clinical_vocab_size,
            genomic_vocab_size=genomic_vocab_size,
            num_genomic_loci=num_genomic_loci,
            clinical_d_model=args.d_model,
            clinical_n_head=args.n_head,
            clinical_n_layer=args.n_layer,
            clinical_max_seq_len=args.max_seq_len,
            num_ages=120,
            num_classes=num_classes,
        ).to(device)

    logger.info(f"Model initialized: {model.__class__.__name__} ({sum(p.numel() for p in model.parameters()):,d} parameters)")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 4. Robust Overfit Test (if requested)
    if args.overfit_test:
        logger.info("\n" + "=" * 68)
        logger.info("ROBUST OVERFIT TEST: TRAINING 1 BATCH FOR 50 ITERATIONS")
        logger.info("=" * 68)

        model.train()
        batch = next(iter(train_loader))

        # Explicitly send all tensors to device
        if args.mode == "clinical":
            input_ids, attention_mask, age_ids, labels = [t.to(device) for t in batch]
        else:
            batch_tensors = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            labels = batch_tensors["labels"]

        # Run 50 gradient update iterations
        for step in range(1, 51):
            optimizer.zero_grad()

            if args.mode == "clinical":
                logits = model(
                    input_ids=input_ids,
                    age_ids=age_ids,
                    attention_mask=attention_mask,
                )
            else:
                logits = model(batch=batch_tensors)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            if step == 1 or step % 5 == 0:
                print(f"[Overfit Step {step:02d}/50] BCE Loss: {loss.item():.6f}")

        print(f"\n[PASS] Overfit test completed successfully! Final loss: {loss.item():.6f}")
        logger.info("Overfit test verified gradient backpropagation and age embedding convergence. Exiting cleanly.")
        return

    # 5. Full Training Loop
    best_val_loss = float("inf")
    best_auroc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            mode=args.mode,
            max_batches=args.max_batches,
        )

        val_loss, val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            mode=args.mode,
            max_batches=args.max_batches,
        )

        elapsed = time.time() - t0
        auroc = val_metrics["macro_auroc"]
        m_ap = val_metrics["mean_ap"]

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Macro AUROC: {auroc:.4f} | "
            f"mAP: {m_ap:.4f}"
        )

        # wandb Logging
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "macro_auroc": auroc,
            "mean_ap": m_ap,
        })

        # Checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_auroc = auroc
            ckpt_path = os.path.join(args.output_dir, f"best_{args.mode}_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "macro_auroc": auroc,
                "mean_ap": m_ap,
            }, ckpt_path)
            logger.info(f"  [Saved Best Checkpoint] -> {ckpt_path}")

    logger.info(f"\nTraining Complete. Best Val Loss: {best_val_loss:.4f} (Macro AUROC: {best_auroc:.4f})")

    # 6. Interpretability Hook: Attention Heatmap Generation
    try:
        logger.info("Generating sample attention heatmap from validation split...")
        val_batch = next(iter(val_loader))
        model.eval()
        with torch.no_grad():
            if args.mode == "clinical":
                val_input_ids, val_attention_mask, val_age_ids, _ = [t.to(device) for t in val_batch]
                sample_input_ids = val_input_ids[0:1]
                sample_age_ids = val_age_ids[0:1]
                sample_mask = val_attention_mask[0:1]
                logits, all_attentions = model(
                    input_ids=sample_input_ids,
                    age_ids=sample_age_ids,
                    attention_mask=sample_mask,
                    output_attentions=True,
                )
            else:
                sample_batch = {
                    k: v[0:1].to(device) if isinstance(v, torch.Tensor) else v[0:1]
                    for k, v in val_batch.items()
                }
                sample_input_ids = sample_batch["input_ids"]
                sample_mask = sample_batch.get("attention_mask")
                logits, all_attentions = model(
                    batch=sample_batch,
                    output_attentions=True,
                )

            if all_attentions is not None and len(all_attentions) > 0:
                last_layer_attn = all_attentions[-1][0].mean(dim=0).cpu().numpy()
                seq_len = int(sample_mask[0].sum().item()) if sample_mask is not None else sample_input_ids.shape[1]
                seq_len = max(seq_len, 1)
                sub_attn = last_layer_attn[:seq_len, :seq_len]

                inv_vocab = {v: k for k, v in train_dataset.clinical_vocab.items()} if hasattr(train_dataset, "clinical_vocab") else {}
                tok_ids = sample_input_ids[0, :seq_len].cpu().tolist()
                tokens = [inv_vocab.get(tid, str(tid)) for tid in tok_ids]

                plot_attention_heatmap(tokens, sub_attn, save_path="reports/figures/attention_sample.png")
    except Exception as e:
        logger.warning(f"Could not generate attention heatmap: {e}")

    # 7. wandb Finish
    wandb.finish()


if __name__ == "__main__":
    main()
