"""EHR PyTorch Dataset and DataLoader Module

Provides EHRDataset class supporting pretraining (next-token prediction)
and fine-tuning (multi-label condition prediction) with dynamic padding collate function.
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
from pathlib import Path
from typing import Any
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PAD_ID = 0  # <PAD> token ID from vocab.json


def load_genomic_variant_map(
    genomics_dir: str | Path | None = "genomics",
    vocab_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Loads variant token IDs from genomics/ folder for all available patients.

    Scans for *_dna.csv files, extracts loci where VARIANT is true,
    and returns a mapping {patient_id: tensor of variant token IDs in [1, vocab_size-1]}.
    """
    genomics_map: dict[str, torch.Tensor] = {}
    if genomics_dir is None:
        return genomics_map

    genomics_path = Path(genomics_dir)
    if not genomics_path.exists():
        for candidate in [
            Path("genomics"),
            Path(__file__).resolve().parent.parent / "genomics",
            Path(__file__).resolve().parent / "genomics",
        ]:
            if candidate.exists():
                genomics_path = candidate
                break

    if not genomics_path.exists() or not genomics_path.is_dir():
        return genomics_map

    dna_files = list(genomics_path.glob("*_dna.csv"))
    if not dna_files:
        return genomics_map

    for fpath in dna_files:
        pid = fpath.name.replace("_dna.csv", "")
        try:
            df = pd.read_csv(fpath, usecols=["INDEX", "VARIANT"])
            var_rows = df[df["VARIANT"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])]
            if len(var_rows) > 0:
                locus_indices = var_rows.index.to_numpy()
                token_ids = (locus_indices % (vocab_size - 1)) + 1
                genomics_map[pid] = torch.tensor(token_ids, dtype=torch.long)
            else:
                genomics_map[pid] = torch.empty(0, dtype=torch.long)
        except Exception:
            continue

    return genomics_map


class EHRDataset(Dataset):
    """PyTorch Dataset for patient timeline sequences.

    Supports:
    - mode="pretrain": Autoregressive next-token prediction targets.
    - mode="finetune": 40-class multi-label binary classification targets.
    - Actively loads and aligns genomic sequence tokens alongside clinical streams.
    """

    def __init__(
        self,
        records: list[dict],
        mode: str = "finetune",
        max_len: int | None = None,
        genomic_map: dict[str, torch.Tensor] | None = None,
        genomics_dir: str | Path | None = "genomics",
    ):
        """Args:
            records: List of patient records with keys ['patient_id', 'input_ids', 'age_ids', 'target'].
            mode: 'pretrain' or 'finetune'.
            max_len: Optional maximum sequence length (truncates from the front, keeping recent events).
            genomic_map: Optional preloaded mapping of patient_id -> genomic variant tensor.
            genomics_dir: Directory containing <Id>_dna.csv files if genomic_map not provided.
        """
        if mode not in ("pretrain", "finetune"):
            raise ValueError(f"Mode must be 'pretrain' or 'finetune', got '{mode}'")

        self.records = records
        self.mode = mode
        self.max_len = max_len

        if genomic_map is not None:
            self.genomic_map = genomic_map
        else:
            self.genomic_map = load_genomic_variant_map(genomics_dir)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        patient_id = rec["patient_id"]

        input_ids = rec["input_ids"]
        age_ids = rec["age_ids"]

        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        else:
            input_ids = input_ids.clone().detach().long()

        if not isinstance(age_ids, torch.Tensor):
            age_ids = torch.tensor(age_ids, dtype=torch.long)
        else:
            age_ids = age_ids.clone().detach().long()

        # Optional truncation (keep most recent events prior to anchor)
        if self.max_len is not None and len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len :]
            age_ids = age_ids[-self.max_len :]

        # Prepare genomic tokens aligned with clinical sequence length
        seq_len = len(input_ids)
        if patient_id in self.genomic_map and len(self.genomic_map[patient_id]) > 0:
            var_tokens = self.genomic_map[patient_id]
            genomic_ids = torch.zeros(seq_len, dtype=torch.long)
            k = min(len(var_tokens), seq_len)
            genomic_ids[:k] = var_tokens[:k]
        else:
            genomic_ids = torch.zeros(seq_len, dtype=torch.long)

        if self.mode == "pretrain":
            # Autoregressive next-token prediction:
            # Inputs: tokens[:-1], Targets: tokens[1:]
            if len(input_ids) > 1:
                seq_in = input_ids[:-1]
                age_in = age_ids[:-1]
                gen_in = genomic_ids[:-1]
                target_out = input_ids[1:]
            else:
                seq_in = input_ids
                age_in = age_ids
                gen_in = genomic_ids
                target_out = input_ids

            return {
                "patient_id": patient_id,
                "input_ids": seq_in,
                "age_ids": age_in,
                "genomic_ids": gen_in,
                "target": target_out,
                "mode": "pretrain",
            }

        else:
            # mode == "finetune":
            # Target is the 40-class multi-label binary vector
            raw_target = rec["target"]
            if not isinstance(raw_target, torch.Tensor):
                target_vec = torch.tensor(raw_target, dtype=torch.float32)
            else:
                target_vec = raw_target.clone().detach().float()

            return {
                "patient_id": patient_id,
                "input_ids": input_ids,
                "age_ids": age_ids,
                "genomic_ids": genomic_ids,
                "target": target_vec,
                "mode": "finetune",
            }


def ehr_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Dynamically pads sequences in the batch to the maximum sequence length.

    - Uses PAD_ID (0) for padding both input_ids and age_ids.
    - Generates a boolean attention mask: True for valid tokens, False for PAD.
    - Returns padded tensors and targets (shifted sequence for pretrain, stacked vectors for finetune).
    """
    patient_ids = [item["patient_id"] for item in batch]
    mode = batch[0]["mode"]
    batch_size = len(batch)

    # Determine maximum sequence length in this batch
    max_len = max(len(item["input_ids"]) for item in batch)

    # Allocate padded tensors
    padded_input_ids = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
    padded_age_ids = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
    padded_genomic_ids = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    if mode == "pretrain":
        padded_targets = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
        for i, item in enumerate(batch):
            seq_len = len(item["input_ids"])
            padded_input_ids[i, :seq_len] = item["input_ids"]
            padded_age_ids[i, :seq_len] = item["age_ids"]
            if "genomic_ids" in item:
                padded_genomic_ids[i, :seq_len] = item["genomic_ids"]
            attention_mask[i, :seq_len] = True
            padded_targets[i, :seq_len] = item["target"]

        return {
            "patient_ids": patient_ids,
            "input_ids": padded_input_ids,
            "age_ids": padded_age_ids,
            "genomic_ids": padded_genomic_ids,
            "attention_mask": attention_mask,
            "targets": padded_targets,
        }

    else:
        # Finetune mode: targets are [batch_size, 40]
        stacked_targets = torch.stack([item["target"] for item in batch], dim=0)
        for i, item in enumerate(batch):
            seq_len = len(item["input_ids"])
            padded_input_ids[i, :seq_len] = item["input_ids"]
            padded_age_ids[i, :seq_len] = item["age_ids"]
            if "genomic_ids" in item:
                padded_genomic_ids[i, :seq_len] = item["genomic_ids"]
            attention_mask[i, :seq_len] = True

        return {
            "patient_ids": patient_ids,
            "input_ids": padded_input_ids,
            "age_ids": padded_age_ids,
            "genomic_ids": padded_genomic_ids,
            "attention_mask": attention_mask,
            "targets": stacked_targets,
        }


def get_dataloaders(
    batch_size: int = 32,
    mode: str = "finetune",
    data_path: str | Path = "data/tokenized_data.pt",
    splits_path: str | Path = "data/patient_splits.csv",
    genomics_dir: str | Path | None = "genomics",
    max_len: int | None = None,
    num_workers: int = 0,
    include_test: bool = False,
) -> tuple[DataLoader, DataLoader] | tuple[DataLoader, DataLoader, DataLoader]:
    """Loads tokenized data, filters by patient_splits.csv, and constructs PyTorch DataLoaders.

    Args:
        batch_size: Batch size for training and validation.
        mode: 'pretrain' or 'finetune'.
        data_path: Path to tokenized_data.pt (default: data/tokenized_data.pt).
        splits_path: Path to patient_splits.csv (default: data/patient_splits.csv).
        genomics_dir: Directory containing genomics files (default: genomics).
        max_len: Optional sequence length cutoff.
        num_workers: DataLoader worker count.
        include_test: If True, also returns test_loader.

    Returns:
        (train_loader, val_loader) or (train_loader, val_loader, test_loader)
    """
    data_path = Path(data_path)
    if not data_path.exists():
        for candidate in [
            Path("data") / data_path.name,
            Path(__file__).resolve().parent.parent / "data" / data_path.name,
            Path(__file__).resolve().parent / "data" / data_path.name,
        ]:
            if candidate.exists():
                data_path = candidate
                break

    splits_path = Path(splits_path)
    if not splits_path.exists():
        for candidate in [
            Path("data") / splits_path.name,
            Path(__file__).resolve().parent.parent / "data" / splits_path.name,
            Path(__file__).resolve().parent / "data" / splits_path.name,
        ]:
            if candidate.exists():
                splits_path = candidate
                break

    if not data_path.exists():
        raise FileNotFoundError(f"Tokenized data file not found at {data_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Patient splits file not found at {splits_path}")

    # Load splits mapping
    splits_df = pd.read_csv(splits_path)
    split_map = dict(zip(splits_df["Id"], splits_df["split"]))

    # Preload genomics variant map once for efficiency
    genomic_map = load_genomic_variant_map(genomics_dir)

    # Load tokenized data
    raw_data = torch.load(data_path)

    # Collect all records across splits in tokenized data
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

    # Cross-reference with patient_splits.csv
    train_records = []
    val_records = []
    test_records = []

    for rec in all_records:
        pid = rec["patient_id"]
        assigned_split = split_map.get(pid)
        if assigned_split == "train":
            train_records.append(rec)
        elif assigned_split == "val":
            val_records.append(rec)
        elif assigned_split == "test":
            test_records.append(rec)

    # Instantiate EHRDataset objects with active genomic map
    train_dataset = EHRDataset(train_records, mode=mode, max_len=max_len, genomic_map=genomic_map)
    val_dataset = EHRDataset(val_records, mode=mode, max_len=max_len, genomic_map=genomic_map)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ehr_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ehr_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if include_test:
        test_dataset = EHRDataset(test_records, mode=mode, max_len=max_len, genomic_map=genomic_map)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=ehr_collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return train_loader, val_loader, test_loader

    return train_loader, val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test EHR Dataset and DataLoaders")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--mode", type=str, default="finetune", choices=["pretrain", "finetune"])
    args = parser.parse_args()

    print(f"Testing EHRDataset and DataLoaders in '{args.mode}' mode with batch_size={args.batch_size}...")
    train_dl, val_dl = get_dataloaders(batch_size=args.batch_size, mode=args.mode)

    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")

    # Fetch a sample batch
    sample_batch = next(iter(train_dl))
    print("\nSample batch structure:")
    print(f"  patient_ids count : {len(sample_batch['patient_ids'])}")
    print(f"  input_ids shape   : {sample_batch['input_ids'].shape} (dtype: {sample_batch['input_ids'].dtype})")
    print(f"  age_ids shape     : {sample_batch['age_ids'].shape} (dtype: {sample_batch['age_ids'].dtype})")
    print(f"  genomic_ids shape : {sample_batch['genomic_ids'].shape} (dtype: {sample_batch['genomic_ids'].dtype})")
    print(f"  attention_mask    : {sample_batch['attention_mask'].shape} (dtype: {sample_batch['attention_mask'].dtype})")
    print(f"  targets shape     : {sample_batch['targets'].shape} (dtype: {sample_batch['targets'].dtype})")

    # Verify mask
    mask_matches_nonzero = (sample_batch["attention_mask"] == (sample_batch["input_ids"] != PAD_ID)).all().item()
    print(f"  attention_mask matches non-pad tokens: {mask_matches_nonzero}")
    patients_with_dna = (sample_batch["genomic_ids"] != PAD_ID).any(dim=1).sum().item()
    print(f"  patients with active genomic variants in batch: {patients_with_dna}/{len(sample_batch['patient_ids'])}")
    print("\nDataset test completed successfully.")
