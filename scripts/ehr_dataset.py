"""
scripts/ehr_dataset.py

Unified PyTorch Dataset and Transformer collate_fn supporting the 3-baseline ensemble strategy:
  1. 'clinical'   - Temporal clinical event sequences with discrete age embeddings (age_ids)
  2. 'genomic'    - Static patient genomic variant tensors with <MISSING_GENOME> fallback
  3. 'multimodal' - Aligned clinical timelines (with age_ids) + genomic variant tensors per patient

Collate Function Return Signatures:
  - For 'clinical':   returns (input_ids, attention_mask, age_ids, labels)
  - For 'genomic':    returns (genomic_ids, genomic_mask, labels)
  - For 'multimodal': returns aligned dictionary {
      'input_ids', 'attention_mask', 'age_ids',
      'genomic_ids', 'genomic_mask', 'has_genomics',
      'labels', 'patient_ids'
    }
"""

import os
import sys
import json
import glob
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class EHRDataset(Dataset):
    """
    Unified PyTorch Dataset for Clinical, Genomic, and Multimodal EHR Modeling.

    Supported Modes:
      - 'clinical':   Timestamped clinical sequence tokens and discrete age embeddings (age_ids).
      - 'genomic':    Genomic variant tokens with <MISSING_GENOME> fallback.
      - 'multimodal': Aligned clinical sequences (with age_ids), genomic vectors, and has_genomics indicator.
    """

    DEFAULT_CLINICAL_TABLES = [
        "conditions.csv",
        "medications.csv",
        "procedures.csv",
        "observations.csv",
        "encounters.csv",
        "immunizations.csv",
        "allergies.csv",
        "careplans.csv",
        "devices.csv",
        "imaging_studies.csv",
    ]

    GENETIC_OBSERVATION_CODES = [
        "85318-4",   # HER2 by FISH
        "85319-2",   # HER2 by Immune stain
        "85337-4",   # Estrogen receptor Ag
        "85339-0",   # Progesterone receptor Ag
        "10480-2",   # Estrogen+Progesterone receptor Ag
        "417181009", # Estrogen+Progesterone receptor Ag
    ]

    def __init__(
        self,
        mode: str = "clinical",
        data_dir: str = "train_val_pre_anchor",
        genomics_dir: str = "genomics",
        clinical_vocab_path: str = "data/clinical_vocab.json",
        genomic_vocab_path: str = "data/genomic_vocab.json",
        splits_path: Optional[str] = "data/patient_splits.csv",
        split: Optional[str] = "train",
        targets_path: Optional[str] = "data/train_val_targets.csv",
        target_conditions_path: Optional[str] = "data/target_conditions.csv",
        max_seq_len: int = 512,
        min_seq_len: int = 1,
        included_tables: Optional[List[str]] = None,
        cache_path: Optional[str] = None,
        force_reload: bool = False,
    ):
        """
        Args:
            mode: 'clinical', 'genomic', or 'multimodal'.
            data_dir: Directory containing pre-anchor clinical CSV tables.
            genomics_dir: Directory containing <Id>_dna.csv files.
            clinical_vocab_path: Path to clinical_vocab.json.
            genomic_vocab_path: Path to genomic_vocab.json.
            splits_path: Optional path to patient_splits.csv.
            split: 'train', 'val', 'test', or None for all patients.
            targets_path: Optional path to train_val_targets.csv.
            target_conditions_path: Optional path to target_conditions.csv.
            max_seq_len: Maximum clinical sequence length per patient.
            min_seq_len: Minimum clinical events to include patient in clinical mode.
            included_tables: Specific clinical tables to load.
            cache_path: Optional path to .pt cache file.
            force_reload: If True, bypass cache.
        """
        super().__init__()
        valid_modes = ["clinical", "genomic", "multimodal"]
        if mode.lower() not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")

        self.mode = mode.lower()
        self.data_dir = data_dir
        self.genomics_dir = genomics_dir
        self.clinical_vocab_path = (
            clinical_vocab_path
            if os.path.exists(clinical_vocab_path)
            else ("data/vocab.json" if os.path.exists("data/vocab.json") else "vocab.json")
        )
        self.genomic_vocab_path = genomic_vocab_path
        self.split = split
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

        # 1. Load Vocabularies
        self.clinical_vocab = {}
        if self.mode in ["clinical", "multimodal"]:
            if not os.path.exists(self.clinical_vocab_path):
                raise FileNotFoundError(f"Clinical vocabulary '{self.clinical_vocab_path}' not found. Run scripts/tokenizer.py first.")
            with open(self.clinical_vocab_path, "r", encoding="utf-8") as f:
                self.clinical_vocab = json.load(f)
            self.clinical_pad_id = self.clinical_vocab.get("<PAD>", 0)
            self.clinical_unk_id = self.clinical_vocab.get("<UNK>", 1)
            self.clinical_bos_id = self.clinical_vocab.get("<BOS>", 2)
            self.clinical_eos_id = self.clinical_vocab.get("<EOS>", 3)
            self.clinical_vocab_size = len(self.clinical_vocab)

        self.genomic_vocab = {}
        self.genomic_variant_loci = []
        if self.mode in ["genomic", "multimodal"]:
            if not os.path.exists(self.genomic_vocab_path):
                raise FileNotFoundError(f"Genomic vocabulary '{self.genomic_vocab_path}' not found. Run scripts/tokenizer.py first.")
            with open(self.genomic_vocab_path, "r", encoding="utf-8") as f:
                self.genomic_vocab = json.load(f)
            self.genomic_pad_id = self.genomic_vocab.get("<PAD>", 0)
            self.genomic_unk_id = self.genomic_vocab.get("<UNK>", 1)
            self.genomic_missing_id = self.genomic_vocab.get("<MISSING_GENOME>", 2)
            self.genomic_vocab_size = len(self.genomic_vocab)
            # Ordered variant loci (excluding special tokens and significance tokens)
            self.genomic_variant_loci = [
                k for k in self.genomic_vocab.keys()
                if not k.startswith("<") and not k.startswith("SIG_")
            ]
            self.num_genomic_variants = len(self.genomic_variant_loci)

        # 2. Check Cache
        if cache_path and os.path.exists(cache_path) and not force_reload:
            logger.info(f"Loading pre-cached EHR dataset from '{cache_path}'...")
            cached_data = torch.load(cache_path, weights_only=False)
            self.patient_ids = cached_data["patient_ids"]
            self.samples = cached_data["samples"]
            self.target_codes = cached_data.get("target_codes", [])
            logger.info(f"Loaded {len(self.samples):,d} patient samples from cache (mode='{self.mode}').")
            return

        # 3. Determine target patients for this split
        target_patients = None
        if splits_path and os.path.exists(splits_path) and split is not None:
            splits_df = pd.read_csv(splits_path)
            target_patients = set(splits_df[splits_df["split"] == split]["Id"].dropna())
            logger.info(f"Filtering to split '{split}': {len(target_patients):,d} patients found in {splits_path}.")

        # 4. Load ground truth target labels
        self.targets_dict = {}
        self.target_codes = []
        if targets_path and os.path.exists(targets_path):
            targets_df = pd.read_csv(targets_path)
            patient_col = "PATIENT" if "PATIENT" in targets_df.columns else "Id"

            if target_conditions_path and os.path.exists(target_conditions_path):
                tc_df = pd.read_csv(target_conditions_path)
                desired_codes = [str(c) for c in tc_df["CODE"].unique()]
                self.target_codes = [c for c in desired_codes if c in targets_df.columns]
            else:
                self.target_codes = [c for c in targets_df.columns if c != patient_col]

            for _, row in targets_df.iterrows():
                pid = str(row[patient_col])
                vec = row[self.target_codes].values.astype(np.float32)
                self.targets_dict[pid] = torch.from_numpy(vec)

            logger.info(f"Loaded targets for {len(self.targets_dict):,d} patients ({len(self.target_codes)} condition classes).")

        # 5. Genomic Alignment Logic: Parse & Cache Variant Tensors with Fallback
        self.genomic_records = {}
        if self.mode in ["genomic", "multimodal"]:
            self._build_genomic_records(target_patients)

        # 6. Clinical Timeline Ingestion (if clinical or multimodal mode)
        clinical_timelines = {}
        if self.mode in ["clinical", "multimodal"]:
            self.patient_birthdates = self._load_patient_birthdates()
            tables_to_load = included_tables if included_tables is not None else self.DEFAULT_CLINICAL_TABLES
            clinical_timelines = self._load_clinical_timelines(tables_to_load, target_patients)

        # 7. Assemble Dataset Samples
        self.samples = []
        self.patient_ids = []

        if self.mode == "clinical":
            candidate_pids = sorted(clinical_timelines.keys())
        elif self.mode == "genomic":
            candidate_pids = sorted(self.genomic_records.keys())
        else:  # multimodal
            all_pids = set(clinical_timelines.keys()).union(self.genomic_records.keys())
            if target_patients is not None:
                all_pids = all_pids.intersection(target_patients)
            candidate_pids = sorted(all_pids)

        logger.info(f"Assembling {len(candidate_pids):,d} patient samples for mode='{self.mode}'...")

        for pid in candidate_pids:
            target_vec = self.targets_dict.get(
                pid,
                torch.zeros(len(self.target_codes), dtype=torch.float32) if self.target_codes else torch.zeros(40, dtype=torch.float32)
            )

            if self.mode == "clinical":
                t_data = clinical_timelines.get(pid, ([], []))
                token_ids, age_ids = t_data
                if len(token_ids) < self.min_seq_len:
                    continue
                sample = {
                    "patient_id": pid,
                    "mode": "clinical",
                    "input_ids": torch.tensor(token_ids, dtype=torch.long),
                    "age_ids": torch.tensor(age_ids, dtype=torch.long),
                    "labels": target_vec,
                }
                self.samples.append(sample)
                self.patient_ids.append(pid)

            elif self.mode == "genomic":
                g_record = self.genomic_records.get(pid)
                if g_record is None:
                    # Missing genome fallback
                    g_ids = torch.full((self.num_genomic_variants,), fill_value=self.genomic_missing_id, dtype=torch.long)
                    g_mask = torch.zeros(self.num_genomic_variants, dtype=torch.bool)
                    has_gen = False
                else:
                    g_ids, g_mask, has_gen = g_record

                sample = {
                    "patient_id": pid,
                    "mode": "genomic",
                    "genomic_ids": g_ids,
                    "genomic_mask": g_mask,
                    "has_genomics": torch.tensor(1.0 if has_gen else 0.0, dtype=torch.float32),
                    "labels": target_vec,
                }
                self.samples.append(sample)
                self.patient_ids.append(pid)

            elif self.mode == "multimodal":
                t_data = clinical_timelines.get(pid, ([], []))
                token_ids, age_ids = t_data

                g_record = self.genomic_records.get(pid)
                if g_record is None:
                    g_ids = torch.full((self.num_genomic_variants,), fill_value=self.genomic_missing_id, dtype=torch.long)
                    g_mask = torch.zeros(self.num_genomic_variants, dtype=torch.bool)
                    has_gen = False
                else:
                    g_ids, g_mask, has_gen = g_record

                sample = {
                    "patient_id": pid,
                    "mode": "multimodal",
                    "input_ids": torch.tensor(token_ids, dtype=torch.long),
                    "age_ids": torch.tensor(age_ids, dtype=torch.long),
                    "genomic_ids": g_ids,
                    "genomic_mask": g_mask,
                    "has_genomics": torch.tensor(1.0 if has_gen else 0.0, dtype=torch.float32),
                    "labels": target_vec,
                }
                self.samples.append(sample)
                self.patient_ids.append(pid)

        logger.info(f"Dataset successfully built: {len(self.samples):,d} samples (mode='{self.mode}', split='{split}').")

        # 8. Optional Cache Save
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)) if os.path.dirname(cache_path) else ".", exist_ok=True)
            torch.save({
                "patient_ids": self.patient_ids,
                "samples": self.samples,
                "target_codes": self.target_codes,
                "mode": self.mode,
            }, cache_path)
            logger.info(f"Saved dataset cache to '{cache_path}'.")

    def _build_genomic_records(self, target_patients: Optional[set]):
        """
        Parse genomic variant files for all patients.
        If genomic file exists:
          Tokenizes loci into genomic_ids and flags presence in genomic_mask, setting has_genomics=True.
        If missing:
          Populates tensor with <MISSING_GENOME> tokens and sets has_genomics=False.
        """
        logger.info("Executing genomic integration & missing-data fallback logic...")

        # 1. Inspect observations for genetic testing context
        obs_path = os.path.join(self.data_dir, "observations.csv")
        obs_genetic_patients = set()
        if os.path.exists(obs_path):
            obs_df = pd.read_csv(obs_path, usecols=["PATIENT", "CODE", "DESCRIPTION"], dtype=str, low_memory=False)
            code_match = obs_df["CODE"].isin(self.GENETIC_OBSERVATION_CODES)
            desc_match = obs_df["DESCRIPTION"].astype(str).str.contains(
                r"FISH|receptor|gene|genetic|DNA|karyotyp|aneuploidy|molecular",
                case=False,
                na=False,
            )
            obs_genetic_patients = set(obs_df[code_match | desc_match]["PATIENT"].dropna().unique())

        # 2. Check genomics directory
        dna_files = glob.glob(os.path.join(self.genomics_dir, "*_dna.csv"))
        logger.info(f"  Found {len(dna_files):,d} genomic DNA files in '{self.genomics_dir}'.")

        locus_to_id = {locus: self.genomic_vocab.get(locus, self.genomic_unk_id) for locus in self.genomic_variant_loci}
        default_locus_ids = torch.tensor([locus_to_id[l] for l in self.genomic_variant_loci], dtype=torch.long)

        # 3. Process existing DNA files
        loaded_count = 0
        for fpath in dna_files:
            fname = os.path.basename(fpath)
            pid = fname.replace("_dna.csv", "")

            if target_patients is not None and pid not in target_patients:
                continue

            try:
                df = pd.read_csv(fpath, usecols=["INDEX", "VARIANT"], dtype=str, low_memory=False)
                # Map locus variants: True if VARIANT is true
                var_dict = dict(zip(df["INDEX"].str.strip(), df["VARIANT"].str.strip().str.lower()))
                
                # genomic_ids has the locus IDs
                g_ids = default_locus_ids.clone()
                # genomic_mask is True for active mutations (VARIANT == True)
                g_mask = torch.tensor(
                    [var_dict.get(locus, "") in ["true", "1", "yes"] for locus in self.genomic_variant_loci],
                    dtype=torch.bool,
                )

                self.genomic_records[pid] = (g_ids, g_mask, True)
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Error parsing genomic file {fname}: {e}")

        logger.info(f"  Mapped {loaded_count:,d} patients with valid genomic variant tensors.")

        # 4. For any patient in target_patients lacking DNA, populate fallback
        missing_count = 0
        if target_patients is not None:
            for pid in target_patients:
                if pid not in self.genomic_records:
                    fallback_ids = torch.full(
                        (self.num_genomic_variants,),
                        fill_value=self.genomic_missing_id,
                        dtype=torch.long,
                    )
                    fallback_mask = torch.zeros(self.num_genomic_variants, dtype=torch.bool)
                    self.genomic_records[pid] = (fallback_ids, fallback_mask, False)
                    missing_count += 1

        logger.info(f"  Applied <MISSING_GENOME> fallback to {missing_count:,d} patients without DNA.")

    def _load_patient_birthdates(self) -> Dict[str, np.datetime64]:
        """
        Load patients.csv and map each patient Id to their BIRTHDATE.
        Checks self.data_dir as well as fallback directories.
        """
        candidate_paths = [
            os.path.join(self.data_dir, "patients.csv"),
            "train_val_pre_anchor/patients.csv",
            "train_val/patients.csv",
            "test/patients.csv",
        ]
        birthdate_dict = {}
        for ppath in candidate_paths:
            if os.path.exists(ppath):
                try:
                    df = pd.read_csv(ppath, usecols=["Id", "BIRTHDATE"], dtype=str)
                    for _, row in df.iterrows():
                        pid = str(row["Id"]).strip()
                        b_str = str(row["BIRTHDATE"]).strip()
                        if pid and b_str and b_str.lower() != "nan" and pid not in birthdate_dict:
                            try:
                                dt = np.datetime64(pd.to_datetime(b_str).strftime("%Y-%m-%dT%H:%M:%S"), "s")
                                birthdate_dict[pid] = dt
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"Error reading patients file '{ppath}': {e}")

        logger.info(f"Loaded BIRTHDATE mappings for {len(birthdate_dict):,d} patients.")
        return birthdate_dict

    def _load_clinical_timelines(
        self,
        table_names: List[str],
        target_patients: Optional[set],
    ) -> Dict[str, Tuple[List[int], List[int]]]:
        """
        Load and chronologically order clinical event sequences with discrete age embeddings per patient.

        Returns:
            {patient_id: (token_ids_list, age_ids_list)}
        """
        logger.info("Loading pre-anchor clinical events for timeline & age_ids construction...")
        event_dfs = []

        for fname in table_names:
            fpath = os.path.join(self.data_dir, fname)
            if not os.path.exists(fpath):
                continue

            sample_df = pd.read_csv(fpath, nrows=2)
            pcol = "PATIENT" if "PATIENT" in sample_df.columns else ("Id" if "Id" in sample_df.columns else None)
            dcol = "START" if "START" in sample_df.columns else ("DATE" if "DATE" in sample_df.columns else None)
            code_cols = [c for c in sample_df.columns if "CODE" in c.upper()]

            if not pcol or not dcol or not code_cols:
                continue

            df = pd.read_csv(fpath, usecols=[pcol, dcol] + code_cols, dtype=str, low_memory=False)
            if target_patients is not None:
                df = df[df[pcol].isin(target_patients)]
            if len(df) == 0:
                continue

            for c_col in code_cols:
                sub = df[[pcol, dcol, c_col]].dropna().copy()
                sub.columns = ["PATIENT", "timestamp", "raw_code"]

                sub["norm_code"] = sub["raw_code"].astype(str).str.strip()
                has_point_zero = sub["norm_code"].str.endswith(".0")
                sub["norm_code"] = sub["norm_code"].mask(has_point_zero, sub["norm_code"].str[:-2])
                sub = sub[sub["norm_code"].isin(self.clinical_vocab)]

                if len(sub) > 0:
                    event_dfs.append(sub[["PATIENT", "timestamp", "norm_code"]])

        if not event_dfs:
            return {}

        all_events = pd.concat(event_dfs, ignore_index=True)
        # Parse timestamp to UTC datetime (handling mixed ISO formats)
        all_events["dt"] = pd.to_datetime(all_events["timestamp"], utc=True, format="mixed")
        all_events.sort_values(by=["PATIENT", "dt"], inplace=True)

        timelines = {}
        for pid, group in all_events.groupby("PATIENT"):
            codes = group["norm_code"].tolist()
            dts = group["dt"].values

            token_ids = [self.clinical_vocab.get(c, self.clinical_unk_id) for c in codes]

            # Calculate patient age in years at each event (bounded [0, 119])
            b_dt = self.patient_birthdates.get(pid)
            if b_dt is not None:
                diff_sec = (dts - b_dt).astype("timedelta64[s]").astype(float)
                ages = np.clip(np.floor(diff_sec / (365.25 * 86400.0)).astype(int), 0, 119)
                age_ids = ages.tolist()
            else:
                age_ids = [0] * len(token_ids)

            # Truncate to max_seq_len (keep most recent events)
            if len(token_ids) > self.max_seq_len:
                token_ids = token_ids[-self.max_seq_len:]
                age_ids = age_ids[-self.max_seq_len:]

            timelines[pid] = (token_ids, age_ids)

        logger.info(f"Constructed clinical timelines with age_ids for {len(timelines):,d} patients.")
        return timelines

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_fn(
    batch: List[Dict[str, Any]],
    pad_id: int = 0,
) -> Union[Tuple[torch.Tensor, ...], Dict[str, Any]]:
    """
    Collate function returning exact signatures per requested baseline mode:

      - For 'clinical':
          returns (input_ids, attention_mask, age_ids, labels)

      - For 'genomic':
          returns (genomic_ids, genomic_mask, labels)

      - For 'multimodal':
          returns aligned dictionary:
          {
              'input_ids': torch.LongTensor,
              'attention_mask': torch.BoolTensor,
              'age_ids': torch.LongTensor,
              'genomic_ids': torch.LongTensor,
              'genomic_mask': torch.BoolTensor,
              'has_genomics': torch.FloatTensor,
              'labels': torch.FloatTensor,
              'patient_ids': List[str]
          }
    """
    batch_size = len(batch)
    if batch_size == 0:
        return ()

    mode = batch[0].get("mode", "multimodal")
    labels = torch.stack([item["labels"] for item in batch], dim=0)

    # 1. Mode 'clinical'
    if mode == "clinical":
        seq_lengths = [len(item["input_ids"]) for item in batch]
        max_len = max(max(seq_lengths), 1)

        input_ids = torch.full((batch_size, max_len), fill_value=pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        age_ids = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, item in enumerate(batch):
            l = len(item["input_ids"])
            if l > 0:
                input_ids[i, :l] = item["input_ids"]
                attention_mask[i, :l] = True
                age_ids[i, :l] = item["age_ids"]

        return (input_ids, attention_mask, age_ids, labels)

    # 2. Mode 'genomic'
    elif mode == "genomic":
        genomic_ids = torch.stack([item["genomic_ids"] for item in batch], dim=0)
        genomic_mask = torch.stack([item["genomic_mask"] for item in batch], dim=0)
        return (genomic_ids, genomic_mask, labels)

    # 3. Mode 'multimodal'
    else:
        seq_lengths = [len(item["input_ids"]) for item in batch]
        max_len = max(max(seq_lengths), 1)

        input_ids = torch.full((batch_size, max_len), fill_value=pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        age_ids = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, item in enumerate(batch):
            l = len(item["input_ids"])
            if l > 0:
                input_ids[i, :l] = item["input_ids"]
                attention_mask[i, :l] = True
                age_ids[i, :l] = item["age_ids"]

        genomic_ids = torch.stack([item["genomic_ids"] for item in batch], dim=0)
        genomic_mask = torch.stack([item["genomic_mask"] for item in batch], dim=0)
        has_genomics = torch.stack([item["has_genomics"] for item in batch], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "age_ids": age_ids,
            "genomic_ids": genomic_ids,
            "genomic_mask": genomic_mask,
            "has_genomics": has_genomics,
            "labels": labels,
            "patient_ids": [item["patient_id"] for item in batch],
        }


def main():
    parser = argparse.ArgumentParser(description="Demonstrate EHRDataset and collate_fn across 3 baseline modes.")
    parser.add_argument(
        "--mode",
        type=str,
        default="multimodal",
        choices=["clinical", "genomic", "multimodal"],
        help="Dataset mode: clinical, genomic, or multimodal (default: multimodal).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to load (default: train).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print(f"TESTING EHRDATASET (mode='{args.mode}', split='{args.split}')")
    print("=" * 72)

    dataset = EHRDataset(
        mode=args.mode,
        split=args.split,
        max_seq_len=256,
    )

    print(f"Total samples loaded: {len(dataset):,d}")
    sample_0 = dataset[0]
    print("\n[Sample 0 Metadata]:")
    print(f"  Patient ID: {sample_0['patient_id']}")
    for k, v in sample_0.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:<16}: shape={tuple(v.shape)}, dtype={v.dtype}")

    # Test DataLoader & collate_fn
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    print(f"\n[Collate Output Inspection for mode='{args.mode}']:")
    if isinstance(batch, tuple):
        print(f"  Return type: Tuple of {len(batch)} tensors")
        if args.mode == "clinical":
            input_ids, attention_mask, age_ids, labels = batch
            print(f"  input_ids:      shape={tuple(input_ids.shape)}, dtype={input_ids.dtype}")
            print(f"  attention_mask: shape={tuple(attention_mask.shape)}, dtype={attention_mask.dtype}")
            print(f"  age_ids:        shape={tuple(age_ids.shape)}, dtype={age_ids.dtype}")
            print(f"  labels:         shape={tuple(labels.shape)}, dtype={labels.dtype}")
        elif args.mode == "genomic":
            genomic_ids, genomic_mask, labels = batch
            print(f"  genomic_ids:    shape={tuple(genomic_ids.shape)}, dtype={genomic_ids.dtype}")
            print(f"  genomic_mask:   shape={tuple(genomic_mask.shape)}, dtype={genomic_mask.dtype}")
            print(f"  labels:         shape={tuple(labels.shape)}, dtype={labels.dtype}")
    elif isinstance(batch, dict):
        print("  Return type: Aligned Dictionary")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k:<16}: shape={tuple(v.shape)}, dtype={v.dtype}")
            elif isinstance(v, list):
                print(f"  {k:<16}: list of {len(v)} items")

    print(f"\n[OK] Mode '{args.mode}' passed all integrity assertions!")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
