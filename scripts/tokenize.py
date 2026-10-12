"""Tokenization Script

Builds patient timelines, performs continuous lab quantile binning,
calculates temporal age embeddings, builds a token vocabulary,
computes multi-label future condition targets, and exports PyTorch-ready datasets.
"""

from __future__ import annotations

import sys

# Prevent shadowing of Python's built-in standard library `tokenize` module
_scripts_dir = sys.path[0] if sys.path else None
if sys.path and ("scripts" in sys.path[0] or sys.path[0].replace("\\", "/").rstrip("/").endswith("/scripts")):
    del sys.path[0]
import tokenize as _stdlib_tokenize

if _scripts_dir and _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import argparse
import json
import logging
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tokenize")

# Special token constants
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"
PAD_ID = 0
UNK_ID = 1
EOS_ID = 2


def fit_observation_quantiles(obs_train_path: Path) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Fits quantile thresholds (Q1-Q5) for numeric observations on training data.

    Returns:
        thresholds_dict: Mapping of CODE -> array of 4 quintile cutoffs [q0.2, q0.4, q0.6, q0.8]
        binned_df: Processed observation DataFrame with discrete 'TOKEN' column
    """
    logger.info(f"Loading observations from {obs_train_path} to fit quantile thresholds...")
    obs = pd.read_csv(obs_train_path)

    thresholds_dict: dict[str, np.ndarray] = {}
    tokens = pd.Series(index=obs.index, dtype=object)

    # 1. Numeric observations: calculate quintile boundaries
    num_mask = obs["TYPE"] == "numeric"
    num_df = obs[num_mask].copy()
    num_df["VAL_FLOAT"] = pd.to_numeric(num_df["VALUE"], errors="coerce")

    for code, group in num_df.groupby("CODE"):
        vals = group["VAL_FLOAT"].dropna().to_numpy()
        if len(vals) == 0:
            thresholds_dict[str(code)] = np.array([0.0, 0.0, 0.0, 0.0])
            continue
        # 20th, 40th, 60th, 80th percentiles
        q_cutoffs = np.quantile(vals, [0.2, 0.4, 0.6, 0.8])
        thresholds_dict[str(code)] = q_cutoffs
        # Bin into 1..5 using right=True
        bin_indices = np.digitize(group["VAL_FLOAT"], q_cutoffs, right=True) + 1
        bin_indices = np.clip(bin_indices, 1, 5)
        tokens.loc[group.index] = [f"{code}_Q{b}" for b in bin_indices]

    # 2. Text observations: append text value
    text_mask = obs["TYPE"] == "text"
    text_df = obs[text_mask]
    text_tokens = text_df["CODE"].astype(str) + "_" + text_df["VALUE"].astype(str).str.strip()
    tokens.loc[text_mask] = text_tokens

    # Fallback for any unhandled rows
    unhandled = tokens.isna()
    if unhandled.any():
        tokens.loc[unhandled] = obs.loc[unhandled, "CODE"].astype(str)

    obs["TOKEN"] = tokens
    logger.info(
        f"Fitted quantile thresholds for {len(thresholds_dict)} numeric codes across {len(obs):,} observations."
    )
    return thresholds_dict, obs


def transform_observations(obs_path: Path, thresholds_dict: dict[str, np.ndarray]) -> pd.DataFrame:
    """Transforms an observations table using pre-fitted quantile thresholds."""
    logger.info(f"Transforming observations from {obs_path} using fitted thresholds...")
    obs = pd.read_csv(obs_path)
    tokens = pd.Series(index=obs.index, dtype=object)

    num_mask = obs["TYPE"] == "numeric"
    num_df = obs[num_mask].copy()
    num_df["VAL_FLOAT"] = pd.to_numeric(num_df["VALUE"], errors="coerce")

    for code, group in num_df.groupby("CODE"):
        code_str = str(code)
        if code_str in thresholds_dict:
            q_cutoffs = thresholds_dict[code_str]
            bin_indices = np.digitize(group["VAL_FLOAT"], q_cutoffs, right=True) + 1
            bin_indices = np.clip(bin_indices, 1, 5)
            tokens.loc[group.index] = [f"{code}_Q{b}" for b in bin_indices]
        else:
            tokens.loc[group.index] = f"{code}_Q3"

    text_mask = obs["TYPE"] == "text"
    text_df = obs[text_mask]
    text_tokens = text_df["CODE"].astype(str) + "_" + text_df["VALUE"].astype(str).str.strip()
    tokens.loc[text_mask] = text_tokens

    unhandled = tokens.isna()
    if unhandled.any():
        tokens.loc[unhandled] = obs.loc[unhandled, "CODE"].astype(str)

    obs["TOKEN"] = tokens
    return obs


def extract_events_from_directory(
    data_dir: Path,
    binned_obs: pd.DataFrame | None = None,
    thresholds_dict: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Extracts all clinical events from a directory into a unified event table.

    Each event record has: ['patient_id', 'timestamp', 'token']
    """
    event_dfs = []

    # 1. Observations
    if binned_obs is not None:
        obs_df = binned_obs[["PATIENT", "DATE", "TOKEN"]].copy()
        obs_df.rename(columns={"PATIENT": "patient_id", "DATE": "timestamp", "TOKEN": "token"}, inplace=True)
        event_dfs.append(obs_df)
    elif (data_dir / "observations.csv").exists():
        assert thresholds_dict is not None, "Thresholds dict required to transform observations"
        obs = transform_observations(data_dir / "observations.csv", thresholds_dict)
        obs_df = obs[["PATIENT", "DATE", "TOKEN"]].copy()
        obs_df.rename(columns={"PATIENT": "patient_id", "DATE": "timestamp", "TOKEN": "token"}, inplace=True)
        event_dfs.append(obs_df)

    # 2. Standard clinical tables with CODE column
    standard_tables = [
        ("conditions.csv", "START"),
        ("medications.csv", "START"),
        ("procedures.csv", "DATE"),
        ("immunizations.csv", "DATE"),
        ("careplans.csv", "START"),
        ("allergies.csv", "START"),
        ("devices.csv", "START"),
        ("encounters.csv", "START"),
    ]

    for fname, date_col in standard_tables:
        fpath = data_dir / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, usecols=["PATIENT", date_col, "CODE"])
        df["token"] = df["CODE"].dropna().astype(str)
        df_clean = df[["PATIENT", date_col, "token"]].copy()
        df_clean.rename(columns={"PATIENT": "patient_id", date_col: "timestamp"}, inplace=True)
        event_dfs.append(df_clean)

    # 3. Imaging studies table
    imaging_path = data_dir / "imaging_studies.csv"
    if imaging_path.exists():
        img = pd.read_csv(imaging_path)
        img["token"] = "IMG_" + img["MODALITY_CODE"].astype(str) + "_" + img["BODYSITE_CODE"].astype(str)
        img_clean = img[["PATIENT", "DATE", "token"]].copy()
        img_clean.rename(columns={"PATIENT": "patient_id", "DATE": "timestamp"}, inplace=True)
        event_dfs.append(img_clean)

    combined = pd.concat(event_dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined):,} total events from {data_dir.name}")
    return combined


def load_birthdates(pts_paths: list[Path]) -> dict[str, pd.Timestamp]:
    """Loads patient birthdates across provided patients.csv files."""
    birth_dict = {}
    for p in pts_paths:
        if not p.exists():
            continue
        df = pd.read_csv(p, usecols=["Id", "BIRTHDATE"])
        for _, row in df.iterrows():
            birth_dict[str(row["Id"])] = pd.to_datetime(row["BIRTHDATE"], utc=True)
    logger.info(f"Loaded birth dates for {len(birth_dict):,} patients.")
    return birth_dict


def compute_target_vectors(
    full_conditions_path: Path,
    anchors_path: Path,
    target_conditions_path: Path,
    target_patients: set[str],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Calculates ground-truth multi-label target condition vectors.

    A condition is positive iff its first-ever diagnosis occurs in
    [anchor_date, anchor_date + 5 years] (conditions diagnosed prior to anchor do NOT count).
    """
    logger.info("Computing multi-label ground-truth target vectors...")
    targets_df = pd.read_csv(target_conditions_path)
    target_codes = targets_df["CODE"].astype(str).tolist()
    target_code_to_idx = {code: i for i, code in enumerate(target_codes)}
    num_targets = len(target_codes)

    # Load anchors
    anchors_df = pd.read_csv(anchors_path)
    id_col = "patient_id" if "patient_id" in anchors_df.columns else "Id"
    anchors_map = pd.to_datetime(anchors_df.set_index(id_col)["anchor_date"], utc=True).to_dict()

    # Load full conditions
    cond_df = pd.read_csv(full_conditions_path, usecols=["PATIENT", "START", "CODE"])
    cond_df["START_dt"] = pd.to_datetime(cond_df["START"], format="ISO8601", utc=True)
    cond_df["CODE_str"] = cond_df["CODE"].astype(str)

    # Filter to patients and target codes
    cond_df = cond_df[
        cond_df["PATIENT"].isin(target_patients) & cond_df["CODE_str"].isin(target_code_to_idx)
    ].copy()

    # Determine pre-anchor and post-anchor diagnoses
    cond_df["anchor_dt"] = cond_df["PATIENT"].map(anchors_map)

    # Pre-anchor conditions (patient, code)
    pre_mask = cond_df["START_dt"] < cond_df["anchor_dt"]
    pre_existing = set(zip(cond_df.loc[pre_mask, "PATIENT"], cond_df.loc[pre_mask, "CODE_str"]))

    # Post-anchor conditions in the 5-year window
    five_years_post = cond_df["anchor_dt"] + pd.DateOffset(years=5)
    post_mask = (cond_df["START_dt"] >= cond_df["anchor_dt"]) & (cond_df["START_dt"] <= five_years_post)
    post_diag = set(zip(cond_df.loc[post_mask, "PATIENT"], cond_df.loc[post_mask, "CODE_str"]))

    # Newly diagnosed in window
    newly_diagnosed = post_diag - pre_existing

    target_vectors: dict[str, np.ndarray] = {}
    for p in target_patients:
        vec = np.zeros(num_targets, dtype=np.float32)
        target_vectors[p] = vec

    for p, code in newly_diagnosed:
        if p in target_vectors and code in target_code_to_idx:
            idx = target_code_to_idx[code]
            target_vectors[p][idx] = 1.0

    logger.info(
        f"Computed ground-truth target vectors for {len(target_vectors):,} patients. "
        f"Total positive labels: {len(newly_diagnosed):,}"
    )
    return target_vectors, target_codes


def build_vocabulary(tokens_list: list[str]) -> dict[str, int]:
    """Builds a deterministic vocabulary dictionary reserving special tokens:

    0: <PAD>, 1: <UNK>, 2: <EOS>
    """
    vocab = {
        PAD_TOKEN: PAD_ID,
        UNK_TOKEN: UNK_ID,
        EOS_TOKEN: EOS_ID,
    }

    unique_tokens = sorted(set(tokens_list))
    for t in unique_tokens:
        if t not in vocab:
            vocab[t] = len(vocab)

    logger.info(f"Built vocabulary containing {len(vocab):,} tokens (including special tokens).")
    return vocab


def main():
    parser = argparse.ArgumentParser(
        description="Build patient timelines, compute age IDs, build vocab, and tokenize dataset."
    )
    parser.add_argument(
        "--truncated_dir",
        type=Path,
        default=Path("truncated_train_val"),
        help="Path to truncated train/val directory (default: truncated_train_val)",
    )
    parser.add_argument(
        "--test_dir",
        type=Path,
        default=Path("test"),
        help="Path to test directory (default: test)",
    )
    parser.add_argument(
        "--raw_train_dir",
        type=Path,
        default=Path("train_val"),
        help="Path to full train_val directory (for ground-truth targets) (default: train_val)",
    )
    parser.add_argument(
        "--splits_path",
        type=Path,
        default=Path("data/patient_splits.csv"),
        help="Path to patient_splits.csv (default: data/patient_splits.csv)",
    )
    parser.add_argument(
        "--targets_path",
        type=Path,
        default=Path("data/target_conditions.csv"),
        help="Path to target_conditions.csv (default: data/target_conditions.csv)",
    )
    parser.add_argument(
        "--train_anchors_path",
        type=Path,
        default=Path("data/train_anchors.csv"),
        help="Path to train_anchors.csv (default: data/train_anchors.csv)",
    )
    parser.add_argument(
        "--test_anchors_path",
        type=Path,
        default=Path("data/test_anchors.csv"),
        help="Path to test_anchors.csv (default: data/test_anchors.csv)",
    )
    parser.add_argument(
        "--vocab_path",
        type=Path,
        default=Path("data/vocab.json"),
        help="Output path for vocabulary JSON (default: data/vocab.json)",
    )
    parser.add_argument(
        "--output_pt",
        type=Path,
        default=Path("data/tokenized_data.pt"),
        help="Output path for PyTorch-ready dataset (default: data/tokenized_data.pt)",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("data/tokenized_data.json"),
        help="Output path for JSON dataset (default: data/tokenized_data.json)",
    )
    parser.add_argument(
        "--append_eos",
        action="store_true",
        default=True,
        help="Append <EOS> token at the end of each patient sequence (default: True)",
    )

    args = parser.parse_args()

    # Resolve input paths with fallback to data/ directory
    for attr in ["splits_path", "targets_path", "train_anchors_path", "test_anchors_path"]:
        p = getattr(args, attr)
        if not p.exists():
            for candidate in [
                Path("data") / p.name,
                Path(__file__).resolve().parent.parent / "data" / p.name,
            ]:
                if candidate.exists():
                    setattr(args, attr, candidate)
                    break

    t_start = time.time()

    # Step 1: Load patient splits
    splits_df = pd.read_csv(args.splits_path)
    split_map = dict(zip(splits_df["Id"], splits_df["split"]))
    train_val_patients = set(splits_df[splits_df["split"].isin(["train", "val"])]["Id"])
    test_patients = set(splits_df[splits_df["split"] == "test"]["Id"])
    all_patients = set(splits_df["Id"])
    logger.info(
        f"Total patients in splits: {len(all_patients):,} "
        f"(Train: {(splits_df['split'] == 'train').sum()}, "
        f"Val: {(splits_df['split'] == 'val').sum()}, "
        f"Test: {(splits_df['split'] == 'test').sum()})"
    )

    # Step 2: Continuous lab quantile binning on training observations
    thresholds_dict, binned_train_obs = fit_observation_quantiles(
        args.truncated_dir / "observations.csv"
    )

    # Step 3: Extract events from truncated train/val and test directories
    train_val_events = extract_events_from_directory(
        args.truncated_dir, binned_obs=binned_train_obs
    )
    test_events = extract_events_from_directory(
        args.test_dir, thresholds_dict=thresholds_dict
    )
    all_events = pd.concat([train_val_events, test_events], ignore_index=True)
    logger.info(f"Total events across all cohorts: {len(all_events):,}")

    # Step 4: Build vocabulary
    vocab = build_vocabulary(all_events["token"].tolist())
    args.vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    logger.info(f"Exported vocabulary to {args.vocab_path}")

    # Step 5: Load birthdates and anchor dates
    birth_dict = load_birthdates([
        args.truncated_dir / "patients.csv",
        args.test_dir / "patients.csv",
    ])

    anchors_dict: dict[str, pd.Timestamp] = {}
    if args.train_anchors_path.exists():
        df_anc = pd.read_csv(args.train_anchors_path)
        id_c = "patient_id" if "patient_id" in df_anc.columns else "Id"
        for _, row in df_anc.iterrows():
            anchors_dict[str(row[id_c])] = pd.to_datetime(row["anchor_date"], utc=True)
    if args.test_anchors_path.exists():
        df_anc = pd.read_csv(args.test_anchors_path)
        id_c = "Id" if "Id" in df_anc.columns else "patient_id"
        for _, row in df_anc.iterrows():
            anchors_dict[str(row[id_c])] = pd.to_datetime(row["anchor_date"], utc=True)

    # Step 6: Compute ground-truth target vectors for train and val patients
    target_vectors, target_codes = compute_target_vectors(
        full_conditions_path=args.raw_train_dir / "conditions.csv",
        anchors_path=args.train_anchors_path,
        target_conditions_path=args.targets_path,
        target_patients=train_val_patients,
    )
    # Test target vectors: zeros (withheld)
    num_targets = len(target_codes)
    for p in test_patients:
        target_vectors[p] = np.zeros(num_targets, dtype=np.float32)

    # Step 7: Build chronological sequence for each patient
    logger.info("Building chronological event sequences and temporal age embeddings...")
    all_events["timestamp_dt"] = pd.to_datetime(all_events["timestamp"], format="ISO8601", utc=True)

    # Sort deterministically by patient, timestamp, and token
    all_events.sort_values(by=["patient_id", "timestamp_dt", "token"], inplace=True)

    # Vectorized token ID mapping
    all_events["token_id"] = all_events["token"].map(vocab).fillna(UNK_ID).astype(int)

    # Vectorized age ID calculation
    birth_series = all_events["patient_id"].map(birth_dict)
    diff_days = (all_events["timestamp_dt"] - birth_series).dt.total_seconds() / 86400.0
    age_calc = np.floor(diff_days / 365.25).fillna(0)
    all_events["age_id"] = np.clip(age_calc, 0, 120).astype(int)

    logger.info("Grouping event tokens and ages per patient...")
    grouped_tokens = all_events.groupby("patient_id")["token_id"].apply(list).to_dict()
    grouped_ages = all_events.groupby("patient_id")["age_id"].apply(list).to_dict()

    dataset_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    seq_lengths = []

    for p in sorted(all_patients):
        split = split_map.get(p, "train")
        birth_dt = birth_dict.get(p)
        anchor_dt = anchors_dict.get(p)

        input_tokens = grouped_tokens.get(p, [])
        age_ids = grouped_ages.get(p, [])

        # Append <EOS> at the end with age at anchor date
        if args.append_eos:
            input_tokens = list(input_tokens) + [EOS_ID]
            if birth_dt is not None and anchor_dt is not None:
                diff_days_anc = (anchor_dt - birth_dt).total_seconds() / 86400.0
                anchor_age = max(0, min(120, int(np.floor(diff_days_anc / 365.25))))
            elif len(age_ids) > 0:
                anchor_age = age_ids[-1]
            else:
                anchor_age = 0
            age_ids = list(age_ids) + [anchor_age]

        target_vec = target_vectors.get(p, np.zeros(num_targets, dtype=np.float32))
        seq_lengths.append(len(input_tokens))

        patient_record = {
            "patient_id": p,
            "input_ids": input_tokens,
            "age_ids": age_ids,
            "target": target_vec.tolist(),
        }
        dataset_by_split[split].append(patient_record)

    # Step 8: Export datasets
    logger.info("Saving tokenized datasets...")

    # PyTorch .pt format
    pt_dataset = {
        split: [
            {
                "patient_id": rec["patient_id"],
                "input_ids": torch.tensor(rec["input_ids"], dtype=torch.long),
                "age_ids": torch.tensor(rec["age_ids"], dtype=torch.long),
                "target": torch.tensor(rec["target"], dtype=torch.float32),
            }
            for rec in records
        ]
        for split, records in dataset_by_split.items()
    }
    pt_dataset["metadata"] = {
        "vocab_size": len(vocab),
        "target_conditions": target_codes,
        "pad_id": PAD_ID,
        "unk_id": UNK_ID,
        "eos_id": EOS_ID,
        "split_counts": {k: len(v) for k, v in dataset_by_split.items()},
    }

    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pt_dataset, args.output_pt)
    logger.info(f"Saved PyTorch dataset to {args.output_pt}")

    # JSON format
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": pt_dataset["metadata"],
                "data": dataset_by_split,
            },
            f,
        )
    logger.info(f"Saved JSON dataset to {args.output_json}")

    total_time = time.time() - t_start

    # Execution summary report
    seq_arr = np.array(seq_lengths)
    print("\n" + "=" * 80)
    print("TOKENIZATION SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Patients Tokenized : {len(all_patients):,}")
    print(f"  - Train Cohort         : {len(dataset_by_split['train']):,}")
    print(f"  - Val Cohort           : {len(dataset_by_split['val']):,}")
    print(f"  - Test Cohort          : {len(dataset_by_split['test']):,}")
    print(f"Vocabulary Size          : {len(vocab):,} unique tokens (PAD=0, UNK=1, EOS=2)")
    print(f"Target Conditions        : {num_targets} binary labels per patient")
    print("-" * 80)
    print("Sequence Length Distribution (including <EOS>):")
    print(f"  - Min Length           : {int(np.min(seq_arr))}")
    print(f"  - 25th Percentile      : {int(np.percentile(seq_arr, 25))}")
    print(f"  - Median (50th)        : {int(np.median(seq_arr))}")
    print(f"  - 75th Percentile      : {int(np.percentile(seq_arr, 75))}")
    print(f"  - 95th Percentile      : {int(np.percentile(seq_arr, 95))}")
    print(f"  - Max Length           : {int(np.max(seq_arr))}")
    print("-" * 80)
    train_labels = np.array([r["target"] for r in dataset_by_split["train"]])
    val_labels = np.array([r["target"] for r in dataset_by_split["val"]])
    print("Label Statistics:")
    print(f"  - Train Positive Labels: {int(np.sum(train_labels)):,} (mean {np.mean(train_labels)*100:.2f}% prevalence)")
    print(f"  - Val Positive Labels  : {int(np.sum(val_labels)):,} (mean {np.mean(val_labels)*100:.2f}% prevalence)")
    print("=" * 80)
    print(f"Execution completed in {total_time:.2f} seconds.\n")


if __name__ == "__main__":
    main()
