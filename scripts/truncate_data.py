"""Data Truncation Script

Calculates 5-year pre-encounter anchor dates for each patient and truncates
all clinical tables in train_val/ to prevent data leakage into the future.
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
import logging
from pathlib import Path
import shutil
import time
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("truncate_data")

CLINICAL_FILES = [
    "conditions.csv",
    "medications.csv",
    "procedures.csv",
    "observations.csv",
    "immunizations.csv",
    "careplans.csv",
    "allergies.csv",
    "devices.csv",
    "imaging_studies.csv",
]


def compute_anchor_dates(encounters_path: Path, output_anchors_path: Path) -> pd.DataFrame:
    """Computes anchor dates per patient from encounters.csv and exports them.

    Anchor date = (latest STOP date, or START if STOP is blank) - 5 years.
    Format: YYYY-MM-DD
    """
    logger.info(f"Loading encounters from {encounters_path}...")
    enc = pd.read_csv(encounters_path, usecols=["PATIENT", "START", "STOP"])

    # Fallback to START if STOP is missing or empty
    stop_series = enc["STOP"].fillna("").astype(str).str.strip()
    end_time = stop_series.replace("", pd.NA).fillna(enc["START"])

    logger.info("Converting encounter end timestamps to UTC datetime...")
    enc["end_dt"] = pd.to_datetime(end_time, utc=True)

    logger.info("Finding latest encounter per patient...")
    latest_enc = enc.groupby("PATIENT")["end_dt"].max().reset_index()

    logger.info("Subtracting exactly 5 years to derive anchor date...")
    latest_enc["anchor_dt"] = latest_enc["end_dt"] - pd.DateOffset(years=5)
    latest_enc["anchor_date"] = latest_enc["anchor_dt"].dt.strftime("%Y-%m-%d")

    anchors_df = latest_enc[["PATIENT", "anchor_date"]].copy()
    anchors_df.rename(columns={"PATIENT": "patient_id"}, inplace=True)

    output_anchors_path.parent.mkdir(parents=True, exist_ok=True)
    anchors_df.to_csv(output_anchors_path, index=False)
    logger.info(
        f"Exported {len(anchors_df):,} patient anchor dates to {output_anchors_path}"
    )

    return anchors_df


def get_patient_column(df: pd.DataFrame) -> str:
    """Finds the patient identifier column in a table."""
    for col in ["PATIENT", "patient_id", "Id"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not identify patient column. Columns: {list(df.columns)}")


def get_date_column(df: pd.DataFrame) -> str:
    """Finds the event date column in a table."""
    for col in ["START", "DATE"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not identify date column. Columns: {list(df.columns)}")


def truncate_file(
    input_file: Path,
    output_file: Path,
    anchor_map: pd.Series,
) -> dict:
    """Truncates a single clinical table by dropping rows with event date >= anchor date."""
    logger.info(f"Processing {input_file.name}...")
    t0 = time.time()
    df = pd.read_csv(input_file)
    orig_count = len(df)

    if orig_count == 0:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        return {
            "table": input_file.name,
            "orig_count": 0,
            "trunc_count": 0,
            "dropped_count": 0,
            "pct_kept": 100.0,
            "elapsed_s": time.time() - t0,
        }

    patient_col = get_patient_column(df)
    date_col = get_date_column(df)

    # Convert event date to UTC datetime
    event_dt = pd.to_datetime(df[date_col], utc=True)
    patient_anchors = df[patient_col].map(anchor_map)

    # Ensure all patients in the table have an anchor date
    missing_anchors = patient_anchors.isna().sum()
    if missing_anchors > 0:
        raise ValueError(
            f"Table {input_file.name} contains {missing_anchors} rows with unknown patient anchors!"
        )

    anchor_dt = pd.to_datetime(patient_anchors, utc=True)

    # Strictly retain event date < anchor date (drop >= anchor date)
    keep_mask = event_dt < anchor_dt
    filtered_df = df[keep_mask]
    trunc_count = len(filtered_df)
    dropped_count = orig_count - trunc_count
    pct_kept = (trunc_count / orig_count * 100.0) if orig_count > 0 else 0.0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_file, index=False)

    elapsed_s = time.time() - t0
    logger.info(
        f"Saved {output_file.name}: {trunc_count:,}/{orig_count:,} kept "
        f"({pct_kept:.2f}%), {dropped_count:,} dropped in {elapsed_s:.2f}s"
    )

    return {
        "table": input_file.name,
        "orig_count": orig_count,
        "trunc_count": trunc_count,
        "dropped_count": dropped_count,
        "pct_kept": pct_kept,
        "elapsed_s": elapsed_s,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate patient anchor dates and truncate clinical tables."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("train_val"),
        help="Path to original data directory (default: train_val)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("truncated_train_val"),
        help="Path to output truncated directory (default: truncated_train_val)",
    )
    parser.add_argument(
        "--anchors_path",
        type=Path,
        default=Path("data/train_anchors.csv"),
        help="Path to export computed anchor dates (default: data/train_anchors.csv)",
    )
    parser.add_argument(
        "--include_encounters",
        action="store_true",
        default=True,
        help="Also truncate encounters.csv into output directory (default: True)",
    )
    parser.add_argument(
        "--copy_patients",
        action="store_true",
        default=True,
        help="Also copy patients.csv demographics to output directory (default: True)",
    )

    args = parser.parse_args()

    encounters_path = args.data_dir / "encounters.csv"
    if not encounters_path.exists():
        logger.error(f"Encounters file not found at {encounters_path}")
        sys.exit(1)

    # Step 1: Compute anchors
    t_start = time.time()
    anchors_df = compute_anchor_dates(encounters_path, args.anchors_path)
    anchor_map = anchors_df.set_index("patient_id")["anchor_date"]

    # Step 2: Prepare target file list
    target_files = list(CLINICAL_FILES)
    if args.include_encounters and "encounters.csv" not in target_files:
        target_files.append("encounters.csv")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: Truncate each clinical file
    results = []
    for file_name in target_files:
        input_path = args.data_dir / file_name
        output_path = args.output_dir / file_name
        if not input_path.exists():
            logger.warning(f"File {input_path} does not exist, skipping.")
            continue
        stats = truncate_file(input_path, output_path, anchor_map)
        results.append(stats)

    # Step 4: Copy static patients.csv demographics if present
    patients_file = args.data_dir / "patients.csv"
    if args.copy_patients and patients_file.exists():
        dest_patients = args.output_dir / "patients.csv"
        shutil.copy2(patients_file, dest_patients)
        logger.info(f"Copied static demographics {patients_file.name} to {dest_patients}")

    total_time = time.time() - t_start

    # Summary report
    print("\n" + "=" * 80)
    print("DATA TRUNCATION SUMMARY")
    print("=" * 80)
    print(
        f"{'Table':<22} | {'Original Rows':>14} | {'Truncated Rows':>14} | {'Dropped':>10} | {'% Kept':>8}"
    )
    print("-" * 80)
    tot_orig = 0
    tot_trunc = 0
    tot_dropped = 0
    for r in results:
        tot_orig += r["orig_count"]
        tot_trunc += r["trunc_count"]
        tot_dropped += r["dropped_count"]
        print(
            f"{r['table']:<22} | {r['orig_count']:>14,d} | {r['trunc_count']:>14,d} | "
            f"{r['dropped_count']:>10,d} | {r['pct_kept']:>7.2f}%"
        )
    print("-" * 80)
    overall_pct = (tot_trunc / tot_orig * 100.0) if tot_orig > 0 else 0.0
    print(
        f"{'TOTAL':<22} | {tot_orig:>14,d} | {tot_trunc:>14,d} | "
        f"{tot_dropped:>10,d} | {overall_pct:>7.2f}%"
    )
    print("=" * 80)
    print(f"Total processing completed in {total_time:.2f} seconds.\n")


if __name__ == "__main__":
    main()

