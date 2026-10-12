"""Leakage Verification Script

Verifies data truncation by comparing row counts between original and truncated
clinical tables and strictly asserting zero dates occur on or after patient anchor dates.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("check_leakage")

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


def load_anchors(anchors_path: Path) -> pd.Series:
    """Loads anchor dates mapping: patient_id -> anchor_date."""
    if not anchors_path.exists():
        fallback_candidates = [
            Path("data") / anchors_path.name,
            Path(__file__).resolve().parent.parent / "data" / anchors_path.name,
            Path(__file__).resolve().parent / "data" / anchors_path.name,
        ]
        for fb in fallback_candidates:
            if fb.exists():
                anchors_path = fb
                break
        else:
            raise FileNotFoundError(f"Anchors file not found: {anchors_path}")

    anchors_df = pd.read_csv(anchors_path)
    id_col = None
    for candidate in ["patient_id", "Id", "PATIENT"]:
        if candidate in anchors_df.columns:
            id_col = candidate
            break

    if id_col is None:
        raise ValueError(
            f"Anchor file {anchors_path} must contain patient ID column. Found: {list(anchors_df.columns)}"
        )

    if "anchor_date" not in anchors_df.columns:
        raise ValueError(
            f"Anchor file {anchors_path} must contain 'anchor_date' column. Found: {list(anchors_df.columns)}"
        )

    return anchors_df.set_index(id_col)["anchor_date"]


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


def verify_table(
    table_name: str,
    orig_path: Path,
    trunc_path: Path,
    anchor_map: pd.Series,
) -> dict:
    """Verifies a single table for row counts, retention percentage, and leakage."""
    if not orig_path.exists():
        raise FileNotFoundError(f"Original file not found: {orig_path}")
    if not trunc_path.exists():
        raise FileNotFoundError(f"Truncated file not found: {trunc_path}")

    orig_df = pd.read_csv(orig_path)
    trunc_df = pd.read_csv(trunc_path)

    orig_count = len(orig_df)
    trunc_count = len(trunc_df)
    dropped_count = orig_count - trunc_count
    pct_kept = (trunc_count / orig_count * 100.0) if orig_count > 0 else 0.0

    if trunc_count == 0:
        return {
            "table": table_name,
            "orig_count": orig_count,
            "trunc_count": 0,
            "dropped_count": dropped_count,
            "pct_kept": pct_kept,
            "violations": 0,
            "status": "PASS",
        }

    patient_col = get_patient_column(trunc_df)
    date_col = get_date_column(trunc_df)

    event_dt = pd.to_datetime(trunc_df[date_col], utc=True)
    patient_anchors = trunc_df[patient_col].map(anchor_map)

    # Check for missing anchors
    missing_anchors = patient_anchors.isna().sum()
    if missing_anchors > 0:
        raise AssertionError(
            f"Table {table_name} has {missing_anchors} records with no matching anchor date!"
        )

    anchor_dt = pd.to_datetime(patient_anchors, utc=True)

    # Leakage condition: event_date >= anchor_date
    leak_mask = event_dt >= anchor_dt
    violation_count = int(leak_mask.sum())

    # STRICT ASSERTION CHECK
    if violation_count > 0:
        violating_rows = trunc_df[leak_mask][[patient_col, date_col]].copy()
        violating_rows["anchor_date"] = patient_anchors[leak_mask]
        sample_violations = violating_rows.head(10).to_dict(orient="records")
        raise AssertionError(
            f"DATA LEAKAGE DETECTED in {table_name}! "
            f"{violation_count} row(s) have event date >= anchor date.\n"
            f"Sample violations:\n{sample_violations}"
        )

    assert violation_count == 0, f"Violation count in {table_name} must be 0, got {violation_count}"

    return {
        "table": table_name,
        "orig_count": orig_count,
        "trunc_count": trunc_count,
        "dropped_count": dropped_count,
        "pct_kept": pct_kept,
        "violations": 0,
        "status": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify truncated clinical data against patient anchor dates."
    )
    parser.add_argument(
        "--orig_dir",
        type=Path,
        default=Path("train_val"),
        help="Path to original train_val directory (default: train_val)",
    )
    parser.add_argument(
        "--truncated_dir",
        type=Path,
        default=Path("truncated_train_val"),
        help="Path to truncated directory (default: truncated_train_val)",
    )
    def resolve_anchors_path(filename: str = "train_anchors.csv") -> Path:
        p_data = Path("data") / filename
        if p_data.exists():
            return p_data
        p_root = Path(__file__).resolve().parent.parent / "data" / filename
        if p_root.exists():
            return p_root
        return Path(filename)

    parser.add_argument(
        "--anchors_path",
        type=Path,
        default=resolve_anchors_path("train_anchors.csv"),
        help="Path to computed anchor dates CSV (default: data/train_anchors.csv)",
    )

    args = parser.parse_args()

    logger.info("Loading patient anchor dates...")
    anchor_map = load_anchors(args.anchors_path)
    logger.info(f"Loaded {len(anchor_map):,} patient anchors.")

    # Determine files to verify: check clinical files + encounters if present
    tables_to_check = list(CLINICAL_FILES)
    if (args.truncated_dir / "encounters.csv").exists():
        tables_to_check.append("encounters.csv")

    results = []
    has_failure = False

    for table_name in tables_to_check:
        orig_file = args.orig_dir / table_name
        trunc_file = args.truncated_dir / table_name
        if not trunc_file.exists():
            logger.warning(f"File {trunc_file} not found in truncated directory. Skipping.")
            continue

        try:
            res = verify_table(table_name, orig_file, trunc_file, anchor_map)
            results.append(res)
        except AssertionError as e:
            logger.error(str(e))
            results.append({
                "table": table_name,
                "orig_count": -1,
                "trunc_count": -1,
                "dropped_count": -1,
                "pct_kept": 0.0,
                "violations": -1,
                "status": "FAIL",
            })
            has_failure = True

    # Terminal report output
    print("\n" + "=" * 90)
    print("LEAKAGE VERIFICATION REPORT")
    print("=" * 90)
    print(
        f"{'Table Name':<22} | {'Original':>11} | {'Truncated':>11} | {'Dropped':>9} | {'% Kept':>7} | {'Violations':>10} | {'Status':>6}"
    )
    print("-" * 90)

    tot_orig = 0
    tot_trunc = 0
    tot_dropped = 0
    tot_violations = 0

    for r in results:
        if r["status"] == "PASS":
            tot_orig += r["orig_count"]
            tot_trunc += r["trunc_count"]
            tot_dropped += r["dropped_count"]
            tot_violations += r["violations"]
            print(
                f"{r['table']:<22} | {r['orig_count']:>11,d} | {r['trunc_count']:>11,d} | "
                f"{r['dropped_count']:>9,d} | {r['pct_kept']:>6.2f}% | {r['violations']:>10d} | {r['status']:>6}"
            )
        else:
            print(
                f"{r['table']:<22} | {'ERROR':>11} | {'ERROR':>11} | "
                f"{'ERROR':>9} | {'ERROR':>7} | {'ERROR':>10} | {r['status']:>6}"
            )

    print("-" * 90)
    if not has_failure and tot_orig > 0:
        overall_pct = tot_trunc / tot_orig * 100.0
        print(
            f"{'TOTAL / OVERALL':<22} | {tot_orig:>11,d} | {tot_trunc:>11,d} | "
            f"{tot_dropped:>9,d} | {overall_pct:>6.2f}% | {tot_violations:>10d} | {'PASS':>6}"
        )
    print("=" * 90)

    if has_failure:
        print("\n[RESULT] VERIFICATION FAILED: Data leakage or errors detected!\n")
        sys.exit(1)
    else:
        print("\n[RESULT] VERIFICATION PASSED: Absolutely zero data leakage detected across all tables.")
        print("All event dates strictly precede each patient's 5-year anchor date.\n")


if __name__ == "__main__":
    main()

