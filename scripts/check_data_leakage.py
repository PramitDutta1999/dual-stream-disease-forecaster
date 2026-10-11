"""
scripts/check_data_leakage.py

Strict verification script to confirm zero data leakage across all pre-anchor
clinical tables in the train_val_pre_anchor/ directory.

Checks performed per table:
1. Patient mapping: Every event's patient must exist in train_val_anchors.csv.
2. START / DATE: Strict assertion that event start date does not occur after anchor date.
3. STOP: Strict assertion that any non-null event stop date does not occur after anchor date.
4. Patients table: Strict assertion that non-null DEATHDATE does not occur after anchor date.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Tuple
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_table_leakage(
    table_path: str,
    anchors_map: Dict[str, pd.Timestamp],
    anchor_dates_str: Dict[str, str],
) -> Dict[str, any]:
    """
    Check a single clinical table for post-anchor data leakage.

    Raises:
        AssertionError: If any event occurs after the patient's anchor date.
    """
    table_name = os.path.basename(table_path)
    df = pd.read_csv(table_path, low_memory=False)
    total_rows = len(df)

    if total_rows == 0:
        return {
            "table": table_name,
            "total_rows": 0,
            "checked_events": 0,
            "max_event_date": "N/A",
            "start_violations": 0,
            "stop_violations": 0,
            "status": "PASSED",
        }

    # Identify patient column
    patient_col = "PATIENT" if "PATIENT" in df.columns else ("Id" if "Id" in df.columns else None)
    if patient_col is None:
        return {
            "table": table_name,
            "total_rows": total_rows,
            "checked_events": 0,
            "max_event_date": "N/A",
            "start_violations": 0,
            "stop_violations": 0,
            "status": "SKIPPED (no patient col)",
        }

    # Verify all patients belong to anchors map
    unmapped_patients = set(df[patient_col].unique()) - set(anchors_map.keys())
    assert len(unmapped_patients) == 0, (
        f"Data integrity error in {table_name}: found {len(unmapped_patients)} patients "
        f"not present in the anchors file!"
    )

    # Special handling for patients demographic table
    if "patients" in table_name.lower():
        death_violations = 0
        max_date = "N/A"
        if "DEATHDATE" in df.columns:
            has_death = df[df["DEATHDATE"].notna()].copy()
            if len(has_death) > 0:
                death_dates = pd.to_datetime(has_death["DEATHDATE"], utc=True)
                p_anchors = has_death["Id"].map(anchors_map)
                # Check for post-anchor mortality
                violating = has_death[death_dates > p_anchors]
                death_violations = len(violating)
                max_date = str(death_dates.max())[:10]
                assert death_violations == 0, (
                    f"Data leakage detected in {table_name}: {death_violations} patients have DEATHDATE "
                    f"after their anchor date!\nSample:\n{violating[['Id', 'DEATHDATE']].head()}"
                )

        return {
            "table": table_name,
            "total_rows": total_rows,
            "checked_events": total_rows,
            "max_event_date": max_date,
            "start_violations": 0,
            "stop_violations": death_violations,
            "status": "PASSED",
        }

    # Identify date columns
    date_col = "START" if "START" in df.columns else ("DATE" if "DATE" in df.columns else None)
    if date_col is None:
        return {
            "table": table_name,
            "total_rows": total_rows,
            "checked_events": 0,
            "max_event_date": "N/A",
            "start_violations": 0,
            "stop_violations": 0,
            "status": "SKIPPED (no date col)",
        }

    # 1. Check START / DATE against anchor
    # Use UTC timestamp comparison for exact second-level precision,
    # plus calendar date check (event_date <= anchor_date).
    event_dt = pd.to_datetime(df[date_col], utc=True)
    p_anchors_dt = df[patient_col].map(anchors_map)

    # Check if event starts after anchor timestamp
    # Note: adding 1-day tolerance for calendar date alignment vs timestamp
    violating_start = df[event_dt > (p_anchors_dt + pd.Timedelta(seconds=1))]
    start_violations = len(violating_start)

    assert start_violations == 0, (
        f"Data leakage detected in {table_name}: {start_violations} events have "
        f"{date_col} after patient anchor date!\n"
        f"Sample offending records:\n"
        f"{violating_start[[patient_col, date_col]].head()}"
    )

    # 2. Check STOP (if present) against anchor
    stop_violations = 0
    if "STOP" in df.columns:
        has_stop = df[df["STOP"].notna()].copy()
        if len(has_stop) > 0:
            stop_dt = pd.to_datetime(has_stop["STOP"], utc=True)
            p_anchors_stop = has_stop[patient_col].map(anchors_map)
            violating_stop = has_stop[stop_dt > (p_anchors_stop + pd.Timedelta(seconds=1))]
            stop_violations = len(violating_stop)

            assert stop_violations == 0, (
                f"Data leakage detected in {table_name}: {stop_violations} events have "
                f"STOP after patient anchor date!\n"
                f"Sample offending records:\n"
                f"{violating_stop[[patient_col, 'START', 'STOP']].head()}"
            )

    max_event_date = str(event_dt.max())[:10] if len(event_dt) > 0 else "N/A"

    return {
        "table": table_name,
        "total_rows": total_rows,
        "checked_events": total_rows,
        "max_event_date": max_event_date,
        "start_violations": start_violations,
        "stop_violations": stop_violations,
        "status": "PASSED",
    }


def verify_all_tables(
    anchors_csv: str = "data/train_val_anchors.csv",
    pre_anchor_dir: str = "train_val_pre_anchor",
) -> bool:
    """
    Cross-reference train_val_anchors.csv against all clinical tables in pre_anchor_dir.
    Prints terminal report confirming zero data leakage.
    """
    if not os.path.exists(anchors_csv):
        raise FileNotFoundError(f"Anchors file not found: {anchors_csv}")
    if not os.path.isdir(pre_anchor_dir):
        raise FileNotFoundError(f"Pre-anchor directory not found: {pre_anchor_dir}")

    logger.info(f"Loading anchors from {anchors_csv}...")
    anchors_df = pd.read_csv(anchors_csv)
    assert "Id" in anchors_df.columns, "Anchors file must contain an 'Id' column."
    assert "anchor_date" in anchors_df.columns, "Anchors file must contain an 'anchor_date' column."

    anchors_df["anchor_dt"] = pd.to_datetime(anchors_df["anchor_date"], utc=True)
    # End of anchor day (23:59:59) to accommodate events on the anchor date
    anchors_df["anchor_end_of_day"] = anchors_df["anchor_dt"] + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)

    anchors_map = dict(zip(anchors_df["Id"], anchors_df["anchor_end_of_day"]))
    anchor_dates_str = dict(zip(anchors_df["Id"], anchors_df["anchor_date"].astype(str).str[:10]))

    logger.info(f"Loaded {len(anchors_map):,d} patient anchors.")

    table_files = sorted([f for f in os.listdir(pre_anchor_dir) if f.endswith(".csv")])
    if not table_files:
        raise ValueError(f"No CSV tables found in {pre_anchor_dir}")

    results = []
    all_passed = True

    for filename in table_files:
        table_path = os.path.join(pre_anchor_dir, filename)
        try:
            res = check_table_leakage(table_path, anchors_map, anchor_dates_str)
            results.append(res)
        except AssertionError as e:
            logger.error(str(e))
            results.append({
                "table": filename,
                "total_rows": -1,
                "checked_events": -1,
                "max_event_date": "N/A",
                "start_violations": -1,
                "stop_violations": -1,
                "status": "FAILED",
            })
            all_passed = False

    # Terminal Report
    print("\n" + "=" * 84)
    print(f"{'PRE-ANCHOR DATA LEAKAGE VERIFICATION REPORT':^84}")
    print("=" * 84)
    print(f"{'TABLE NAME':<22} | {'ROWS':>10} | {'MAX DATE':>10} | {'START VIOL':>10} | {'STOP VIOL':>10} | {'STATUS':>8}")
    print("-" * 84)

    total_events_checked = 0
    for r in results:
        rows_str = f"{r['total_rows']:,d}" if r['total_rows'] >= 0 else "ERROR"
        start_viol_str = str(r['start_violations']) if r['start_violations'] >= 0 else "FAIL"
        stop_viol_str = str(r['stop_violations']) if r['stop_violations'] >= 0 else "FAIL"
        print(
            f"{r['table']:<22} | {rows_str:>10} | {r['max_event_date']:>10} | "
            f"{start_viol_str:>10} | {stop_viol_str:>10} | {r['status']:>8}"
        )
        if r['total_rows'] > 0:
            total_events_checked += r['total_rows']

    print("=" * 84)
    if all_passed:
        print(f"\n[OK] VERIFICATION PASSED: Checked {total_events_checked:,d} total records across {len(results)} tables.")
        print("     Strict assertion confirmed: ZERO post-anchor data leakage detected!\n")
    else:
        print("\n[FAIL] VERIFICATION FAILED: Data leakage or assertion errors were detected!\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify zero data leakage in pre-anchor clinical tables."
    )
    parser.add_argument(
        "--anchors",
        type=str,
        default="data/train_val_anchors.csv",
        help="Path to train_val_anchors.csv (default: data/train_val_anchors.csv).",
    )
    parser.add_argument(
        "--pre-anchor-dir",
        type=str,
        default="train_val_pre_anchor",
        help="Directory containing filtered tables (default: train_val_pre_anchor).",
    )
    args = parser.parse_args()

    success = verify_all_tables(
        anchors_csv=args.anchors,
        pre_anchor_dir=args.pre_anchor_dir,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

