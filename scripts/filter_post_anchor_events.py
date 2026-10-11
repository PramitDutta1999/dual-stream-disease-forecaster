"""
filter_post_anchor_events.py

End-to-end Python pipeline to:
1. Calculate anchor dates from encounters.csv (train/val patients).
2. Filter out post-anchor events across clinical tables to prevent data leakage.
3. (Optional) Extract ground-truth multi-label targets for future condition onset.

Task Specification:
- Landmark / Anchor: Date 5 years before each patient's last recorded encounter in encounters.csv
  (latest STOP, or START if STOP is blank).
- Input timeline: Events strictly before the anchor date/time.
- Prediction window: 5 years following the anchor date (conditions newly diagnosed in that window).
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def calculate_anchor_dates(
    encounters_df: Union[str, pd.DataFrame],
    output_csv: Optional[str] = None,
    save_format: str = "date",
) -> pd.DataFrame:
    """
    Calculate anchor dates from encounters.csv for each patient.

    Definition:
      Anchor date = latest encounter date/time minus 5 years.
      Latest encounter end = STOP (or START if STOP is blank).

    Args:
        encounters_df: Path to encounters.csv or loaded pd.DataFrame.
        output_csv: Optional path to save calculated anchors.
        save_format: 'date' for YYYY-MM-DD (matches test_anchors.csv) or 'all' for full metadata.

    Returns:
        pd.DataFrame with columns:
          - Id: patient ID
          - anchor_date: YYYY-MM-DD string
          - anchor_timestamp: UTC Timestamp
          - last_encounter: UTC Timestamp of patient's last recorded encounter
    """
    if isinstance(encounters_df, str):
        logger.info(f"Loading encounters from {encounters_df}...")
        df = pd.read_csv(encounters_df, low_memory=False)
    else:
        df = encounters_df.copy()

    patient_col = "PATIENT" if "PATIENT" in df.columns else "Id"
    if patient_col not in df.columns:
        raise ValueError("Patient column ('PATIENT' or 'Id') not found in encounters data.")

    # Parse timestamps in UTC
    start_dt = pd.to_datetime(df["START"], utc=True)
    if "STOP" in df.columns:
        stop_dt = pd.to_datetime(df["STOP"], utc=True)
        end_dt = stop_dt.fillna(start_dt)
    else:
        end_dt = start_dt

    df_ends = pd.DataFrame({
        "Id": df[patient_col],
        "end_dt": end_dt
    })

    # Group by patient to find the latest encounter end
    logger.info("Computing latest encounter per patient...")
    patient_max = df_ends.groupby("Id", as_index=False)["end_dt"].max()
    patient_max.rename(columns={"end_dt": "last_encounter"}, inplace=True)

    # Subtract 5 years using calendar DateOffset (handles leap years correctly)
    logger.info("Computing anchor date (latest encounter - 5 years)...")
    patient_max["anchor_timestamp"] = patient_max["last_encounter"] - pd.DateOffset(years=5)
    patient_max["anchor_date"] = patient_max["anchor_timestamp"].dt.strftime("%Y-%m-%d")

    # Format output DataFrame
    result = patient_max[["Id", "anchor_date", "anchor_timestamp", "last_encounter"]].sort_values("Id").reset_index(drop=True)

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        # Matches test_anchors.csv standard: ['Id', 'anchor_date']
        out_df = result[["Id", "anchor_date"]] if save_format == "date" else result
        out_df.to_csv(output_csv, index=False)
        logger.info(f"Saved {len(result)} anchor dates to {output_csv}")

    return result


def filter_post_anchor_events(
    df: pd.DataFrame,
    anchors: Union[Dict[str, pd.Timestamp], Dict[str, str], pd.DataFrame],
    table_name: str = "",
    use_timestamp: bool = True,
    strict_intervals: bool = True,
    mask_future_stop: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Filter out post-anchor events for a clinical table.

    Args:
        df: Clinical DataFrame (e.g. encounters, conditions, medications, observations, etc.).
        anchors: Either dict of {patient_id: anchor_timestamp/date} or DataFrame with 'Id' and 'anchor_timestamp' / 'anchor_date'.
        table_name: Optional name for logging and specific rules (e.g. 'patients.csv').
        use_timestamp: If True, uses exact UTC timestamp comparison; if False, uses date strings (YYYY-MM-DD).
        strict_intervals: For tables with START & STOP (e.g. encounters):
                          If True, encounters extending past the anchor are excluded to prevent leakage.
                          If False, encounters starting before anchor are kept.
        mask_future_stop: For ongoing conditions/medications (START < anchor but STOP > anchor),
                          if True, retains the record but masks STOP = NaN (as done in test set)
                          to avoid leaking the future resolution date.

    Returns:
        (filtered_df, stats_dict)
    """
    total_rows = len(df)
    if total_rows == 0:
        return df.copy(), {"total": 0, "kept": 0, "dropped": 0}

    # Identify patient column
    patient_col = "PATIENT" if "PATIENT" in df.columns else ("Id" if "Id" in df.columns else None)
    if patient_col is None:
        logger.warning(f"No PATIENT or Id column found in {table_name}; returning unfiltered.")
        return df.copy(), {"total": total_rows, "kept": total_rows, "dropped": 0}

    # Convert anchors to dict if DataFrame
    if isinstance(anchors, pd.DataFrame):
        if use_timestamp and "anchor_timestamp" in anchors.columns:
            anchor_map = dict(zip(anchors["Id"], pd.to_datetime(anchors["anchor_timestamp"], utc=True)))
        else:
            anchor_map = dict(zip(anchors["Id"], anchors["anchor_date"].astype(str).str[:10]))
    else:
        anchor_map = anchors

    # Special handling for patients demographic table
    if "patients" in table_name.lower():
        out_df = df.copy()
        if "DEATHDATE" in out_df.columns:
            death_dates = pd.to_datetime(out_df["DEATHDATE"], utc=True, errors="coerce")
            patient_anchors = out_df["Id"].map(anchor_map)
            if use_timestamp:
                mask = death_dates > patient_anchors
            else:
                death_str = out_df["DEATHDATE"].astype(str).str[:10]
                mask = death_str >= patient_anchors
            masked_count = int(mask.sum())
            out_df.loc[mask, "DEATHDATE"] = np.nan
            if masked_count > 0:
                logger.info(f"Masked {masked_count} post-anchor DEATHDATEs in {table_name}")
        return out_df, {"total": total_rows, "kept": total_rows, "dropped": 0}

    # Identify primary date column
    date_col = "START" if "START" in df.columns else ("DATE" if "DATE" in df.columns else None)
    if date_col is None:
        logger.warning(f"No date column found in table {table_name}; returning original.")
        return df.copy(), {"total": total_rows, "kept": total_rows, "dropped": 0}

    out_df = df.copy()

    if use_timestamp:
        event_dt = pd.to_datetime(out_df[date_col], utc=True)
        patient_anchor = out_df[patient_col].map(anchor_map)
        
        # Primary filter: event must start before anchor
        valid_mask = event_dt <= patient_anchor

        # For encounters: if strict_intervals, encounter must also end before anchor
        if "STOP" in out_df.columns:
            has_stop = out_df["STOP"].notna()
            stop_dt = pd.to_datetime(out_df.loc[has_stop, "STOP"], utc=True)
            
            if "encounter" in table_name.lower() and strict_intervals:
                stop_after = stop_dt > patient_anchor[has_stop]
                valid_mask.loc[has_stop[has_stop].index[stop_after]] = False
            elif mask_future_stop:
                stop_after_idx = stop_dt[stop_dt > patient_anchor[has_stop]].index
                out_df.loc[stop_after_idx, "STOP"] = np.nan
    else:
        event_date_str = out_df[date_col].astype(str).str[:10]
        patient_anchor_str = out_df[patient_col].map(anchor_map)
        valid_mask = event_date_str < patient_anchor_str

        if "STOP" in out_df.columns:
            has_stop = out_df["STOP"].notna()
            stop_str = out_df.loc[has_stop, "STOP"].astype(str).str[:10]
            if "encounter" in table_name.lower() and strict_intervals:
                stop_after = stop_str >= patient_anchor_str[has_stop]
                valid_mask.loc[has_stop[has_stop].index[stop_after]] = False
            elif mask_future_stop:
                stop_after_idx = stop_str[stop_str >= patient_anchor_str[has_stop]].index
                out_df.loc[stop_after_idx, "STOP"] = np.nan

    filtered_df = out_df[valid_mask].reset_index(drop=True)
    stats = {
        "total": total_rows,
        "kept": len(filtered_df),
        "dropped": total_rows - len(filtered_df),
    }
    return filtered_df, stats


def extract_future_condition_labels(
    conditions_df: Union[str, pd.DataFrame],
    anchors_df: Union[str, pd.DataFrame],
    target_conditions_df: Optional[Union[str, pd.DataFrame]] = None,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build ground-truth binary labels for newly diagnosed conditions in the 5 years post-anchor.

    Per ReadMe.md:
    "Predict which clinical conditions are newly diagnosed in the five years after the anchor.
     A condition counts only if its first-ever diagnosis falls in that window;
     conditions the patient already had before the anchor do not count."

    Args:
        conditions_df: Full conditions.csv (from train_val).
        anchors_df: DataFrame with 'Id' and anchor date/timestamp.
        target_conditions_df: target_conditions.csv or DataFrame containing scored condition CODEs.
        output_csv: Optional path to save the resulting labels matrix.

    Returns:
        DataFrame with 'PATIENT' as index or column, and binary indicator (0 or 1) for each target condition code.
    """
    if isinstance(conditions_df, str):
        conds = pd.read_csv(conditions_df, low_memory=False)
    else:
        conds = conditions_df.copy()

    if isinstance(anchors_df, str):
        anchors = pd.read_csv(anchors_df)
    else:
        anchors = anchors_df.copy()

    # Parse anchor timestamp
    if "anchor_timestamp" in anchors.columns:
        anchors["anchor_dt"] = pd.to_datetime(anchors["anchor_timestamp"], utc=True)
    else:
        anchors["anchor_dt"] = pd.to_datetime(anchors["anchor_date"], utc=True)

    anchors["future_window_end"] = anchors["anchor_dt"] + pd.DateOffset(years=5)
    anchor_dict = dict(zip(anchors["Id"], anchors["anchor_dt"]))
    window_end_dict = dict(zip(anchors["Id"], anchors["future_window_end"]))

    # Parse condition dates and find first-ever diagnosis date per (PATIENT, CODE)
    conds["START_dt"] = pd.to_datetime(conds["START"], utc=True)
    first_diag = conds.groupby(["PATIENT", "CODE"], as_index=False)["START_dt"].min()
    first_diag.rename(columns={"START_dt": "first_onset"}, inplace=True)

    # Attach anchor and window end
    first_diag["anchor_dt"] = first_diag["PATIENT"].map(anchor_dict)
    first_diag["future_window_end"] = first_diag["PATIENT"].map(window_end_dict)

    # Condition counts only if first-ever diagnosis falls in [anchor, anchor + 5 years]
    # (i.e. first_onset >= anchor and first_onset <= anchor + 5 years)
    new_in_window = first_diag[
        (first_diag["first_onset"] >= first_diag["anchor_dt"]) &
        (first_diag["first_onset"] <= first_diag["future_window_end"])
    ]

    # Filter to target conditions if provided
    if target_conditions_df is not None:
        if isinstance(target_conditions_df, str):
            target_df = pd.read_csv(target_conditions_df)
        else:
            target_df = target_conditions_df
        target_codes = [str(c) for c in target_df["CODE"].unique()]
    else:
        target_codes = [str(c) for c in new_in_window["CODE"].unique()]

    new_in_window = new_in_window.copy()
    new_in_window["CODE_str"] = new_in_window["CODE"].astype(str)
    new_in_window = new_in_window[new_in_window["CODE_str"].isin(target_codes)]

    # Pivot to patient x condition binary matrix
    pivoted = pd.crosstab(new_in_window["PATIENT"], new_in_window["CODE_str"]).clip(upper=1)

    # Ensure all target patients and target condition columns are present
    all_patients = sorted(anchors["Id"].unique())
    labels_df = pd.DataFrame(0, index=all_patients, columns=target_codes, dtype=int)
    labels_df.index.name = "PATIENT"

    for col in pivoted.columns:
        if col in labels_df.columns:
            labels_df.loc[pivoted.index, col] = pivoted[col]

    labels_df.reset_index(inplace=True)

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        labels_df.to_csv(output_csv, index=False)
        logger.info(f"Saved future labels matrix ({labels_df.shape[0]} patients, {len(target_codes)} targets) to {output_csv}")

    return labels_df


def process_directory(
    data_dir: str = "train_val",
    output_dir: str = "train_val_pre_anchor",
    anchors_csv: Optional[str] = "data/train_val_anchors.csv",
    tables_to_filter: Optional[List[str]] = None,
    use_timestamp: bool = True,
    strict_intervals: bool = True,
    mask_future_stop: bool = True,
    build_targets: bool = False,
    target_conditions_csv: Optional[str] = "data/target_conditions.csv",
    targets_output_csv: Optional[str] = "data/train_val_targets.csv",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Process all tables in data_dir:
    1. Calculate anchors from encounters.csv
    2. Save anchors to anchors_csv
    3. Filter out post-anchor events from all specified tables
    4. Save truncated tables to output_dir
    5. Optionally extract future condition target labels
    """
    encounters_path = os.path.join(data_dir, "encounters.csv")
    if not os.path.exists(encounters_path):
        raise FileNotFoundError(f"encounters.csv not found in {data_dir}")

    # 1. Calculate anchor dates
    anchors_df = calculate_anchor_dates(
        encounters_path,
        output_csv=anchors_csv,
        save_format="date",
    )

    anchor_time_map = dict(zip(anchors_df["Id"], anchors_df["anchor_timestamp"]))
    anchor_date_map = dict(zip(anchors_df["Id"], anchors_df["anchor_date"]))
    anchors_ref = anchor_time_map if use_timestamp else anchor_date_map

    # 2. Determine which tables to filter
    os.makedirs(output_dir, exist_ok=True)
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    if tables_to_filter:
        files_to_process = [f for f in all_files if f in tables_to_filter]
    else:
        files_to_process = all_files

    all_stats = {}
    print("\n" + "=" * 72)
    print(f"{'TABLE':<22} | {'ORIGINAL':>10} | {'PRE-ANCHOR':>10} | {'DROPPED':>10} | {'KEPT %':>7}")
    print("=" * 72)

    for filename in files_to_process:
        input_path = os.path.join(data_dir, filename)
        output_path = os.path.join(output_dir, filename)

        table_df = pd.read_csv(input_path, low_memory=False)
        filtered_df, stats = filter_post_anchor_events(
            table_df,
            anchors=anchors_ref,
            table_name=filename,
            use_timestamp=use_timestamp,
            strict_intervals=strict_intervals,
            mask_future_stop=mask_future_stop,
        )

        filtered_df.to_csv(output_path, index=False)
        all_stats[filename] = stats

        pct = (stats["kept"] / stats["total"] * 100) if stats["total"] > 0 else 100.0
        print(f"{filename:<22} | {stats['total']:>10,d} | {stats['kept']:>10,d} | {stats['dropped']:>10,d} | {pct:>6.1f}%")

    print("=" * 72 + "\n")
    logger.info(f"All filtered tables saved to directory: {output_dir}")

    # 3. Optional: build target labels
    if build_targets:
        conditions_path = os.path.join(data_dir, "conditions.csv")
        if os.path.exists(conditions_path):
            logger.info("Extracting ground-truth target labels for post-anchor conditions...")
            extract_future_condition_labels(
                conditions_df=conditions_path,
                anchors_df=anchors_df,
                target_conditions_df=target_conditions_csv if (target_conditions_csv and os.path.exists(target_conditions_csv)) else None,
                output_csv=targets_output_csv,
            )

    return anchors_df, all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Calculate anchor dates from encounters.csv and filter post-anchor events."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="train_val",
        help="Directory containing the input CSV tables (default: train_val).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="train_val_pre_anchor",
        help="Directory to save the pre-anchor filtered CSV tables (default: train_val_pre_anchor).",
    )
    parser.add_argument(
        "--anchors-output",
        type=str,
        default="data/train_val_anchors.csv",
        help="Path to save calculated patient anchors (default: data/train_val_anchors.csv).",
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        default=None,
        help="Specific table(s) to filter (e.g. encounters.csv conditions.csv). Default: all CSV tables.",
    )
    parser.add_argument(
        "--use-date-only",
        action="store_true",
        help="Filter strictly by calendar date (YYYY-MM-DD) instead of UTC timestamp.",
    )
    parser.add_argument(
        "--allow-straddling-encounters",
        action="store_true",
        help="Allow encounters that started before anchor but ended after anchor to be kept.",
    )
    parser.add_argument(
        "--build-targets",
        action="store_true",
        help="Extract binary ground-truth labels for newly diagnosed target conditions post-anchor.",
    )
    parser.add_argument(
        "--target-conditions",
        type=str,
        default="data/target_conditions.csv",
        help="Path to target_conditions.csv (default: data/target_conditions.csv).",
    )
    parser.add_argument(
        "--targets-output",
        type=str,
        default="data/train_val_targets.csv",
        help="Path to save ground-truth labels matrix (default: data/train_val_targets.csv).",
    )

    args = parser.parse_args()

    process_directory(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        anchors_csv=args.anchors_output,
        tables_to_filter=args.tables,
        use_timestamp=not args.use_date_only,
        strict_intervals=not args.allow_straddling_encounters,
        mask_future_stop=True,
        build_targets=args.build_targets,
        target_conditions_csv=args.target_conditions,
        targets_output_csv=args.targets_output,
    )


if __name__ == "__main__":
    main()