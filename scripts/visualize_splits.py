"""Data Split Verification and Visualization Script

Cross-references patient_splits.csv against tokenized_data.pt,
identifies any missing or dropped patients per split,
prints a verified summary report to the terminal,
and generates an expected vs actual distribution bar chart saved to reports/.
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def load_splits(splits_path: Path) -> dict[str, list[str]]:
    """Loads patient_splits.csv and returns dictionary mapping split -> list of patient IDs."""
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")

    df = pd.read_csv(splits_path)
    splits = {}
    for s in ["train", "val", "test"]:
        splits[s] = df[df["split"] == s]["Id"].astype(str).tolist()
    return splits


def load_tokenized_patient_ids(data_path: Path) -> dict[str, list[str]]:
    """Loads tokenized dataset and extracts patient IDs per split."""
    if not data_path.exists():
        raise FileNotFoundError(f"Tokenized data file not found: {data_path}")

    raw_data = torch.load(data_path)
    actual_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for s in ["train", "val", "test"]:
        if s in raw_data and isinstance(raw_data[s], list):
            actual_ids[s] = [str(item["patient_id"]) for item in raw_data[s]]
        elif s in raw_data and isinstance(raw_data[s], dict):
            actual_ids[s] = [str(item["patient_id"]) for item in raw_data[s].values()]
        elif "data" in raw_data and s in raw_data["data"]:
            actual_ids[s] = [str(item["patient_id"]) for item in raw_data["data"][s]]

    return actual_ids


def plot_split_distribution(
    comparison_data: list[dict],
    output_image_path: Path,
):
    """Generates and saves a clean, publication-ready bar chart comparing expected vs actual patient counts across splits."""
    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    splits = [d["split"].capitalize() for d in comparison_data]
    expected_counts = [d["expected"] for d in comparison_data]
    actual_counts = [d["actual"] for d in comparison_data]

    x = np.arange(len(splits))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # Styling colors
    color_expected = "#4A90E2"  # Soft corporate blue
    color_actual = "#2ECC71"    # Vibrant emerald green

    rects1 = ax.bar(x - bar_width / 2, expected_counts, bar_width, label="Expected (patient_splits.csv)", color=color_expected, edgecolor="#2C3E50", linewidth=1.2)
    rects2 = ax.bar(x + bar_width / 2, actual_counts, bar_width, label="Actual (tokenized_data.pt)", color=color_actual, edgecolor="#2C3E50", linewidth=1.2)

    ax.set_ylabel("Patient Count", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("EHR Patient Cohort Distribution: Expected vs. Actual Counts", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#BDC3C7", fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Set appropriate y-axis limit
    max_val = max(max(expected_counts), max(actual_counts))
    ax.set_ylim(0, max_val * 1.15)

    # Add count labels on top of each bar
    def autolabel(rects, is_actual=False):
        for idx, rect in enumerate(rects):
            height = rect.get_height()
            comp = comparison_data[idx]
            if is_actual and comp["missing"] > 0:
                txt = f"{height:,}\n(-{comp['missing']})"
                ax.annotate(
                    txt,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="#C0392B",
                )
            else:
                pct = comp["retention_pct"]
                txt = f"{height:,}\n({pct:.1f}%)" if is_actual else f"{height:,}"
                ax.annotate(
                    txt,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="#2C3E50",
                )

    autolabel(rects1, is_actual=False)
    autolabel(rects2, is_actual=True)

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Verify patient cohort splits and generate distribution visualization."
    )
    parser.add_argument(
        "--splits_path",
        type=Path,
        default=Path("data/patient_splits.csv"),
        help="Path to patient_splits.csv (default: data/patient_splits.csv)",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("data/tokenized_data.pt"),
        help="Path to tokenized_data.pt (default: data/tokenized_data.pt)",
    )
    parser.add_argument(
        "--output_plot",
        type=Path,
        default=Path("reports/patient_split_verification.png"),
        help="Output image path for visualization (default: reports/patient_split_verification.png)",
    )

    args = parser.parse_args()

    for attr in ["splits_path", "data_path"]:
        p = getattr(args, attr)
        if not p.exists():
            for candidate in [
                Path("data") / p.name,
                Path(__file__).resolve().parent.parent / "data" / p.name,
            ]:
                if candidate.exists():
                    setattr(args, attr, candidate)
                    break

    # Step 1: Load expected splits
    expected_splits = load_splits(args.splits_path)

    # Step 2: Load actual tokenized patient IDs
    actual_splits = load_tokenized_patient_ids(args.data_path)

    # Step 3: Cross-reference and identify missing patients
    comparison_data = []
    tot_expected = 0
    tot_actual = 0
    tot_missing = 0

    for s in ["train", "val", "test"]:
        exp_set = set(expected_splits.get(s, []))
        act_set = set(actual_splits.get(s, []))

        exp_count = len(exp_set)
        act_count = len(act_set)
        missing_pids = exp_set - act_set
        unexpected_pids = act_set - exp_set

        missing_count = len(missing_pids)
        retention_pct = (act_count / exp_count * 100.0) if exp_count > 0 else 0.0

        status = "MATCH" if missing_count == 0 and len(unexpected_pids) == 0 else "MISMATCH"

        comparison_data.append({
            "split": s,
            "expected": exp_count,
            "actual": act_count,
            "missing": missing_count,
            "unexpected": len(unexpected_pids),
            "retention_pct": retention_pct,
            "status": status,
            "missing_pids": sorted(list(missing_pids)),
        })

        tot_expected += exp_count
        tot_actual += act_count
        tot_missing += missing_count

    # Step 4: Terminal Summary Report
    print("\n" + "=" * 80)
    print("PATIENT DATA SPLIT VERIFICATION REPORT")
    print("=" * 80)
    print(
        f"{'Cohort Split':<14} | {'Expected':>10} | {'Actual':>10} | {'Missing':>9} | {'% Retention':>11} | {'Status':>8}"
    )
    print("-" * 80)

    for item in comparison_data:
        print(
            f"{item['split'].capitalize():<14} | {item['expected']:>10,d} | {item['actual']:>10,d} | "
            f"{item['missing']:>9,d} | {item['retention_pct']:>10.2f}% | {item['status']:>8}"
        )

    print("-" * 80)
    tot_retention = (tot_actual / tot_expected * 100.0) if tot_expected > 0 else 0.0
    overall_status = "MATCH" if tot_missing == 0 else "MISMATCH"
    print(
        f"{'TOTAL':<14} | {tot_expected:>10,d} | {tot_actual:>10,d} | "
        f"{tot_missing:>9,d} | {tot_retention:>10.2f}% | {overall_status:>8}"
    )
    print("=" * 80)

    # Check for missing patients and print specifics
    if tot_missing > 0:
        print("\n[WARNING] Detected missing patients:")
        for item in comparison_data:
            if item["missing"] > 0:
                print(f"  - {item['split'].capitalize()} missing {item['missing']} patients: {item['missing_pids'][:5]}")
    else:
        print("\n[VERIFIED] 100% of expected patients are present in the tokenized dataset.")
        print("Zero patients dropped across train, val, and test splits.")

    # Step 5: Generate and save visualization
    plot_split_distribution(comparison_data, args.output_plot)
    print(f"\n[VISUALIZATION] Saved verification plot to: {args.output_plot}\n")


if __name__ == "__main__":
    main()

