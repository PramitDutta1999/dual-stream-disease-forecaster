"""Data Modality Coverage and Cohort Missingness Analysis Script

Calculates patient cohort coverage across data modalities (Structured EHR,
Genomics, and Imaging Studies) to evaluate missingness and justify modality selection.
Generates publication-quality figures saved into the reports/ directory.
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
import re
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_default_project_root() -> Path:
    """Returns the project root directory assuming this script is in scripts/."""
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "scripts":
        return script_dir.parent
    return Path.cwd()


def load_cohort_baseline(splits_path: Path) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """Loads patient_splits.csv and extracts baseline patient IDs.

    Returns:
        total_patients: Set of all unique patient IDs across all splits.
        split_to_patients: Mapping of split name ('train', 'val', 'test') to patient ID sets.
    """
    if not splits_path.exists():
        fallback = splits_path.parent / "data" / splits_path.name
        if fallback.exists():
            splits_path = fallback
        elif (Path("data") / splits_path.name).exists():
            splits_path = Path("data") / splits_path.name
        elif not splits_path.exists():
            raise FileNotFoundError(f"Patient splits file not found: {splits_path}")

    df = pd.read_csv(splits_path)
    if "Id" not in df.columns:
        raise ValueError(f"Expected 'Id' column in {splits_path}, found: {df.columns.tolist()}")

    total_patients = set(df["Id"].dropna().astype(str))
    split_to_patients: Dict[str, Set[str]] = {}

    split_col = "split" if "split" in df.columns else None
    if split_col:
        for split_name, group in df.groupby(split_col):
            split_to_patients[str(split_name)] = set(group["Id"].dropna().astype(str))

    return total_patients, split_to_patients


def analyze_genomics_coverage(
    genomics_dir: Path, baseline_patients: Set[str]
) -> Tuple[Set[str], Dict[str, float]]:
    """Analyzes genomics directory for patient coverage against baseline."""
    if not genomics_dir.exists():
        print(f"Warning: Genomics directory not found: {genomics_dir}")
        return set(), {"available": 0, "missing": len(baseline_patients), "coverage_pct": 0.0, "missing_pct": 100.0}

    # Extract patient IDs from filenames matching *_dna.csv
    genomic_files = list(genomics_dir.glob("*_dna.csv"))
    extracted_ids: Set[str] = set()

    for file_path in genomic_files:
        filename = file_path.name
        # Match pattern <UUID>_dna.csv
        match = re.match(r"^([0-9a-fA-F\-]+)_dna\.csv$", filename)
        if match:
            extracted_ids.add(match.group(1))
        else:
            # Fallback stem removal
            pid = filename.replace("_dna.csv", "")
            extracted_ids.add(pid)

    # Intersect with baseline cohort
    available_ids = extracted_ids.intersection(baseline_patients)
    total_baseline = len(baseline_patients)
    missing_count = total_baseline - len(available_ids)
    coverage_pct = (len(available_ids) / total_baseline * 100.0) if total_baseline > 0 else 0.0
    missing_pct = (missing_count / total_baseline * 100.0) if total_baseline > 0 else 0.0

    stats = {
        "files_count": len(genomic_files),
        "unique_genomic_ids": len(extracted_ids),
        "available": len(available_ids),
        "missing": missing_count,
        "coverage_pct": coverage_pct,
        "missing_pct": missing_pct,
    }
    return available_ids, stats


def analyze_imaging_coverage(
    imaging_paths: List[Path], baseline_patients: Set[str]
) -> Tuple[Set[str], pd.DataFrame, Dict[str, float]]:
    """Loads imaging study records and calculates patient coverage and modality breakdown."""
    dfs: List[pd.DataFrame] = []

    for path in imaging_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Failed reading imaging file {path}: {e}")

    if not dfs:
        print(f"Warning: No valid imaging files found among provided paths.")
        combined_df = pd.DataFrame(columns=["PATIENT", "MODALITY_DESCRIPTION"])
    else:
        combined_df = pd.concat(dfs, ignore_index=True)

    if "PATIENT" in combined_df.columns:
        imaging_patient_ids = set(combined_df["PATIENT"].dropna().astype(str))
    else:
        imaging_patient_ids = set()

    available_ids = imaging_patient_ids.intersection(baseline_patients)
    total_baseline = len(baseline_patients)
    missing_count = total_baseline - len(available_ids)
    coverage_pct = (len(available_ids) / total_baseline * 100.0) if total_baseline > 0 else 0.0
    missing_pct = (missing_count / total_baseline * 100.0) if total_baseline > 0 else 0.0

    stats = {
        "total_records": len(combined_df),
        "unique_imaging_ids": len(imaging_patient_ids),
        "available": len(available_ids),
        "missing": missing_count,
        "coverage_pct": coverage_pct,
        "missing_pct": missing_pct,
    }
    return available_ids, combined_df, stats


def plot_modality_coverage(
    coverage_data: List[Dict[str, any]],
    total_cohort_size: int,
    output_path: Path,
    dpi: int = 300,
) -> None:
    """Generates an academic bar plot comparing patient cohort availability across modalities."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [item["name"] for item in coverage_data]
    counts = [item["count"] for item in coverage_data]
    percentages = [item["pct"] for item in coverage_data]

    # Academic aesthetic palette
    colors = ["#2B4C7E", "#2A9D8F", "#E76F51"]  # Deep Blue, Sage Teal, Coral Amber

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=dpi)

    x = np.arange(len(labels))
    bar_width = 0.55

    rects = ax.bar(
        x,
        counts,
        bar_width,
        color=colors,
        edgecolor="#1E293B",
        linewidth=1.2,
        zorder=3,
        alpha=0.92,
    )

    # Reference 100% baseline dashed line
    ax.axhline(
        total_cohort_size,
        color="#64748B",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label=f"Baseline Cohort (N = {total_cohort_size:,})",
        zorder=2,
    )

    # Styling
    ax.set_title("Patient Cohort Coverage Across Data Modalities", fontsize=14, fontweight="bold", pad=16, color="#0F172A")
    ax.set_ylabel("Available Patient Count", fontsize=11, fontweight="bold", labelpad=10, color="#1E293B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold", color="#1E293B")
    ax.set_ylim(0, total_cohort_size * 1.18)
    ax.grid(axis="y", linestyle=":", alpha=0.6, color="#94A3B8", zorder=1)
    ax.set_axisbelow(True)

    # Spines styling
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#64748B")
        ax.spines[spine].set_linewidth(1.0)

    # Annotations on top of each bar
    for rect, count, pct in zip(rects, counts, percentages):
        height = rect.get_height()
        missing = total_cohort_size - count
        label_text = f"N = {count:,}\n({pct:.1f}%)"
        if missing > 0:
            label_text += f"\n[-{missing:,} missing]"

        ax.annotate(
            label_text,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#0F172A",
            linespacing=1.2,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        )

    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f" Saved modality coverage plot to: {output_path}")


def plot_imaging_modalities(
    modality_counts: pd.Series,
    total_studies: int,
    output_path: Path,
    dpi: int = 300,
) -> None:
    """Generates a horizontal bar chart showing distribution of imaging modalities."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort descending
    modality_counts = modality_counts.sort_values(ascending=True)
    labels = [str(name) for name in modality_counts.index]
    counts = modality_counts.values
    pcts = (counts / total_studies * 100.0) if total_studies > 0 else np.zeros_like(counts)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)

    y = np.arange(len(labels))
    bar_height = 0.55

    # Gradient-like cohesive color shades
    color_palette = plt.cm.Blues(np.linspace(0.45, 0.85, len(labels)))

    rects = ax.barh(
        y,
        counts,
        bar_height,
        color=color_palette,
        edgecolor="#1E293B",
        linewidth=1.1,
        zorder=3,
        alpha=0.95,
    )

    ax.set_title("Distribution of Recorded Imaging Modalities", fontsize=14, fontweight="bold", pad=16, color="#0F172A")
    ax.set_xlabel("Number of Imaging Studies", fontsize=11, fontweight="bold", labelpad=10, color="#1E293B")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5, fontweight="bold", color="#1E293B")
    ax.set_xlim(0, max(counts) * 1.25)
    ax.grid(axis="x", linestyle=":", alpha=0.6, color="#94A3B8", zorder=1)
    ax.set_axisbelow(True)

    # Spines styling
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#64748B")
        ax.spines[spine].set_linewidth(1.0)

    # Annotate bars with counts and percentages
    for rect, count, pct in zip(rects, counts, pcts):
        width = rect.get_width()
        ax.annotate(
            f"{count:,} ({pct:.1f}%)",
            xy=(width, rect.get_y() + rect.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#0F172A",
        )

    # Add subtitle text indicating total studies and patients
    ax.text(
        0.70,
        0.12,
        f"Total Imaging Studies: {total_studies:,}",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.95),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f" Saved imaging modalities plot to: {output_path}")


def print_console_summary(
    total_cohort_size: int,
    split_counts: Dict[str, int],
    genomics_stats: Dict[str, float],
    imaging_stats: Dict[str, float],
    modality_breakdown: pd.Series,
    both_count: int,
    ehr_only_count: int,
) -> None:
    """Prints a structured, formatted console summary for documentation and terminal inspection."""
    sep = "=" * 70
    sub_sep = "-" * 70

    print("\n" + sep)
    print("        COHORT MODALITY COVERAGE & MISSINGNESS ANALYSIS REPORTS        ")
    print(sep)

    print(f"\n1. BASELINE STRUCTURED EHR COHORT:")
    print(f"   Total Unique Patients:     {total_cohort_size:,} (100.0%)")
    if split_counts:
        split_details = ", ".join([f"{s.capitalize()}: {c:,} ({c/total_cohort_size*100:.1f}%)" for s, c in split_counts.items()])
        print(f"   Cohort Split Breakdown:    {split_details}")

    print(f"\n2. GENOMICS MODALITY (DNA Sequences):")
    print(f"   Patients with Genomics:    {int(genomics_stats['available']):,} ({genomics_stats['coverage_pct']:.2f}%)")
    print(f"   Missing Genomics:          {int(genomics_stats['missing']):,} ({genomics_stats['missing_pct']:.2f}%)")
    print(f"   Total Genomic Files Found: {int(genomics_stats.get('files_count', 0)):,}")

    print(f"\n3. IMAGING MODALITY (imaging_studies.csv):")
    print(f"   Patients with Imaging:     {int(imaging_stats['available']):,} ({imaging_stats['coverage_pct']:.2f}%)")
    print(f"   Missing Imaging:           {int(imaging_stats['missing']):,} ({imaging_stats['missing_pct']:.2f}%)")
    print(f"   Total Imaging Records:     {int(imaging_stats.get('total_records', 0)):,}")

    print(f"\n4. MULTIMODAL OVERLAP ANALYSIS:")
    both_pct = (both_count / total_cohort_size * 100.0) if total_cohort_size > 0 else 0.0
    ehr_only_pct = (ehr_only_count / total_cohort_size * 100.0) if total_cohort_size > 0 else 0.0
    print(f"   Tripartite Multimodal (EHR + Genomics + Imaging): {both_count:,} ({both_pct:.2f}%)")
    print(f"   Structured EHR Only (No Genomics & No Imaging):  {ehr_only_count:,} ({ehr_only_pct:.2f}%)")

    print(f"\n5. IMAGING MODALITY DISTRIBUTION:")
    print(f"   {'Modality Description':<30} {'Studies Count':<15} {'Percentage':<10}")
    print(f"   {'-'*30} {'-'*15} {'-'*10}")
    total_studies = int(imaging_stats.get('total_records', 0))
    for mod_name, count in modality_breakdown.items():
        pct = (count / total_studies * 100.0) if total_studies > 0 else 0.0
        print(f"   {str(mod_name):<30} {count:<15,} {pct:>6.2f}%")

    print("\n" + sub_sep)
    print(" SUMMARY RATIONALE FOR TECHNICAL REPORTS:")
    print(" - Discarding missing modalities (complete-case analysis) would discard")
    print(f"   {total_cohort_size - both_count:,} patients ({(100.0 - both_pct):.2f}% of the full cohort).")
    print(" - Structured EHR represents the universal anchor (100% complete).")
    print(sep + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze patient coverage and missingness across EHR, Genomics, and Imaging modalities."
    )
    project_root = get_default_project_root()

    default_splits = project_root / "data" / "patient_splits.csv"
    if not default_splits.exists():
        default_splits = project_root / "patient_splits.csv"

    parser.add_argument(
        "--splits_path",
        type=Path,
        default=default_splits,
        help="Path to patient_splits.csv (default: data/patient_splits.csv)",
    )
    parser.add_argument(
        "--genomics_dir",
        type=Path,
        default=project_root / "genomics",
        help="Directory containing genomics files (default: genomics/ in project root)",
    )
    parser.add_argument(
        "--imaging_paths",
        type=Path,
        nargs="+",
        default=[
            project_root / "train_val" / "imaging_studies.csv",
            project_root / "test" / "imaging_studies.csv",
        ],
        help="Path(s) to imaging_studies.csv tables",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root / "reports",
        help="Directory where output plots will be saved (default: reports/)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for saved visualizations (default: 300)",
    )

    args = parser.parse_args()

    # 1. Cohort Baseline & Patient Sets
    print(f"Loading cohort baseline from: {args.splits_path}")
    total_baseline_patients, split_to_patients = load_cohort_baseline(args.splits_path)
    split_counts = {s: len(pids) for s, pids in split_to_patients.items()}
    total_cohort_size = len(total_baseline_patients)

    # 2. Genomics Coverage
    print(f"Analyzing genomics coverage from: {args.genomics_dir}")
    genomic_patient_ids, genomics_stats = analyze_genomics_coverage(args.genomics_dir, total_baseline_patients)

    # 3. Imaging Coverage & Modality Distribution
    # Also check if root imaging_studies.csv exists if defaults missing
    imaging_paths = args.imaging_paths
    if not any(p.exists() for p in imaging_paths):
        root_img = project_root / "imaging_studies.csv"
        if root_img.exists():
            imaging_paths = [root_img]

    print(f"Analyzing imaging studies from: {[str(p) for p in imaging_paths if p.exists()]}")
    imaging_patient_ids, imaging_df, imaging_stats = analyze_imaging_coverage(imaging_paths, total_baseline_patients)

    # Modality breakdown
    if "MODALITY_DESCRIPTION" in imaging_df.columns:
        modality_breakdown = imaging_df["MODALITY_DESCRIPTION"].fillna("Unknown").value_counts()
    elif "MODALITY_CODE" in imaging_df.columns:
        modality_breakdown = imaging_df["MODALITY_CODE"].fillna("Unknown").value_counts()
    else:
        modality_breakdown = pd.Series(dtype=int)

    # Overlap metrics
    tripartite_ids = genomic_patient_ids.intersection(imaging_patient_ids)
    ehr_only_ids = total_baseline_patients - genomic_patient_ids - imaging_patient_ids

    # 4. Generate Visualizations (Save to reports/)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coverage_data = [
        {"name": "Structured EHR", "count": total_cohort_size, "pct": 100.0},
        {"name": "Imaging Studies", "count": int(imaging_stats["available"]), "pct": imaging_stats["coverage_pct"]},
        {"name": "Genomics (DNA)", "count": int(genomics_stats["available"]), "pct": genomics_stats["coverage_pct"]},
    ]

    coverage_plot_path = args.output_dir / "modality_coverage.png"
    plot_modality_coverage(coverage_data, total_cohort_size, coverage_plot_path, dpi=args.dpi)

    imaging_plot_path = args.output_dir / "imaging_modalities.png"
    total_studies = int(imaging_stats["total_records"])
    plot_imaging_modalities(modality_breakdown, total_studies, imaging_plot_path, dpi=args.dpi)

    # 5. Terminal Summary
    print_console_summary(
        total_cohort_size=total_cohort_size,
        split_counts=split_counts,
        genomics_stats=genomics_stats,
        imaging_stats=imaging_stats,
        modality_breakdown=modality_breakdown,
        both_count=len(tripartite_ids),
        ehr_only_count=len(ehr_only_ids),
    )


if __name__ == "__main__":
    main()
