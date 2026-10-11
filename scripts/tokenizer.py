"""
scripts/tokenizer.py

Clinical and genomic code tokenizers, vocabulary builders, and distribution visualization
for longitudinal EHR and multimodal ensemble modeling.

Features:
1. Clinical Vocabulary (clinical_vocab.json):
   - Vectorized code extraction across train_val_pre_anchor/ event tables
   - Preserves special tokens: "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3
   - Clinical codes mapped by frequency descending starting from index 4
2. Genomic Vocabulary (genomic_vocab.json):
   - Extracts unique variant loci (INDEX) and clinical significance tokens
   - Preserves special tokens: "<PAD>": 0, "<UNK>": 1, "<MISSING_GENOME>": 2
   - Variant identifiers and significance tokens mapped starting from index 3
3. Summary Visualizations (reports/figures/vocab_distribution.png):
   - Frequency bar chart of top clinical tokens
   - Event count breakdown across clinical domains (conditions, medications, procedures, labs)
   - Frequency distribution of genomic variant categories / clinical significance
"""

import os
import sys
import json
import glob
import argparse
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =========================================================================
# 1. Clinical Vocabulary Generator
# =========================================================================

def normalize_code_series(series: pd.Series) -> pd.Series:
    """Vectorized normalization of clinical codes, removing trailing .0 from float conversions."""
    s = series.astype(str).str.strip()
    invalid = s.isna() | s.isin(["nan", "None", "", "NaN", "null"])
    has_point_zero = s.str.endswith(".0")
    clean_s = s.mask(has_point_zero, s.str[:-2])
    return clean_s.mask(invalid, None)


def build_clinical_vocab(
    data_dir: str = "train_val_pre_anchor",
    output_vocab_path: str = "data/clinical_vocab.json",
    output_metadata_path: Optional[str] = "data/clinical_vocab_metadata.json",
    include_tables: Optional[List[str]] = None,
) -> Tuple[Dict[str, int], Dict[str, dict], Dict[str, int]]:
    """
    Scan pre-anchor clinical CSVs and build clinical vocabulary.

    Special tokens:
      "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3

    Returns:
        (vocab_dict, metadata_dict, domain_counts_dict)
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Clinical data directory not found: {data_dir}")

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    code_counter = Counter()
    code_descriptions = {}
    domain_counts = Counter()

    logger.info(f"Scanning pre-anchor clinical tables in '{data_dir}' for clinical vocabulary...")

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        if "patients" in fname.lower():
            continue
        if include_tables and fname not in include_tables:
            continue

        # Map to domain
        domain = "other"
        if "condition" in fname.lower():
            domain = "conditions"
        elif "medication" in fname.lower():
            domain = "medications"
        elif "procedure" in fname.lower():
            domain = "procedures"
        elif "observation" in fname.lower():
            domain = "labs"

        df = pd.read_csv(fpath, dtype=str, low_memory=False)
        code_cols = [c for c in df.columns if "CODE" in c.upper()]
        desc_cols = [c for c in df.columns if "DESC" in c.upper()]

        if not code_cols:
            continue

        table_codes = 0
        for code_col in code_cols:
            matching_desc = None
            if code_col == "CODE" and "DESCRIPTION" in df.columns:
                matching_desc = "DESCRIPTION"
            elif code_col == "REASONCODE" and "REASONDESCRIPTION" in df.columns:
                matching_desc = "REASONDESCRIPTION"
            elif desc_cols:
                matching_desc = desc_cols[0]

            clean_codes = normalize_code_series(df[code_col])
            valid_mask = clean_codes.notna()
            valid_codes = clean_codes[valid_mask]

            counts = valid_codes.value_counts()
            code_counter.update(counts.to_dict())
            domain_counts[domain] += len(valid_codes)
            table_codes += len(valid_codes)

            if matching_desc and matching_desc in df.columns:
                sub_df = pd.DataFrame({"code": valid_codes, "desc": df.loc[valid_mask, matching_desc]})
                sub_df = sub_df.dropna(subset=["code", "desc"]).drop_duplicates(subset=["code"])
                for _, r in sub_df.iterrows():
                    c_val = str(r["code"])
                    if c_val not in code_descriptions:
                        code_descriptions[c_val] = str(r["desc"]).strip()

        logger.info(f"  {fname:<22} ({domain:<12}): scanned {table_codes:,d} code occurrences.")

    total_occurrences = sum(code_counter.values())
    unique_codes = len(code_counter)
    logger.info(f"Clinical codes: {total_occurrences:,d} occurrences across {unique_codes:,d} unique codes.")

    # Special tokens per specification: <PAD>: 0, <UNK>: 1, <BOS>: 2, <EOS>: 3
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
    }

    metadata = {
        "<PAD>": {"id": 0, "count": 0, "description": "Padding token"},
        "<UNK>": {"id": 1, "count": 0, "description": "Unknown / Out-of-vocabulary code"},
        "<BOS>": {"id": 2, "count": 0, "description": "Beginning of patient sequence"},
        "<EOS>": {"id": 3, "count": 0, "description": "End of patient sequence"},
    }

    current_id = 4
    for code, count in code_counter.most_common():
        vocab[code] = current_id
        metadata[code] = {
            "id": current_id,
            "count": count,
            "description": code_descriptions.get(code, "No description"),
        }
        current_id += 1

    # Save primary clinical_vocab.json
    os.makedirs(os.path.dirname(os.path.abspath(output_vocab_path)) if os.path.dirname(output_vocab_path) else ".", exist_ok=True)
    with open(output_vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)
    logger.info(f"Saved clinical vocabulary ({len(vocab):,d} tokens) to '{output_vocab_path}'")

    # Also save vocab.json for backwards compatibility
    vocab_compat_path = "data/vocab.json" if os.path.isdir("data") else "vocab.json"
    with open(vocab_compat_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    if output_metadata_path:
        with open(output_metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved clinical vocabulary metadata to '{output_metadata_path}'")

    return vocab, metadata, dict(domain_counts)


# =========================================================================
# 2. Genomic Vocabulary Generator
# =========================================================================

def build_genomic_vocab(
    genomics_dir: str = "genomics",
    output_vocab_path: str = "data/genomic_vocab.json",
    output_metadata_path: Optional[str] = "data/genomic_vocab_metadata.json",
) -> Tuple[Dict[str, int], Dict[str, dict], Dict[str, int]]:
    """
    Extract unique genetic variant identifiers and clinical significance tokens.

    Special tokens:
      "<PAD>": 0, "<UNK>": 1, "<MISSING_GENOME>": 2

    Returns:
        (genomic_vocab_dict, metadata_dict, significance_counts_dict)
    """
    if not os.path.isdir(genomics_dir):
        raise FileNotFoundError(f"Genomics directory not found: {genomics_dir}")

    dna_files = sorted(glob.glob(os.path.join(genomics_dir, "*_dna.csv")))
    if not dna_files:
        raise FileNotFoundError(f"No *_dna.csv files found in '{genomics_dir}'")

    logger.info(f"Scanning {len(dna_files):,d} genomic DNA files in '{genomics_dir}' for unique variants & significance...")

    variant_metadata = {}
    variant_presence_counts = Counter()
    significance_counts = Counter()

    # Fast scan: collect metadata and counts
    for fpath in dna_files:
        df = pd.read_csv(fpath, low_memory=False)
        if "INDEX" not in df.columns:
            continue

        for _, row in df.iterrows():
            idx_name = str(row["INDEX"]).strip()
            if not idx_name:
                continue

            sig = str(row.get("CLINICAL_SIGNIFICANCE", "")).strip()
            if sig and sig.lower() != "nan":
                significance_counts[sig] += 1

            is_variant = str(row.get("VARIANT", "")).strip().lower() in ["true", "1", "yes"]
            if is_variant:
                variant_presence_counts[idx_name] += 1

            if idx_name not in variant_metadata:
                variant_metadata[idx_name] = {
                    "index_prefix": str(row.get("INDEX_PREFIX", "")).strip(),
                    "chromosome": str(row.get("CHROMOSOME", "")).strip(),
                    "location": str(row.get("LOCATION", "")).strip(),
                    "strand": str(row.get("STRAND", "")).strip(),
                    "gene": str(row.get("GENE", "")).strip(),
                    "clinical_significance": sig,
                    "ancestral_allele": str(row.get("ANCESTRAL_ALLELE", "")).strip(),
                    "variant_alleles": str(row.get("VARIANT_ALLELE_LIST", "")).strip(),
                    "allele": str(row.get("ALLELE", "")).strip(),
                }

    unique_variants = sorted(variant_metadata.keys())
    unique_significance = sorted(significance_counts.keys())

    logger.info(f"Genomic tokens: {len(unique_variants):,d} variant loci, {len(unique_significance):,d} significance tokens.")

    # Dedicated special tokens: <PAD>: 0, <UNK>: 1, <MISSING_GENOME>: 2
    genomic_vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<MISSING_GENOME>": 2,
    }

    metadata = {
        "<PAD>": {"id": 0, "count": 0, "type": "special", "description": "Genomic padding token"},
        "<UNK>": {"id": 1, "count": 0, "type": "special", "description": "Unknown genomic token"},
        "<MISSING_GENOME>": {"id": 2, "count": 0, "type": "special", "description": "Missing genomic modality fallback token"},
    }

    current_id = 3
    # 1. Variant Loci
    for v_idx in unique_variants:
        genomic_vocab[v_idx] = current_id
        meta = variant_metadata[v_idx]
        meta["id"] = current_id
        meta["type"] = "variant_locus"
        meta["positive_variant_count"] = variant_presence_counts.get(v_idx, 0)
        metadata[v_idx] = meta
        current_id += 1

    # 2. Clinical Significance tokens
    for sig in unique_significance:
        sig_token = f"SIG_{sig.upper().replace(' ', '_')}"
        genomic_vocab[sig_token] = current_id
        metadata[sig_token] = {
            "id": current_id,
            "type": "clinical_significance",
            "name": sig,
            "count": significance_counts.get(sig, 0),
        }
        current_id += 1

    # Save genomic_vocab.json
    os.makedirs(os.path.dirname(os.path.abspath(output_vocab_path)) if os.path.dirname(output_vocab_path) else ".", exist_ok=True)
    with open(output_vocab_path, "w", encoding="utf-8") as f:
        json.dump(genomic_vocab, f, indent=2)
    logger.info(f"Saved genomic vocabulary ({len(genomic_vocab):,d} tokens) to '{output_vocab_path}'")

    if output_metadata_path:
        with open(output_metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved genomic vocabulary metadata to '{output_metadata_path}'")

    return genomic_vocab, metadata, dict(significance_counts)


# =========================================================================
# 3. Summary Visualizations
# =========================================================================

def generate_vocab_distribution_plot(
    clinical_vocab_meta: Dict[str, dict],
    domain_counts: Dict[str, int],
    significance_counts: Dict[str, int],
    output_path: str = "reports/figures/vocab_distribution.png",
    top_k_clinical: int = 20,
) -> str:
    """
    Generate a 3-panel summary visualization:
      1. Frequency bar chart of top clinical tokens.
      2. Event count breakdown across clinical domains.
      3. Frequency distribution of genomic variant clinical significance.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), dpi=300)

    # -------------------------------------------------------------
    # Panel 1: Top Clinical Tokens
    # -------------------------------------------------------------
    clinical_items = [
        (k, v.get("count", 0), v.get("description", k))
        for k, v in clinical_vocab_meta.items()
        if not k.startswith("<")
    ]
    clinical_items.sort(key=lambda x: x[1], reverse=True)
    top_clinical = clinical_items[:top_k_clinical]

    labels = []
    counts = []
    for code, cnt, desc in top_clinical:
        short_desc = (desc[:24] + "..") if len(desc) > 26 else desc
        labels.append(f"{code}\n({short_desc})")
        counts.append(cnt)

    y_pos = np.arange(len(labels))
    palette1 = sns.color_palette("mako", len(labels))
    axes[0].barh(y_pos, counts, color=palette1, edgecolor="#2d3748", linewidth=0.6)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Event Occurrences", fontsize=11, fontweight="bold")
    axes[0].set_title(f"Top {top_k_clinical} Most Frequent Clinical Tokens", fontsize=12, fontweight="bold", pad=12)
    for i, c in enumerate(counts):
        axes[0].text(c + max(counts) * 0.01, i, f"{c:,d}", va="center", fontsize=8, color="#2d3748")

    # -------------------------------------------------------------
    # Panel 2: Clinical Domain Breakdown
    # -------------------------------------------------------------
    domains = list(domain_counts.keys())
    domain_vals = [domain_counts[d] for d in domains]
    # Clean domain titles
    domain_titles = [d.replace("_", " ").title() for d in domains]

    palette2 = sns.color_palette("crest", len(domains))
    bars = axes[1].bar(domain_titles, domain_vals, color=palette2, edgecolor="#2d3748", linewidth=0.8, width=0.6)
    axes[1].set_ylabel("Total Event Count", fontsize=11, fontweight="bold")
    axes[1].set_title("Clinical Event Breakdown Across Domains", fontsize=12, fontweight="bold", pad=12)
    axes[1].tick_params(axis="x", rotation=25)
    for bar in bars:
        height = bar.get_height()
        axes[1].annotate(
            f"{int(height):,d}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # -------------------------------------------------------------
    # Panel 3: Genomic Significance Distribution
    # -------------------------------------------------------------
    sorted_sig = sorted(significance_counts.items(), key=lambda x: x[1], reverse=True)
    sig_labels = [s[0] for s in sorted_sig]
    sig_vals = [s[1] for s in sorted_sig]

    palette3 = sns.color_palette("flare", len(sig_labels))
    bars3 = axes[2].bar(sig_labels, sig_vals, color=palette3, edgecolor="#2d3748", linewidth=0.8, width=0.6)
    axes[2].set_ylabel("Occurrences in Genomic Cohort", fontsize=11, fontweight="bold")
    axes[2].set_title("Distribution of Genomic Variant Categories", fontsize=12, fontweight="bold", pad=12)
    axes[2].tick_params(axis="x", rotation=35)
    for bar in bars3:
        height = bar.get_height()
        axes[2].annotate(
            f"{int(height):,d}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    abs_path = os.path.abspath(output_path)
    logger.info(f"Saved vocabulary distribution figure to '{abs_path}'")
    return abs_path


# =========================================================================
# 4. Tokenizer Classes
# =========================================================================

class EHRTokenizer:
    """Clinical code tokenizer mapping diagnostic/medication/procedure codes to integer tokens."""

    def __init__(self, vocab_path_or_dict: Union[str, Dict[str, int]] = "data/clinical_vocab.json"):
        if isinstance(vocab_path_or_dict, str):
            actual_path = vocab_path_or_dict if os.path.exists(vocab_path_or_dict) else (
                "clinical_vocab.json" if os.path.exists("clinical_vocab.json") else "data/vocab.json"
            )
            with open(actual_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            self.vocab = vocab_path_or_dict

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.pad_id = self.vocab.get(self.pad_token, 0)
        self.unk_id = self.vocab.get(self.unk_token, 1)
        self.bos_id = self.vocab.get(self.bos_token, 2)
        self.eos_id = self.vocab.get(self.eos_token, 3)
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode_single(self, code: Any) -> int:
        if pd.isna(code):
            return self.unk_id
        s = str(code).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return self.vocab.get(s, self.unk_id)

    def encode(self, codes: List[Any], add_special_tokens: bool = False) -> List[int]:
        tokens = [self.encode_single(c) for c in codes if pd.notna(c) and str(c).strip() not in ["", "nan", "None"]]
        if add_special_tokens:
            return [self.bos_id] + tokens + [self.eos_id]
        return tokens

    def decode(self, token_ids: List[int]) -> List[str]:
        return [self.id_to_token.get(i, self.unk_token) for i in token_ids]


class GenomicTokenizer:
    """Genomic variant tokenizer mapping variant locus indices and significance tokens to integers."""

    def __init__(self, vocab_path_or_dict: Union[str, Dict[str, int]] = "data/genomic_vocab.json"):
        if isinstance(vocab_path_or_dict, str):
            actual_path = vocab_path_or_dict if os.path.exists(vocab_path_or_dict) else "genomic_vocab.json"
            with open(actual_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            self.vocab = vocab_path_or_dict

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.missing_token = "<MISSING_GENOME>"

        self.pad_id = self.vocab.get(self.pad_token, 0)
        self.unk_id = self.vocab.get(self.unk_token, 1)
        self.missing_id = self.vocab.get(self.missing_token, 2)
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode_single(self, token: Any) -> int:
        if pd.isna(token):
            return self.unk_id
        s = str(token).strip()
        return self.vocab.get(s, self.unk_id)

    def encode(self, tokens: List[Any]) -> List[int]:
        return [self.encode_single(t) for t in tokens if pd.notna(t) and str(t).strip()]

    def decode(self, token_ids: List[int]) -> List[str]:
        return [self.id_to_token.get(i, self.unk_token) for i in token_ids]


# =========================================================================
# 5. CLI Execution
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Build clinical & genomic vocabularies and summary plots.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="train_val_pre_anchor",
        help="Directory containing pre-anchor clinical CSV tables (default: train_val_pre_anchor).",
    )
    parser.add_argument(
        "--genomics-dir",
        type=str,
        default="genomics",
        help="Directory containing genomics DNA tables (default: genomics).",
    )
    parser.add_argument(
        "--clinical-vocab",
        type=str,
        default="data/clinical_vocab.json",
        help="Output path for clinical vocabulary (default: data/clinical_vocab.json).",
    )
    parser.add_argument(
        "--genomic-vocab",
        type=str,
        default="data/genomic_vocab.json",
        help="Output path for genomic vocabulary (default: data/genomic_vocab.json).",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default="reports/figures/vocab_distribution.png",
        help="Output path for distribution visualization plot (default: reports/figures/vocab_distribution.png).",
    )

    args = parser.parse_args()

    # 1. Clinical Vocabulary
    clin_vocab, clin_meta, domain_counts = build_clinical_vocab(
        data_dir=args.data_dir,
        output_vocab_path=args.clinical_vocab,
        output_metadata_path="data/clinical_vocab_metadata.json",
    )

    # 2. Genomic Vocabulary
    gen_vocab, gen_meta, sig_counts = build_genomic_vocab(
        genomics_dir=args.genomics_dir,
        output_vocab_path=args.genomic_vocab,
        output_metadata_path="data/genomic_vocab_metadata.json",
    )

    # 3. Visualizations
    generate_vocab_distribution_plot(
        clinical_vocab_meta=clin_meta,
        domain_counts=domain_counts,
        significance_counts=sig_counts,
        output_path=args.plot_output,
    )

    print("\n" + "=" * 70)
    print("VOCABULARY BUILD & VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"Clinical Vocab:    {args.clinical_vocab} ({len(clin_vocab):,d} tokens)")
    print(f"Genomic Vocab:     {args.genomic_vocab} ({len(gen_vocab):,d} tokens)")
    print(f"Distribution Plot: {args.plot_output}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
