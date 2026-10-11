"""
scripts/models.py

PyTorch Model Architectures for EHR Patient Condition Forecasting:
  1. Discrete Age Embeddings:
     Categorical nn.Embedding(120, d_model) representing patient chronological age
     at the time of each clinical event (bounded [0, 119]).
  2. Baseline 1 (Clinical): ClinicalTransformer
     nanoGPT-style causal decoder-only transformer with causal self-attention,
     masked sequence pooling, and a linear multi-label classification head.
  3. Baseline 2 (Multimodal): MultimodalTransformer
     Late fusion architecture integrating ClinicalTransformer and GenomicMLP
     with missing-DNA gating (has_genomics zero-out) and joint classification head.
"""

import math
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. nanoGPT-Style Causal Transformer Building Blocks
# ==============================================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention module (nanoGPT style).
    Strictly attends to past positions (causal mask) while masking out key padding.
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert d_model % n_head == 0, f"d_model ({d_model}) must be divisible by n_head ({n_head})"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout = dropout

        # Batched Q, K, V linear projections
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape [B, T, d_model].
            attention_mask: Optional boolean or binary mask of shape [B, T],
                            where True (or 1) indicates a valid token and False (or 0) indicates padding.
            output_attentions: If True, returns (output, attention_weights).

        Returns:
            Output tensor of shape [B, T, d_model] (and optionally [B, n_head, T, T] attention weights).
        """
        B, T, C = x.size()

        # Compute queries, keys, and values
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, n_head, T, head_dim]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, n_head, T, head_dim]
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, n_head, T, head_dim]

        # Construct causal mask: position i can only attend to j <= i
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))  # [T, T]

        if attention_mask is not None:
            # Combine causal mask [1, 1, T, T] with key padding mask [B, 1, 1, T]
            pad_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            combined_mask = causal_mask.unsqueeze(0).unsqueeze(1) & pad_mask  # [B, 1, T, T]
        else:
            combined_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, T, T]

        if output_attentions:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(~combined_mask, float("-inf"))
            att_weights = F.softmax(att, dim=-1)
            att_weights = torch.nan_to_num(att_weights, nan=0.0)
            y = att_weights @ v
        else:
            dropout_p = self.dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=combined_mask,
                dropout_p=dropout_p,
                is_causal=False,  # Causal structure is already embedded in combined_mask
            )
            att_weights = None

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, d_model]
        out = self.resid_dropout(self.c_proj(y))
        if output_attentions:
            return out, att_weights
        return out


class MLP(nn.Module):
    """Feedforward network with GELU activation and dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Block (nanoGPT design)."""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout=dropout, bias=bias)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if output_attentions:
            attn_out, att_weights = self.attn(self.ln_1(x), attention_mask=attention_mask, output_attentions=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, att_weights
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


# ==============================================================================
# 2. Baseline 1: ClinicalTransformer (Clinical Baseline)
# ==============================================================================

class ClinicalTransformer(nn.Module):
    """
    Baseline 1: Causal Decoder-Only Transformer for Clinical Event Sequences.

    Inputs:
      - input_ids: Integer tokens representing clinical codes [B, S].
      - age_ids: Integer tokens representing patient age at event in years [0, 119] [B, S].
      - attention_mask: Optional boolean/binary mask [B, S] (True = valid token, False = pad).

    Mechanism:
      - Embedding = TokenEmbedding + AgeEmbedding + PositionalEmbedding
      - Causal self-attention layers strictly masking future clinical events.
      - Masked sequence pooling (mean pooling over valid tokens or last valid token).
      - Multi-label classification head projecting pooled latent vector to 40 logits.
      - Logits returned directly without Softmax (paired with BCEWithLogitsLoss).
    """

    def __init__(
        self,
        vocab_size: int = 735,
        max_seq_len: int = 512,
        num_ages: int = 120,
        d_model: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        num_classes: int = 40,
        pad_id: int = 0,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        """
        Args:
            vocab_size: Number of unique tokens in clinical vocabulary.
            max_seq_len: Maximum sequence length supported by positional embeddings.
            num_ages: Total categorical age bins (default 120, ages 0-119).
            d_model: Hidden representation dimension.
            n_head: Number of attention heads.
            n_layer: Number of transformer blocks.
            num_classes: Number of target condition classes (40).
            pad_id: Token ID representing padding.
            dropout: Dropout probability.
            pooling: Sequence pooling strategy: 'mean' or 'last'.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_ages = num_ages
        self.d_model = d_model
        self.num_classes = num_classes
        self.pad_id = pad_id
        self.pooling = pooling.lower()
        if self.pooling not in ["mean", "last", "eos"]:
            raise ValueError(f"Invalid pooling mode '{pooling}'. Must be 'mean', 'last', or 'eos'.")

        # 1. Embeddings: Token + Discrete Age + Positional
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.age_embedding = nn.Embedding(num_embeddings=num_ages, embedding_dim=d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        # 2. Transformer Backbone
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # 3. Classification Head (Raw logits, no softmax)
        self.classifier = nn.Linear(d_model, num_classes)

    def _pool_sequence(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool hidden states across the sequence dimension.

        Args:
            hidden_states: [B, S, d_model]
            attention_mask: [B, S] (bool/float, True = valid token)

        Returns:
            Pooled summary vector [B, d_model].
        """
        B, S, D = hidden_states.shape

        if attention_mask is None:
            if self.pooling in ["last", "eos"]:
                return hidden_states[:, -1, :]
            return hidden_states.mean(dim=1)

        mask_f = attention_mask.float()  # [B, S]

        if self.pooling in ["last", "eos"]:
            # Extract the last valid non-padded token position per patient
            lengths = mask_f.sum(dim=1).long()  # [B]
            last_indices = torch.clamp(lengths - 1, min=0)  # [B]
            batch_indices = torch.arange(B, device=hidden_states.device)
            return hidden_states[batch_indices, last_indices, :]

        # Default: Masked Mean Pooling
        mask_expanded = mask_f.unsqueeze(-1)  # [B, S, 1]
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # [B, d_model]
        lengths = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
        return sum_hidden / lengths  # [B, d_model]

    def get_clinical_features(
        self,
        input_ids: torch.Tensor,
        age_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Extract the pooled clinical summary vector [B, d_model].
        Used by MultimodalTransformer for late fusion.
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Positional indices [0, 1, ..., S-1]
        positions = torch.arange(0, S, dtype=torch.long, device=device).unsqueeze(0)  # [1, S]

        # Embeddings: Token + Discrete Age + Positional
        tok_emb = self.token_embedding(input_ids)
        age_emb = self.age_embedding(torch.clamp(age_ids, 0, self.num_ages - 1))
        pos_emb = self.pos_embedding(positions)

        x = self.emb_drop(tok_emb + age_emb + pos_emb)  # [B, S, d_model]

        # Pass through Causal Transformer Blocks
        all_attentions = [] if output_attentions else None
        for block in self.blocks:
            if output_attentions:
                x, att_weights = block(x, attention_mask=attention_mask, output_attentions=True)
                all_attentions.append(att_weights)
            else:
                x = block(x, attention_mask=attention_mask)

        # Final LayerNorm
        x = self.ln_f(x)

        # Pool sequence into a single patient summary vector
        pooled = self._pool_sequence(x, attention_mask=attention_mask)
        if output_attentions:
            return pooled, all_attentions
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        age_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
        output_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Args:
            input_ids: [B, S] long tensor.
            age_ids: [B, S] long tensor (ages 0-119).
            attention_mask: [B, S] bool/float tensor.
            return_features: If True, returns (logits, pooled_features).
            output_attentions: If True, returns (logits, all_attentions).

        Returns:
            Logits of shape [B, num_classes] (and optionally [B, d_model] features or attentions).
        """
        if output_attentions:
            features, all_attentions = self.get_clinical_features(
                input_ids=input_ids,
                age_ids=age_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        else:
            features = self.get_clinical_features(
                input_ids=input_ids,
                age_ids=age_ids,
                attention_mask=attention_mask,
            )
            all_attentions = None

        logits = self.classifier(features)  # [B, num_classes]

        if output_attentions:
            return logits, all_attentions
        if return_features:
            return logits, features
        return logits


# ==============================================================================
# 3. Genomic Representation: GenomicMLP
# ==============================================================================

class GenomicMLP(nn.Module):
    """
    Lightweight MLP to process static genomic variant tokens into a genetic risk vector.

    Inputs:
      - genomic_ids: Static integer tokens for fixed variant loci [B, num_loci].
      - genomic_mask: Optional boolean indicator [B, num_loci] where True indicates active mutation.

    Outputs:
      - Genetic risk vector [B, out_dim].
    """

    def __init__(
        self,
        genomic_vocab_size: int = 333,
        num_loci: int = 322,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        out_dim: int = 64,
        pad_id: int = 0,
        dropout: float = 0.1,
    ):
        """
        Args:
            genomic_vocab_size: Total tokens in genomic vocabulary.
            num_loci: Number of genomic variant loci (322).
            embedding_dim: Latent dimension for locus embeddings.
            hidden_dim: Intermediate hidden dimension.
            out_dim: Final genetic risk vector dimension.
            pad_id: Padding token ID.
            dropout: Dropout probability.
        """
        super().__init__()
        self.num_loci = num_loci
        self.embedding_dim = embedding_dim
        self.out_dim = out_dim

        self.embedding = nn.Embedding(genomic_vocab_size, embedding_dim, padding_idx=pad_id)
        self.net = nn.Sequential(
            nn.Linear(num_loci * embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        genomic_ids: torch.Tensor,
        genomic_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            genomic_ids: [B, num_loci] long tensor.
            genomic_mask: Optional [B, num_loci] boolean tensor (active mutation mask).

        Returns:
            Genetic risk vector of shape [B, out_dim].
        """
        B, N = genomic_ids.shape
        emb = self.embedding(genomic_ids)  # [B, N, embedding_dim]

        # If mutation mask is provided, weight locus representations
        if genomic_mask is not None:
            # Active mutations receive positive weighting while reference loci retain baseline
            mask_weight = genomic_mask.float().unsqueeze(-1)  # [B, N, 1]
            emb = emb * (1.0 + mask_weight)

        emb_flat = emb.view(B, N * self.embedding_dim)  # [B, N * embedding_dim]
        return self.net(emb_flat)  # [B, out_dim]


# ==============================================================================
# 4. Baseline 2: MultimodalTransformer (Late Fusion)
# ==============================================================================

class MultimodalTransformer(nn.Module):
    """
    Baseline 2: Late Fusion Multimodal Transformer.

    Integrates:
      1. ClinicalTransformer: Produces patient clinical summary vector [B, d_model] using age_ids.
      2. GenomicMLP: Produces genetic risk vector [B, genomic_dim].
      3. Missing-Data Masking: Uses has_genomics flag [B] to zero out the genetic risk vector
         for patients lacking DNA sequencing data.
      4. Late Fusion Head: Concatenates clinical and masked genomic vectors, projecting
         the joint representation to 40 condition logits (no Softmax).

    Supports invocation via either explicit arguments or the aligned dictionary returned
    by EHRDataset.collate_fn.
    """

    def __init__(
        self,
        clinical_vocab_size: int = 735,
        genomic_vocab_size: int = 333,
        num_genomic_loci: int = 322,
        clinical_d_model: int = 128,
        clinical_n_head: int = 4,
        clinical_n_layer: int = 4,
        clinical_max_seq_len: int = 512,
        num_ages: int = 120,
        genomic_embedding_dim: int = 16,
        genomic_hidden_dim: int = 64,
        genomic_out_dim: int = 64,
        num_classes: int = 40,
        pad_id: int = 0,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.clinical_d_model = clinical_d_model
        self.genomic_out_dim = genomic_out_dim

        # 1. Clinical Timeline Transformer Backbone
        self.clinical_transformer = ClinicalTransformer(
            vocab_size=clinical_vocab_size,
            max_seq_len=clinical_max_seq_len,
            num_ages=num_ages,
            d_model=clinical_d_model,
            n_head=clinical_n_head,
            n_layer=clinical_n_layer,
            num_classes=num_classes,
            pad_id=pad_id,
            dropout=dropout,
            pooling=pooling,
        )

        # 2. Lightweight Genomic Variant MLP
        self.genomic_mlp = GenomicMLP(
            genomic_vocab_size=genomic_vocab_size,
            num_loci=num_genomic_loci,
            embedding_dim=genomic_embedding_dim,
            hidden_dim=genomic_hidden_dim,
            out_dim=genomic_out_dim,
            pad_id=pad_id,
            dropout=dropout,
        )

        # 3. Late Fusion Linear Classification Head
        joint_dim = clinical_d_model + genomic_out_dim
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(joint_dim, num_classes),
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        age_ids: Optional[torch.Tensor] = None,
        genomic_ids: Optional[torch.Tensor] = None,
        has_genomics: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        genomic_mask: Optional[torch.Tensor] = None,
        batch: Optional[Dict[str, Any]] = None,
        output_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Forward pass for late-fusion multimodal classification.

        Args:
            input_ids: [B, S] clinical token IDs.
            age_ids: [B, S] patient age at event in years [0, 119].
            genomic_ids: [B, 322] genomic locus variant tokens.
            has_genomics: [B] or [B, 1] indicator (1.0 = present, 0.0 = missing).
            attention_mask: [B, S] clinical sequence validity mask.
            genomic_mask: [B, 322] variant presence boolean mask.
            batch: Optional dictionary matching EHRDataset collate_fn return format.
            output_attentions: If True, returns (logits, all_attentions).

        Returns:
            Logits tensor of shape [B, 40] (and optionally attention weights).
        """
        # Convenient dictionary unpacking if batch dict is provided
        if batch is not None:
            input_ids = batch["input_ids"]
            age_ids = batch["age_ids"]
            attention_mask = batch.get("attention_mask")
            genomic_ids = batch["genomic_ids"]
            genomic_mask = batch.get("genomic_mask")
            has_genomics = batch.get("has_genomics")

        if input_ids is None or age_ids is None or genomic_ids is None:
            raise ValueError("input_ids, age_ids, and genomic_ids must be provided.")

        # 1. Clinical Patient Summary Vector [B, d_model]
        if output_attentions:
            clinical_vec, all_attentions = self.clinical_transformer.get_clinical_features(
                input_ids=input_ids,
                age_ids=age_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        else:
            clinical_vec = self.clinical_transformer.get_clinical_features(
                input_ids=input_ids,
                age_ids=age_ids,
                attention_mask=attention_mask,
            )
            all_attentions = None

        # 2. Genomic Risk Vector [B, genomic_out_dim]
        genomic_vec = self.genomic_mlp(
            genomic_ids=genomic_ids,
            genomic_mask=genomic_mask,
        )

        # 3. Masking: Zero out genetic risk vector if DNA data is missing
        if has_genomics is not None:
            if has_genomics.ndim == 1:
                has_genomics = has_genomics.unsqueeze(-1)  # [B, 1]
            genomic_vec = genomic_vec * has_genomics.float()

        # 4. Late Fusion: Concatenation along latent feature dimension
        fused_vec = torch.cat([clinical_vec, genomic_vec], dim=-1)  # [B, d_model + genomic_dim]

        # 5. Multilabel Classification Output
        logits = self.fusion_classifier(fused_vec)  # [B, num_classes]

        if output_attentions:
            return logits, all_attentions
        return logits


# ==============================================================================
# 5. Local Verification Suite (__main__)
# ==============================================================================

def main():
    print("=" * 76)
    print("TESTING PYTORCH BASELINE ARCHITECTURES WITH DISCRETE AGE EMBEDDINGS")
    print("=" * 76)

    # 1. Load vocabulary sizes if available, otherwise use defaults
    clinical_vocab_size = 735
    genomic_vocab_size = 333
    num_genomic_loci = 322
    num_classes = 40
    num_ages = 120

    clin_vocab_path = "data/clinical_vocab.json" if os.path.exists("data/clinical_vocab.json") else "clinical_vocab.json"
    if os.path.exists(clin_vocab_path):
        with open(clin_vocab_path, "r", encoding="utf-8") as f:
            clinical_vocab_size = len(json.load(f))
    gen_vocab_path = "data/genomic_vocab.json" if os.path.exists("data/genomic_vocab.json") else "genomic_vocab.json"
    if os.path.exists(gen_vocab_path):
        with open(gen_vocab_path, "r", encoding="utf-8") as f:
            genomic_vocab_size = len(json.load(f))

    print(f"\n[Environment Vocabularies Loaded]:")
    print(f"  Clinical Vocab Size: {clinical_vocab_size}")
    print(f"  Genomic Vocab Size:  {genomic_vocab_size}")
    print(f"  Genomic Loci Count:  {num_genomic_loci}")
    print(f"  Target Classes:      {num_classes}")
    print(f"  Discrete Age Bins:   {num_ages} (0 to 119)")

    # 2. Configure Verified Dataloader Dimensions
    B = 4           # batch_size
    S = 256         # seq_len
    G = 322         # genomic_seq (num_loci)
    C = num_classes # 40

    print(f"\n[Verified DataLoader Dummy Shapes]:")
    print(f"  batch_size:       {B}")
    print(f"  seq_len:          {S}")
    print(f"  genomic_seq:      {G}")
    print(f"  num_classes:      {C}")

    # Synthesize dummy dataloader tensors
    torch.manual_seed(42)
    dummy_input_ids = torch.randint(low=1, high=clinical_vocab_size, size=(B, S), dtype=torch.long)
    dummy_age_ids = torch.randint(low=0, high=num_ages, size=(B, S), dtype=torch.long)
    dummy_attention_mask = torch.ones(B, S, dtype=torch.bool)
    # Simulate variable sequence lengths with padding
    dummy_attention_mask[0, 180:] = False
    dummy_attention_mask[1, 220:] = False
    dummy_input_ids[0, 180:] = 0
    dummy_input_ids[1, 220:] = 0
    dummy_age_ids[0, 180:] = 0
    dummy_age_ids[1, 220:] = 0

    dummy_genomic_ids = torch.randint(low=3, high=genomic_vocab_size, size=(B, G), dtype=torch.long)
    dummy_genomic_mask = torch.rand(B, G) > 0.85
    # Patients 0 and 1 have genomics, patients 2 and 3 are missing genomics
    dummy_has_genomics = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)

    # --------------------------------------------------------------------------
    # Test Baseline 1: ClinicalTransformer with Discrete Age Embeddings
    # --------------------------------------------------------------------------
    print("\n" + "-" * 76)
    print("1. Testing Baseline 1: ClinicalTransformer (Age Embeddings)...")
    clinical_model = ClinicalTransformer(
        vocab_size=clinical_vocab_size,
        max_seq_len=512,
        num_ages=num_ages,
        d_model=128,
        n_head=4,
        n_layer=4,
        num_classes=C,
        dropout=0.1,
        pooling="mean",
    )
    clinical_model.eval()

    with torch.no_grad():
        clinical_logits = clinical_model(
            input_ids=dummy_input_ids,
            age_ids=dummy_age_ids,
            attention_mask=dummy_attention_mask,
        )

    print(f"  ClinicalTransformer Output Shape: {tuple(clinical_logits.shape)}")
    print(f"  Output Dtype:                      {clinical_logits.dtype}")
    print(f"  Output Range:                      min={clinical_logits.min().item():.3f}, max={clinical_logits.max().item():.3f}")
    assert clinical_logits.shape == (B, C), f"Expected shape ({B}, {C}), got {clinical_logits.shape}"
    assert not torch.isnan(clinical_logits).any(), "NaN found in ClinicalTransformer logits!"
    print("  [PASS] Baseline 1 (ClinicalTransformer) successfully yields [4, 40]!")

    # --------------------------------------------------------------------------
    # Test Baseline 2: MultimodalTransformer
    # --------------------------------------------------------------------------
    print("\n" + "-" * 76)
    print("2. Testing Baseline 2: MultimodalTransformer (Late Fusion with Age Embeddings)...")
    multimodal_model = MultimodalTransformer(
        clinical_vocab_size=clinical_vocab_size,
        genomic_vocab_size=genomic_vocab_size,
        num_genomic_loci=G,
        clinical_d_model=128,
        clinical_n_head=4,
        clinical_n_layer=4,
        num_ages=num_ages,
        genomic_embedding_dim=16,
        genomic_hidden_dim=64,
        genomic_out_dim=64,
        num_classes=C,
        dropout=0.1,
        pooling="mean",
    )
    multimodal_model.eval()

    with torch.no_grad():
        multimodal_logits = multimodal_model(
            input_ids=dummy_input_ids,
            age_ids=dummy_age_ids,
            genomic_ids=dummy_genomic_ids,
            has_genomics=dummy_has_genomics,
            attention_mask=dummy_attention_mask,
            genomic_mask=dummy_genomic_mask,
        )

    print(f"  MultimodalTransformer Output Shape: {tuple(multimodal_logits.shape)}")
    print(f"  Output Dtype:                        {multimodal_logits.dtype}")
    print(f"  Output Range:                        min={multimodal_logits.min().item():.3f}, max={multimodal_logits.max().item():.3f}")
    assert multimodal_logits.shape == (B, C), f"Expected shape ({B}, {C}), got {multimodal_logits.shape}"
    assert not torch.isnan(multimodal_logits).any(), "NaN found in MultimodalTransformer logits!"
    print("  [PASS] Baseline 2 (MultimodalTransformer) successfully yields [4, 40]!")

    # --------------------------------------------------------------------------
    # Test Dictionary Input (Matching EHRDataset collate_fn format)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 76)
    print("3. Testing EHRDataset Batch Dictionary Unpacking with age_ids...")
    batch_dict = {
        "input_ids": dummy_input_ids,
        "attention_mask": dummy_attention_mask,
        "age_ids": dummy_age_ids,
        "genomic_ids": dummy_genomic_ids,
        "genomic_mask": dummy_genomic_mask,
        "has_genomics": dummy_has_genomics,
    }

    with torch.no_grad():
        dict_logits = multimodal_model(batch=batch_dict)

    assert torch.allclose(dict_logits, multimodal_logits), "Dictionary pass does not match direct argument pass!"
    print(f"  Batch Dict Logits Shape: {tuple(dict_logits.shape)}")
    print("  [PASS] Dictionary unpacking produces identical output to direct tensor args!")

    # --------------------------------------------------------------------------
    # Test Missing Genomic Zero-Out Invariance
    # --------------------------------------------------------------------------
    print("\n" + "-" * 76)
    print("4. Testing Missing Genomics Zero-Out Invariance (has_genomics=0)...")
    altered_genomic_ids = dummy_genomic_ids.clone()
    altered_genomic_ids[2:] = torch.randint_like(altered_genomic_ids[2:], low=1, high=50)

    with torch.no_grad():
        altered_logits = multimodal_model(
            input_ids=dummy_input_ids,
            age_ids=dummy_age_ids,
            genomic_ids=altered_genomic_ids,
            has_genomics=dummy_has_genomics,
            attention_mask=dummy_attention_mask,
            genomic_mask=dummy_genomic_mask,
        )

    assert torch.allclose(multimodal_logits[2:], altered_logits[2:], atol=1e-6), (
        "Invariance check failed: patients with has_genomics=0 were affected by genomic_ids perturbations!"
    )
    print("  [PASS] Missing genomic gating verified: patients with has_genomics=0 are completely invariant to DNA inputs!")

    print("\n" + "=" * 76)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
