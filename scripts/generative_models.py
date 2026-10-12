"""Generative EHR Model Architecture

Dual-stream causal transformer architecture inspired by nanoGPT and BEHRT
for patient longitudinal electronic health records and genomic sequences.
"""

from __future__ import annotations

import sys

# Prevent scripts/tokenize.py from shadowing Python standard library tokenize
_scripts_dir = sys.path[0] if sys.path else ""
if sys.path and ("scripts" in sys.path[0] or sys.path[0].replace("\\", "/").rstrip("/").endswith("/scripts")):
    del sys.path[0]
import tokenize as _stdlib_tokenize

if _scripts_dir and _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from dataclasses import dataclass
import math
from typing import Any
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class EHRModelConfig:
    """nanoGPT-style configuration managing all architectural hyperparameters."""

    vocab_size: int = 1362
    max_age: int = 121
    genomic_vocab_size: int = 256
    max_seq_len: int = 1024
    max_genomic_len: int = 1024
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    num_classes: int = 40
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional attention weights return."""

    def __init__(self, config: EHRModelConfig, max_len: int):
        super().__init__()
        assert config.hidden_dim % config.num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout

        # Q, K, V projections in a single linear layer
        self.c_attn = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower triangular
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, seq_len, dim = x.size()

        # Compute query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.hidden_dim, dim=2)

        # Reshape to [B, num_heads, seq_len, head_dim]
        k = k.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Raw attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply causal mask
        att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))

        # Apply padding attention mask if provided
        if attention_mask is not None:
            # attention_mask shape: [B, seq_len], True for real tokens, False for pad
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]
            att = att.masked_fill(~pad_mask, float("-inf"))

        att_weights = F.softmax(att, dim=-1)
        # Handle cases where all values are masked (e.g. padding rows)
        att_weights = torch.nan_to_num(att_weights, nan=0.0)

        att_dropped = self.attn_dropout(att_weights)
        y = att_dropped @ v  # [B, num_heads, seq_len, head_dim]

        # Reassemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(b, seq_len, dim)
        y = self.resid_dropout(self.c_proj(y))

        attn_matrix = att_weights if output_attentions else None
        return y, attn_matrix


class MLP(nn.Module):
    """Standard feed-forward network with GELU activation."""

    def __init__(self, config: EHRModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-LN Transformer decoder block."""

    def __init__(self, config: EHRModelConfig, max_len: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_dim)
        self.attn = CausalSelfAttention(config, max_len)
        self.ln_2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, attn_weights = self.attn(
            self.ln_1(x),
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights


class DualStreamEHRTransformer(nn.Module):
    """Dual-Stream EHR Transformer combining Clinical and Genomic streams.

    - Clinical Stream (nanoGPT + BEHRT): code embeddings + age embeddings + positional embeddings.
    - Genomics Stream (nanoGPT): genomic embeddings + positional embeddings.
    - Fusion: Concatenates clinical and genomic streams with explicit zero-out when genomics is absent.
    - Dual Heads: lm_head for next-token pretraining and classification_head for 40-class multi-label.
    """

    def __init__(self, config: EHRModelConfig):
        super().__init__()
        self.config = config

        # ----------------------------------------------------
        # 1. Clinical Stream (BEHRT + nanoGPT)
        # ----------------------------------------------------
        self.clinical_code_emb = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=0)
        self.clinical_age_emb = nn.Embedding(config.max_age, config.hidden_dim, padding_idx=0)
        self.clinical_pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.clinical_drop = nn.Dropout(config.dropout)
        self.clinical_blocks = nn.ModuleList([
            TransformerBlock(config, config.max_seq_len) for _ in range(config.num_layers)
        ])
        self.clinical_ln = nn.LayerNorm(config.hidden_dim)

        # ----------------------------------------------------
        # 2. Genomics Stream (nanoGPT)
        # ----------------------------------------------------
        self.genomic_code_emb = nn.Embedding(config.genomic_vocab_size, config.hidden_dim, padding_idx=0)
        self.genomic_pos_emb = nn.Embedding(config.max_genomic_len, config.hidden_dim)
        self.genomic_drop = nn.Dropout(config.dropout)
        self.genomic_blocks = nn.ModuleList([
            TransformerBlock(config, config.max_genomic_len) for _ in range(config.num_layers)
        ])
        self.genomic_ln = nn.LayerNorm(config.hidden_dim)

        # ----------------------------------------------------
        # 3. Heads
        # ----------------------------------------------------
        fused_dim = 2 * config.hidden_dim
        # Stage 1: LM Head for autoregressive next-token prediction
        self.lm_head = nn.Linear(fused_dim, config.vocab_size, bias=False)

        # Stage 2: 40-class multi-label condition prediction
        self.classification_head = nn.Sequential(
            nn.Linear(fused_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

        # Initialize all weights with strict normal distribution (mean=0.0, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Custom weight initialization using strict normal distribution (mean=0.0, std=0.02)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        age_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        genomic_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> dict[str, Any]:
        """Forward pass of the dual-stream model.

        Args:
            input_ids: Clinical token sequence tensor [B, L]
            age_ids: Parallel patient age IDs tensor [B, L]
            attention_mask: Boolean attention mask tensor [B, L]
            genomic_ids: Optional genomic sequence tensor [B, L]
            output_attentions: If True, returns attention weight matrices.

        Returns:
            Dictionary containing 'lm_logits', 'cls_logits', and optionally 'attentions'.
        """
        b, seq_len = input_ids.size()
        device = input_ids.device

        # Default age_ids to zeros if not provided
        if age_ids is None:
            age_ids = torch.zeros_like(input_ids)

        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)

        # ----------------------------------------------------
        # Clinical Stream Forward Pass
        # ----------------------------------------------------
        tok_emb = self.clinical_code_emb(input_ids)
        age_emb = self.clinical_age_emb(age_ids)
        pos_emb = self.clinical_pos_emb(positions)

        x_clinical = self.clinical_drop(tok_emb + age_emb + pos_emb)

        clinical_attns = []
        for block in self.clinical_blocks:
            x_clinical, attn_w = block(
                x_clinical,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            if output_attentions and attn_w is not None:
                clinical_attns.append(attn_w)

        clinical_out = self.clinical_ln(x_clinical)  # [B, L, D]

        # ----------------------------------------------------
        # Genomics Stream Forward Pass & Zero-Out
        # ----------------------------------------------------
        # If genomic input is None, empty, or all padding, explicitly zero-out output
        has_genomics = (
            genomic_ids is not None
            and genomic_ids.numel() > 0
            and not (genomic_ids == 0).all()
        )

        if has_genomics:
            gen_len = genomic_ids.size(1)
            gen_pos = torch.arange(0, gen_len, dtype=torch.long, device=device).unsqueeze(0)
            x_genomic = self.genomic_drop(
                self.genomic_code_emb(genomic_ids) + self.genomic_pos_emb(gen_pos)
            )
            for block in self.genomic_blocks:
                x_genomic, _ = block(x_genomic, output_attentions=False)
            genomic_out = self.genomic_ln(x_genomic)

            # Explicit zero-out for individual patients who lack genomics in a mixed batch
            patient_has_genomics = (genomic_ids != 0).any(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
            genomic_out = genomic_out * patient_has_genomics.float()

            # Align sequence length between streams if needed
            if genomic_out.shape[1] < clinical_out.shape[1]:
                pad_len = clinical_out.shape[1] - genomic_out.shape[1]
                genomic_out = F.pad(genomic_out, (0, 0, 0, pad_len), value=0.0)
            elif genomic_out.shape[1] > clinical_out.shape[1]:
                genomic_out = genomic_out[:, :clinical_out.shape[1], :]
        else:
            # Explicit zero-out of the genomics stream
            genomic_out = torch.zeros_like(clinical_out)

        # ----------------------------------------------------
        # Size Mismatch & Shape Validation Checks
        # ----------------------------------------------------
        assert clinical_out.shape[0] == genomic_out.shape[0], (
            f"Batch size mismatch between streams: "
            f"clinical={clinical_out.shape[0]} vs genomic={genomic_out.shape[0]}"
        )
        assert clinical_out.shape[1] == genomic_out.shape[1], (
            f"Sequence length mismatch between streams: "
            f"clinical={clinical_out.shape[1]} vs genomic={genomic_out.shape[1]}"
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------
        fused = torch.cat([clinical_out, genomic_out], dim=-1)  # [B, L, 2 * D]

        # LM Head (next-token prediction)
        lm_logits = self.lm_head(fused)  # [B, L, vocab_size]

        # Classification Head (multi-label condition prediction)
        # Extract representation at last valid token (or <EOS>)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            batch_idx = torch.arange(b, device=device)
            last_rep = fused[batch_idx, lengths]  # [B, 2 * D]
        else:
            last_rep = fused[:, -1, :]  # [B, 2 * D]

        cls_logits = self.classification_head(last_rep)  # [B, 40]

        result = {
            "lm_logits": lm_logits,
            "cls_logits": cls_logits,
            "fused": fused,
        }
        if output_attentions:
            result["attentions"] = clinical_attns

        return result

    def load_pretrained_core(self, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Stage 2 Transition Method:

        Loads core transformer layers from pre-trained weights while explicitly
        catching size mismatches and safely discarding pre-trained lm_head weights.
        """
        current_state = self.state_dict()
        filtered_state = {}
        discarded_keys = []
        loaded_keys = []
        mismatched_keys = []

        for key, tensor in state_dict.items():
            # Safely discard lm_head weights for Stage 2 fine-tuning
            if key.startswith("lm_head.") or key.startswith("classification_head."):
                discarded_keys.append(key)
                continue

            if key not in current_state:
                discarded_keys.append(key)
                continue

            # Explicitly catch size mismatches
            target_shape = current_state[key].shape
            if tensor.shape != target_shape:
                mismatched_keys.append((key, tuple(tensor.shape), tuple(target_shape)))
                continue

            filtered_state[key] = tensor
            loaded_keys.append(key)

        # Safely load the filtered core weights
        load_result = self.load_state_dict(filtered_state, strict=False)

        return {
            "loaded_count": len(loaded_keys),
            "loaded_keys": loaded_keys,
            "discarded_count": len(discarded_keys),
            "discarded_keys": discarded_keys,
            "mismatched_count": len(mismatched_keys),
            "mismatched_keys": mismatched_keys,
            "missing_keys": load_result.missing_keys,
            "unexpected_keys": load_result.unexpected_keys,
        }

    def get_attention_heatmap(
        self,
        input_ids: torch.Tensor,
        age_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_reduction: str = "mean",
    ) -> torch.Tensor:
        """Explainability feature:

        Extracts self-attention weights from the final layer of the Clinical Stream.
        Returns:
            Attention weights tensor: [B, L, L] (if reduced) or [B, num_heads, L, L].
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(
                input_ids=input_ids,
                age_ids=age_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        attentions = output.get("attentions", [])
        if not attentions:
            raise RuntimeError("No attention weights returned from forward pass.")

        # Final layer attention weights: shape [B, num_heads, L, L]
        final_layer_attn = attentions[-1]

        if head_reduction == "mean":
            return final_layer_attn.mean(dim=1)  # [B, L, L]
        elif head_reduction == "max":
            return final_layer_attn.max(dim=1).values
        else:
            return final_layer_attn

    def log_model_architecture(self, wandb_run: Any = None) -> dict[str, int]:
        """Calculates parameter counts for clinical, genomics, and head components

        and logs the architectural summary to Weights & Biases if an active run is provided.
        """

        def count_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        clinical_params = (
            count_params(self.clinical_code_emb)
            + count_params(self.clinical_age_emb)
            + count_params(self.clinical_pos_emb)
            + count_params(self.clinical_blocks)
            + count_params(self.clinical_ln)
        )
        genomic_params = (
            count_params(self.genomic_code_emb)
            + count_params(self.genomic_pos_emb)
            + count_params(self.genomic_blocks)
            + count_params(self.genomic_ln)
        )
        lm_params = count_params(self.lm_head)
        cls_params = count_params(self.classification_head)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        summary = {
            "clinical_stream_parameters": clinical_params,
            "genomic_stream_parameters": genomic_params,
            "lm_head_parameters": lm_params,
            "classification_head_parameters": cls_params,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

        if wandb_run is not None:
            if hasattr(wandb_run, "summary"):
                for k, v in summary.items():
                    wandb_run.summary[k] = v
            if hasattr(wandb_run, "log"):
                wandb_run.log({"model_architecture": summary})

        return summary
