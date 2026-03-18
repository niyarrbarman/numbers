"""
LLaMA 3.1 8B with dual-path numerical IO:
  - Input:  AnalyticNumberCodec (frozen, deterministic) → NumAdapter → additive injection
  - Output: Autoregressive structured decoder (sign → scale → length → digits)
  - Text:   Standard LM head (unchanged)

Reference:
  - num_analytic.py: AnalyticNumberCodec (sign + exponent + digit circles + value lane)
  - 124M/fe_adapt/model_analytic_surface.py: surface decoder, feedback embeddings
  - 124M/fe_adapt/numeric_surface.py: surface number helpers
"""

import sys
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# project root for num_analytic.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
# 124M/fe_adapt for numeric_surface.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '124M', 'fe_adapt'))
from num_analytic import AnalyticNumberCodec
from numeric_surface import (
    SurfaceNumberComponents,
    render_surface_components,
    surface_components_from_value,
    surface_components_to_row,
    row_to_surface_components,
)


# =============================================================================
# 1. NumEncoder — frozen AnalyticNumberCodec
# =============================================================================

class NumEncoder(nn.Module):
    """
    Frozen analytic number encoder using AnalyticNumberCodec.

    Maps scalar numbers to structured analytic representations:
      sign (2d) + exponent (4d) + digits (64d, cos/sin on 10-circle) + value (10d)
      = 80d total (with default K=32)

    No learned parameters — the encoding is deterministic and exact.
    The digit lane uses FoNE-style (cos(2πd/10), sin(2πd/10)) pairs, one per
    mantissa digit, giving the downstream adapter clean per-digit phase information.

    The adapter MLP downstream handles projection to LLM hidden dim.
    """

    def __init__(self, K=32, exp_min=-32, exp_max=32):
        super().__init__()
        self.codec = AnalyticNumberCodec(K=K, exp_min=exp_min, exp_max=exp_max)
        self.embed_dim = self.codec.total_dim

    def forward(self, x):
        """x: (N,) flat scalars → (N, embed_dim) analytic encoding."""
        device = x.device
        dtype = x.dtype
        vals = x.detach().cpu().tolist()
        encodings = []
        zero_enc = np.zeros(self.codec.total_dim, dtype=np.float64)
        for val in vals:
            try:
                enc = self.codec.encode(float(val))
            except (ValueError, RuntimeError):
                # out-of-range exponent or NaN/inf → zero vector
                enc = zero_enc
            encodings.append(enc)
        result = np.stack(encodings, axis=0)
        return torch.from_numpy(result).to(device=device, dtype=dtype)


# =============================================================================
# 2. NumAdapter — projects encoder output into LLM hidden dim
# =============================================================================

class NumAdapter(nn.Module):
    """codec_dim → hidden_dim MLP. Output is added to text embeddings (not replaced)."""

    def __init__(self, num_embed_dim=80, hidden_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, num_embeds):
        return self.proj(num_embeds)


# =============================================================================
# 3. SurfaceFeedbackEmbedding — structured number → embedding (for generation)
# =============================================================================

class SurfaceFeedbackEmbedding(nn.Module):
    """
    Embeds a decoded surface row [sign, scale, length, d0, d1, ...] back into
    the LLM's embedding space. Used during generation to feed decoded numbers
    back as input at subsequent <NUM> positions.
    """

    def __init__(self, max_digits=32, n_scale_classes=33, embed_dim=16,
                 hidden_dim=4096):
        super().__init__()
        self.max_digits = max_digits
        self.sign_embed = nn.Embedding(2, embed_dim)
        self.scale_embed = nn.Embedding(n_scale_classes, embed_dim)
        self.length_embed = nn.Embedding(max_digits + 1, embed_dim)
        self.digit_embed = nn.Embedding(10, embed_dim)
        flat_dim = (3 + max_digits) * embed_dim
        self.proj = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, rows):
        """rows: (N, 3 + max_digits) long tensor → (N, hidden_dim)."""
        rows = rows.long()
        sign_e = self.sign_embed(rows[:, 0])
        scale_e = self.scale_embed(rows[:, 1].clamp(0, self.scale_embed.num_embeddings - 1))
        len_e = self.length_embed(rows[:, 2].clamp(0, self.max_digits))
        digit_e = self.digit_embed(rows[:, 3:3 + self.max_digits].clamp(0, 9))
        flat = torch.cat([
            sign_e, scale_e, len_e, digit_e.reshape(rows.size(0), -1)
        ], dim=-1)
        return self.proj(flat)


# =============================================================================
# 4. NumDecoder — autoregressive structured decoder
# =============================================================================

class NumDecoder(nn.Module):
    """
    Autoregressive structured number decoder.

    From a single hidden state, predicts surface components sequentially:
      sign → scale (conditioned on sign) → length (conditioned on sign+scale)
      → digit_0 → digit_1 → ... → digit_{length-1}

    Digits are predicted via a GRUCell that conditions each digit on all
    previously predicted digits, closing the train/inference gap that
    caused parallel decoders to produce 0% exact match during generation.

    `digit_direction` controls prediction order:
      'rtl' (right-to-left): ones place first, MSB last (natural for addition)
      'ltr' (left-to-right): MSB first, LSB last (natural for comparison)
    """

    def __init__(
        self,
        hidden_dim=4096,
        trunk_hidden=1024,
        max_digits=32,
        n_scale_classes=33,
        n_length_classes=33,
        digit_gru_hidden=256,
        digit_embed_dim=32,
        component_embed_dim=32,
        digit_direction="rtl",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_digits = max_digits
        self.n_scale_classes = n_scale_classes
        self.n_length_classes = n_length_classes
        self.digit_direction = digit_direction

        # trunk: residual MLP enrichment
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, hidden_dim),
            nn.GELU(),
        )

        # sign head
        self.sign_head = nn.Linear(hidden_dim, 2)

        # scale head (conditioned on sign)
        self.sign_embed = nn.Embedding(2, component_embed_dim)
        self.scale_head = nn.Linear(hidden_dim + component_embed_dim, n_scale_classes)

        # length head (conditioned on sign + scale)
        self.scale_embed = nn.Embedding(n_scale_classes, component_embed_dim)
        self.length_head = nn.Linear(hidden_dim + 2 * component_embed_dim, n_length_classes)

        # digit GRU (autoregressive)
        self.digit_embed = nn.Embedding(10, digit_embed_dim)
        self.length_embed_for_gru = nn.Embedding(n_length_classes, component_embed_dim)
        self.digit_gru = nn.GRUCell(digit_embed_dim, digit_gru_hidden)
        self.digit_init = nn.Linear(
            hidden_dim + 3 * component_embed_dim,
            digit_gru_hidden,
        )
        self.digit_out = nn.Linear(digit_gru_hidden, 10)
        self.start_digit_embed = nn.Parameter(torch.randn(digit_embed_dim) * 0.02)

    def _reorder_digits(self, digit_targets, lengths):
        """
        Reorder digit targets according to digit_direction.
        Surface format stores digits MSB-first: [d0, d1, ..., d_{len-1}, 0, 0, ...]
        For RTL: reverse the active digits so the GRU predicts LSB first.
        Returns reordered targets (same shape).
        """
        if self.digit_direction == "ltr":
            return digit_targets

        reordered = digit_targets.clone()
        for i in range(digit_targets.size(0)):
            L = int(lengths[i].item())
            L = min(L, self.max_digits)
            if L > 0:
                reordered[i, :L] = digit_targets[i, :L].flip(0)
        return reordered

    def _unreorder_digits(self, digit_preds, lengths):
        """Reverse the reordering to get back to MSB-first for rendering."""
        if self.digit_direction == "ltr":
            return digit_preds

        unreordered = digit_preds.clone()
        for i in range(digit_preds.size(0)):
            L = int(lengths[i].item())
            L = min(L, self.max_digits)
            if L > 0:
                unreordered[i, :L] = digit_preds[i, :L].flip(0)
        return unreordered

    def forward(self, hidden_states, surface_targets=None, teacher_forcing=True):
        """
        Args:
            hidden_states:  (M, hidden_dim) at output <NUM> positions
            surface_targets: (M, 3 + max_digits) [sign, scale, length, d0..dK]
            teacher_forcing: use ground truth for conditioning during training
        Returns:
            dict with sign_logits, scale_logits, length_logits, digit_logits
        """
        M = hidden_states.size(0)
        device = hidden_states.device

        # --- trunk enrichment (residual) ---
        enriched = hidden_states + self.trunk(hidden_states)

        # --- sign ---
        sign_logits = self.sign_head(enriched)

        if teacher_forcing and surface_targets is not None:
            sign_input = surface_targets[:, 0].long()
        else:
            sign_input = sign_logits.argmax(dim=-1)
        sign_e = self.sign_embed(sign_input)

        # --- scale ---
        scale_logits = self.scale_head(torch.cat([enriched, sign_e], dim=-1))

        if teacher_forcing and surface_targets is not None:
            scale_input = surface_targets[:, 1].long()
        else:
            scale_input = scale_logits.argmax(dim=-1)
        scale_e = self.scale_embed(scale_input.clamp(0, self.n_scale_classes - 1))

        # --- length ---
        length_logits = self.length_head(
            torch.cat([enriched, sign_e, scale_e], dim=-1)
        )

        if teacher_forcing and surface_targets is not None:
            length_input = surface_targets[:, 2].long().clamp(1, self.max_digits)
        else:
            length_input = length_logits.argmax(dim=-1).clamp(1, self.max_digits)
        length_e = self.length_embed_for_gru(
            length_input.clamp(0, self.n_length_classes - 1)
        )

        # --- digit GRU init ---
        gru_h = self.digit_init(
            torch.cat([enriched, sign_e, scale_e, length_e], dim=-1)
        )

        # --- prepare digit targets (reordered if RTL) ---
        if surface_targets is not None:
            digit_targets = surface_targets[:, 3:3 + self.max_digits].long().clamp(0, 9)
            digit_targets_ordered = self._reorder_digits(digit_targets, length_input)
        else:
            digit_targets_ordered = None

        # --- autoregressive digit prediction ---
        all_digit_logits = []
        prev_emb = self.start_digit_embed.unsqueeze(0).expand(M, -1)

        for t in range(self.max_digits):
            gru_h = self.digit_gru(prev_emb, gru_h)
            digit_logit = self.digit_out(gru_h)
            all_digit_logits.append(digit_logit)

            if teacher_forcing and digit_targets_ordered is not None:
                prev_emb = self.digit_embed(digit_targets_ordered[:, t])
            else:
                prev_emb = self.digit_embed(digit_logit.argmax(dim=-1))

        digit_logits = torch.stack(all_digit_logits, dim=1)  # (M, max_digits, 10)

        return {
            "sign_logits": sign_logits,
            "scale_logits": scale_logits,
            "length_logits": length_logits,
            "digit_logits": digit_logits,   # in prediction order (RTL or LTR)
            "digit_targets_ordered": digit_targets_ordered,  # matching order for loss
        }

    @torch.no_grad()
    def decode(self, hidden_states):
        """Decode hidden states into SurfaceNumberComponents (inference)."""
        out = self.forward(hidden_states, teacher_forcing=False)
        signs = out["sign_logits"].argmax(dim=-1)
        scales = out["scale_logits"].argmax(dim=-1)
        lengths = out["length_logits"].argmax(dim=-1).clamp(1, self.max_digits)
        raw_digits = out["digit_logits"].argmax(dim=-1)

        # reverse digit order back to MSB-first for rendering
        digits_msb = self._unreorder_digits(raw_digits, lengths)

        results = []
        for i in range(hidden_states.size(0)):
            sign = 1 if signs[i].item() == 0 else -1
            scale = int(scales[i].item())
            length = int(lengths[i].item())
            digs = tuple(int(digits_msb[i, j].item()) for j in range(self.max_digits))
            results.append(SurfaceNumberComponents(
                sign=sign, scale=scale, length=length, digits=digs,
            ))
        return results


# =============================================================================
# 5. NumLoss — structured loss with per-digit weighting
# =============================================================================

class NumLoss(nn.Module):
    """
    Loss for the autoregressive structured decoder.

    L = CE_sign + CE_scale + CE_length + weighted_masked_CE_digits
        + lambda * consistency_loss

    Digit CE is masked to active positions (< length) and weighted by
    positional decay so that more significant digits matter more.
    """

    def __init__(self, max_digits=32, digit_decay=0.85,
                 consistency_lambda=0.1, digit_direction="rtl"):
        super().__init__()
        self.max_digits = max_digits
        self.digit_decay = digit_decay
        self.consistency_lambda = consistency_lambda
        self.digit_direction = digit_direction

    def forward(self, decoder_outputs, surface_targets, batch_indices=None):
        device = surface_targets.device
        sign_targets = surface_targets[:, 0].long()
        scale_targets = surface_targets[:, 1].long()
        len_targets = surface_targets[:, 2].long().clamp(1, self.max_digits)

        sign_loss = F.cross_entropy(decoder_outputs["sign_logits"], sign_targets)
        scale_loss = F.cross_entropy(decoder_outputs["scale_logits"], scale_targets)
        len_loss = F.cross_entropy(decoder_outputs["length_logits"], len_targets)

        # --- digit loss (in prediction order) ---
        digit_logits = decoder_outputs["digit_logits"]  # (M, max_digits, 10)
        digit_targets_ordered = decoder_outputs["digit_targets_ordered"]
        if digit_targets_ordered is None:
            digit_targets_raw = surface_targets[:, 3:3 + self.max_digits].long().clamp(0, 9)
            digit_targets_ordered = digit_targets_raw
            # reorder would be needed here in real use — but if no targets, skip

        K = self.max_digits

        # weighting: in prediction order, what's the significance?
        # RTL: step 0 = LSB (least significant), step len-1 = MSB (most significant)
        #   → weight MSB higher: w[t] = decay^(len-1-t) so last step (MSB) gets w=1
        # LTR: step 0 = MSB (most significant), step len-1 = LSB
        #   → weight MSB higher: w[t] = decay^t so first step (MSB) gets w=1
        digit_weights = torch.tensor(
            [self.digit_decay ** i for i in range(K)],
            device=device, dtype=torch.float32,
        )
        if self.digit_direction == "rtl":
            digit_weights = digit_weights.flip(0)

        per_digit_ce = F.cross_entropy(
            digit_logits.reshape(-1, 10),
            digit_targets_ordered.reshape(-1),
            reduction="none",
        ).reshape(-1, K)

        active_mask = (
            torch.arange(K, device=device).unsqueeze(0) < len_targets.unsqueeze(1)
        ).float()
        weighted = per_digit_ce * digit_weights.unsqueeze(0) * active_mask
        denom = (digit_weights.unsqueeze(0) * active_mask).sum(dim=-1).clamp_min(1.0)
        digit_loss = (weighted.sum(dim=-1) / denom).mean()

        total = sign_loss + scale_loss + len_loss + digit_loss

        # --- consistency loss (optional) ---
        consistency_loss = torch.tensor(0.0, device=device)
        if self.consistency_lambda > 0 and batch_indices is not None:
            consistency_loss = self._compute_consistency(
                surface_targets, batch_indices, decoder_outputs,
            )
            total = total + self.consistency_lambda * consistency_loss

        return {
            "sign_loss": sign_loss,
            "scale_loss": scale_loss,
            "len_loss": len_loss,
            "digit_loss": digit_loss,
            "consistency_loss": consistency_loss,
            "total": total,
        }

    @torch.compiler.disable
    def _compute_consistency(self, surface_targets, batch_indices, decoder_outputs):
        """Penalize inconsistent predictions for repeated mentions of same number."""
        sign_logits = decoder_outputs["sign_logits"]
        if sign_logits.size(0) < 2:
            return sign_logits.new_zeros(())

        rows = surface_targets.detach().cpu().tolist()
        bids = batch_indices.detach().cpu().tolist()
        groups = {}
        for idx, (bid, row) in enumerate(zip(bids, rows)):
            key = (int(bid), tuple(int(v) for v in row))
            groups.setdefault(key, []).append(idx)

        losses = []
        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            t = torch.as_tensor(idxs, device=sign_logits.device, dtype=torch.long)
            sp = F.softmax(sign_logits[t], dim=-1)
            losses.append((sp - sp.mean(0, keepdim=True)).pow(2).mean())

        if not losses:
            return sign_logits.new_zeros(())
        return torch.stack(losses).mean()


# =============================================================================
# 6. Accuracy metrics
# =============================================================================

@torch.no_grad()
def compute_numeric_accuracy(decoder_outputs, surface_targets, max_digits=32,
                             digit_direction="rtl"):
    """Compute per-component and exact-match accuracy."""
    sign_pred = decoder_outputs["sign_logits"].argmax(dim=-1)
    scale_pred = decoder_outputs["scale_logits"].argmax(dim=-1)
    len_pred = decoder_outputs["length_logits"].argmax(dim=-1).clamp(1, max_digits)
    digit_pred_ordered = decoder_outputs["digit_logits"].argmax(dim=-1)

    sign_gt = surface_targets[:, 0].long()
    scale_gt = surface_targets[:, 1].long()
    len_gt = surface_targets[:, 2].long().clamp(1, max_digits)
    digit_gt_msb = surface_targets[:, 3:3 + max_digits].long()

    sign_acc = (sign_pred == sign_gt).float().mean()
    scale_acc = (scale_pred == scale_gt).float().mean()
    len_acc = (len_pred == len_gt).float().mean()

    # reorder predicted digits back to MSB-first for comparison
    if digit_direction == "rtl":
        digit_pred_msb = digit_pred_ordered.clone()
        for i in range(digit_pred_ordered.size(0)):
            L = int(len_pred[i].item())
            L = min(L, max_digits)
            if L > 0:
                digit_pred_msb[i, :L] = digit_pred_ordered[i, :L].flip(0)
    else:
        digit_pred_msb = digit_pred_ordered

    active = torch.arange(max_digits, device=sign_pred.device).unsqueeze(0) < len_gt.unsqueeze(1)
    digit_correct = (digit_pred_msb == digit_gt_msb) & active
    digit_acc = digit_correct.float().sum() / active.float().sum().clamp_min(1.0)

    all_digits_ok = ((digit_pred_msb == digit_gt_msb) | ~active).all(dim=-1)
    exact_match = (
        (sign_pred == sign_gt) & (scale_pred == scale_gt)
        & (len_pred == len_gt) & all_digits_ok
    ).float().mean()

    wrong = (digit_pred_msb != digit_gt_msb) & active
    first_wrong = torch.where(
        wrong.any(dim=-1),
        wrong.float().argmax(dim=-1).float(),
        torch.full_like(len_gt.float(), max_digits),
    ).mean()

    return {
        "sign_acc": sign_acc.item(),
        "scale_acc": scale_acc.item(),
        "len_acc": len_acc.item(),
        "digit_acc": digit_acc.item(),
        "exact_match": exact_match.item(),
        "first_wrong_digit": first_wrong.item(),
    }


# =============================================================================
# 7. LlamaModel — full architecture
# =============================================================================

class LlamaModel(nn.Module):
    """
    LLaMA 3.1 with dual-path numerical IO.

    Input:  text tokens → text_emb ─────────────────┐
            raw numbers → NumEncoder → NumAdapter ──(+)──► LLM
    Output: LLM hidden → lm_head ──► text logits
                       → NumDecoder ──► sign, scale, length, digits
    """

    NUM_TOKEN = "<NUM>"

    def __init__(
        self,
        model_path: str,
        codec_K: int = 32,
        codec_exp_min: int = -32,
        codec_exp_max: int = 32,
        max_digits: int = 32,
        n_scale_classes: int = 33,
        n_length_classes: int = 33,
        digit_gru_hidden: int = 256,
        digit_direction: str = "rtl",
        num_loss_lambda: float = 1.0,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path)
        hidden_dim = self.config.hidden_size
        self.max_digits = max_digits
        self.num_loss_lambda = num_loss_lambda
        self.digit_direction = digit_direction

        # --- LLM backbone ---
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=self.config,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )

        # --- numeric encoder path (frozen analytic codec) ---
        self.num_encoder = NumEncoder(K=codec_K, exp_min=codec_exp_min, exp_max=codec_exp_max)
        num_embed_dim = self.num_encoder.embed_dim
        self.num_adapter = NumAdapter(num_embed_dim=num_embed_dim, hidden_dim=hidden_dim)

        # --- numeric decoder (autoregressive) ---
        self.num_decoder = NumDecoder(
            hidden_dim=hidden_dim,
            max_digits=max_digits,
            n_scale_classes=n_scale_classes,
            n_length_classes=n_length_classes,
            digit_gru_hidden=digit_gru_hidden,
            digit_direction=digit_direction,
        )

        # --- numeric loss ---
        self.num_loss_fn = NumLoss(
            max_digits=max_digits,
            digit_direction=digit_direction,
        )

        # --- surface feedback embedding (for generation) ---
        self.surface_feedback = SurfaceFeedbackEmbedding(
            max_digits=max_digits,
            n_scale_classes=n_scale_classes,
            hidden_dim=hidden_dim,
        )

        # track <NUM> token id after tokenizer setup
        self.num_token_id = None

    def freeze_backbone(self):
        """Freeze the entire LLM (stage 1: adapter + decoder only)."""
        for p in self.model.parameters():
            p.requires_grad = False

    def freeze_layers(self, num_layers: int):
        """Freeze first `num_layers` transformer layers + embeddings."""
        for p in self.model.model.embed_tokens.parameters():
            p.requires_grad = False
        for layer in self.model.model.layers[:num_layers]:
            for p in layer.parameters():
                p.requires_grad = False

    def _inject_num_embeddings(self, input_ids, num_values=None, num_positions=None,
                                num_surface_rows=None, num_surface_mask=None):
        """
        Build input embeddings with additive numeric injection.

        At <NUM> positions, adds NumAdapter output to text embedding.
        If surface_rows are available (during generation), uses structured
        feedback embedding instead.
        """
        embeds = self.model.model.embed_tokens(input_ids)

        if num_values is None or num_positions is None:
            return embeds

        embeds = embeds.clone()
        B = input_ids.size(0)

        for b in range(B):
            for i in range(num_positions.size(1)):
                pos = num_positions[b, i].item()
                if pos < 0:
                    break

                # check if this position has a structured feedback row
                use_surface = (
                    num_surface_mask is not None
                    and num_surface_rows is not None
                    and num_surface_mask[b, i].item()
                )

                if use_surface:
                    row = num_surface_rows[b, i:i+1]  # (1, 3+max_digits)
                    feedback = self.surface_feedback(row)  # (1, hidden)
                    embeds[b, pos] = embeds[b, pos] + feedback.squeeze(0).to(embeds.dtype)
                else:
                    val = num_values[b, i:i+1]  # (1,)
                    enc = self.num_encoder(val)  # (1, 128)
                    adapted = self.num_adapter(enc)  # (1, hidden)
                    embeds[b, pos] = embeds[b, pos] + adapted.squeeze(0).to(embeds.dtype)

        return embeds

    def _extract_num_hidden(self, hidden_states, num_out_positions):
        """Gather hidden states at output <NUM> positions."""
        if num_out_positions is None:
            return None
        B = hidden_states.size(0)
        max_nums = num_out_positions.size(1)
        H = hidden_states.size(-1)
        gathered = torch.zeros(B, max_nums, H,
                               device=hidden_states.device, dtype=hidden_states.dtype)
        for b in range(B):
            for i in range(max_nums):
                pos = num_out_positions[b, i].item()
                if pos < 0:
                    break
                gathered[b, i] = hidden_states[b, pos]
        return gathered

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        num_values=None,
        num_positions=None,
        num_out_positions=None,
        num_surface_targets=None,
        num_surface_rows=None,
        num_surface_mask=None,
        teacher_forcing=True,
    ):
        """
        Forward with dual input/output paths.

        Standard text args:
            input_ids, attention_mask, labels

        Numeric input args:
            num_values:       (B, max_in_nums)  raw float values
            num_positions:    (B, max_in_nums)  token positions, padded with -1
            num_surface_rows: (B, max_in_nums, 3+D)  structured feedback rows
            num_surface_mask: (B, max_in_nums)  which inputs use feedback

        Numeric output args:
            num_out_positions:    (B, max_out_nums) positions, padded with -1
            num_surface_targets:  (B, max_out_nums, 3+D) ground truth surface rows
        """
        inputs_embeds = self._inject_num_embeddings(
            input_ids, num_values, num_positions,
            num_surface_rows, num_surface_mask,
        )

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        txt_loss = outputs.loss if labels is not None else torch.tensor(0.0, device=input_ids.device)

        num_loss_dict = None
        num_acc = None

        if num_out_positions is not None and num_surface_targets is not None:
            last_hidden = outputs.hidden_states[-1]
            num_hidden = self._extract_num_hidden(last_hidden, num_out_positions)

            mask = (num_out_positions >= 0)
            flat_hidden = num_hidden[mask]
            flat_targets = num_surface_targets[mask]

            if flat_hidden.size(0) > 0:
                batch_indices = torch.arange(
                    input_ids.size(0), device=input_ids.device
                ).unsqueeze(1).expand_as(num_out_positions)[mask]

                decoder_out = self.num_decoder(
                    flat_hidden.float(),
                    surface_targets=flat_targets,
                    teacher_forcing=teacher_forcing,
                )
                num_loss_dict = self.num_loss_fn(
                    decoder_out, flat_targets, batch_indices,
                )
                num_acc = compute_numeric_accuracy(
                    decoder_out, flat_targets,
                    max_digits=self.max_digits,
                    digit_direction=self.digit_direction,
                )

        num_total = num_loss_dict["total"] if num_loss_dict else torch.tensor(0.0, device=input_ids.device)
        loss = txt_loss + self.num_loss_lambda * num_total

        return {
            "loss": loss,
            "txt_loss": txt_loss,
            "num_loss_dict": num_loss_dict,
            "num_acc": num_acc,
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate_with_numbers(
        self,
        input_ids,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        num_values=None,
        num_positions=None,
        num_surface_rows=None,
        num_surface_mask=None,
        eos_token_id=None,
    ):
        """
        Autoregressive generation with structured number decoding.

        When the model emits <NUM>:
          1. Extract hidden state
          2. Run autoregressive NumDecoder (inference mode)
          3. Render the decoded SurfaceNumberComponents to text
          4. Feed structured feedback back for subsequent <NUM> inputs
        """
        assert input_ids.size(0) == 1, "generate supports batch_size=1"
        device = input_ids.device

        if num_values is None:
            num_values = torch.zeros(1, 0, device=device)
        if num_positions is None:
            num_positions = torch.full((1, 0), -1, dtype=torch.long, device=device)
        if num_surface_rows is None:
            num_surface_rows = torch.zeros(1, 0, 3 + self.max_digits,
                                           dtype=torch.long, device=device)
        if num_surface_mask is None:
            num_surface_mask = torch.zeros(1, 0, dtype=torch.bool, device=device)

        generated_numbers = []

        for step in range(max_new_tokens):
            inputs_embeds = self._inject_num_embeddings(
                input_ids, num_values, num_positions,
                num_surface_rows, num_surface_mask,
            )

            outputs = self.model(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
            )

            logits = outputs.logits[:, -1, :]
            if temperature is not None and temperature > 0:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = logits.argmax(dim=-1, keepdim=True)

            next_is_num = (self.num_token_id is not None
                           and idx_next.item() == self.num_token_id)

            feedback_row = None
            if next_is_num:
                hidden_last = outputs.hidden_states[-1][:, -1:, :]
                components = self.num_decoder.decode(hidden_last.squeeze(1).float())
                rendered = render_surface_components(components[0])
                generated_numbers.append((input_ids.size(1), rendered, components[0]))

                row = surface_components_to_row(
                    components[0],
                    max_digits=self.max_digits,
                )
                feedback_row = torch.tensor([row], dtype=torch.long, device=device)

            # extend sequences
            input_ids = torch.cat([input_ids, idx_next], dim=1)

            if next_is_num:
                new_val = torch.zeros(1, 1, device=device)
                new_pos = torch.tensor([[input_ids.size(1) - 1]], dtype=torch.long, device=device)
                new_row = feedback_row.unsqueeze(0)
                new_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
            else:
                new_val = torch.zeros(1, 1, device=device)
                new_pos = torch.full((1, 1), -1, dtype=torch.long, device=device)
                new_row = torch.zeros(1, 1, 3 + self.max_digits,
                                      dtype=torch.long, device=device)
                new_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)

            num_values = torch.cat([num_values, new_val], dim=1)
            num_positions = torch.cat([num_positions, new_pos], dim=1)
            num_surface_rows = torch.cat([num_surface_rows, new_row], dim=1)
            num_surface_mask = torch.cat([num_surface_mask, new_mask], dim=1)

            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break

        return input_ids, generated_numbers

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# =============================================================================
# 8. Dataset
# =============================================================================

class FinetuneDataset(Dataset):
    """
    JSONL dataset: {"instruction": "...", "input": "...", "output": "..."}

    Tokenises and masks prompt portion in labels.
    Does NOT handle <NUM> injection yet — that requires task-specific
    number parsing which will be added when the data format is defined.
    """

    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        with open(file_path, "r") as f:
            self.data = [json.loads(line) for line in f if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        instruction = sample["instruction"]
        inp = sample.get("input", "")
        output = sample["output"]

        prompt = f"{instruction}\n{inp}".strip() if inp else instruction
        full_text = f"{prompt}\n{output}{self.tokenizer.eos_token}"

        prompt_ids = self.tokenizer(
            prompt + "\n", truncation=True, max_length=self.max_length
        )["input_ids"]
        prompt_len = len(prompt_ids)

        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# =============================================================================
# 9. Finetuning loop
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out["loss"].item()
    model.train()
    return total_loss / max(len(loader), 1)


def finetune(
    model_path: str,
    data_path: str,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-5,
    max_length: int = 512,
    grad_accum_steps: int = 4,
    warmup_steps: int = 50,
    freeze_layers: int = 0,
    eval_split: float = 0.05,
    log_every: int = 10,
    save_path: str = "checkpoints",
    device: str = "cuda",
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LlamaModel(model_path)
    if freeze_layers > 0:
        model.freeze_layers(freeze_layers)
    model = model.to(device)
    total, trainable = model.param_count()
    print(f"params  total={total:,}  trainable={trainable:,}")

    full_dataset = FinetuneDataset(data_path, tokenizer, max_length=max_length)
    eval_size = max(1, int(len(full_dataset) * eval_split))
    train_size = len(full_dataset) - eval_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    print(f"data  train={train_size}  eval={eval_size}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    total_steps = (len(train_loader) // grad_accum_steps) * epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    global_step = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out["loss"] / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            running_loss += loss.item() * grad_accum_steps

            if (step + 1) % log_every == 0:
                avg = running_loss / log_every
                cur_lr = scheduler.get_last_lr()[0]
                print(f"epoch {epoch+1}/{epochs}  step {step+1}/{len(train_loader)}"
                      f"  loss={avg:.4f}  lr={cur_lr:.2e}")
                running_loss = 0.0

        eval_loss = evaluate(model, eval_loader, device)
        print(f"epoch {epoch+1}/{epochs}  eval_loss={eval_loss:.4f}")

        ckpt_dir = f"{save_path}/epoch_{epoch+1}"
        model.model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"saved checkpoint -> {ckpt_dir}")

    print("finetuning complete")


# =============================================================================
# 10. Entry point
# =============================================================================

if __name__ == "__main__":
    MODEL_PATH = "/path/to/llama-3.1-8b"
    DATA_PATH = "data.jsonl"

    finetune(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        epochs=3,
        batch_size=2,
        lr=2e-5,
        max_length=512,
        grad_accum_steps=4,
        warmup_steps=50,
        freeze_layers=0,
        eval_split=0.05,
        device="cuda",
    )
