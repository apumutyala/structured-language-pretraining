"""Parameter-free FFN/head RMS variant of the fresh typed byte language model.

This is a prospective stability intervention, not a safety or learning theorem.
The original model and its receipts remain separate. State dictionaries have the
same keys as the original; class/source/normalization provenance is mandatory.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .typed_language_model import (
    PAD_ID, VOCAB_SIZE, TypedLanguageModel, TypedLanguageOutput, _gru_sequence,
)

RMS_EPSILON = 1e-5
NORMALIZATION_RECIPE = "parameter_free_ffn_pre_and_both_heads_scaled_rms_v2"


def parameter_free_rms(value: Tensor) -> Tensor:
    """R(v)=v/sqrt(mean(v*v)+1e-5), reducing only the hidden dimension.

    Accumulate at least in float32; preserve float64 and the input/output dtype.
    Compute the algebraically equivalent scaled expression, never x*x or
    scale*scale, so huge finite inputs do not silently normalize to zero due to
    square overflow. The amax scale remains differentiable (piecewise), with no
    detach, clipping, learned scale/bias or nonfinite sanitization. Floating
    association differs from the literal formula; this is not an IEEE proof.
    """
    if (not isinstance(value, Tensor) or value.ndim < 1 or not value.shape[-1]
            or not value.is_floating_point()):
        raise ValueError("RMS input must be floating with a nonempty hidden dimension")
    work = value if value.dtype == torch.float64 else value.float()
    root_epsilon = math.sqrt(RMS_EPSILON)
    scale = work.abs().amax(-1, keepdim=True).clamp_min(root_epsilon)
    scaled = work / scale
    denominator_square = scaled.square().mean(-1, keepdim=True) + (root_epsilon / scale).square()
    return (scaled * torch.rsqrt(denominator_square)).to(value.dtype)


class NormalizedTypedLanguageModel(TypedLanguageModel):
    """Same initialized parameters and evidence routing, RMS before FFNs/heads.

    The inherited constructor never loads learned parameters. Normalization has
    no learned state and does not change the residual or recurrent equations.
    ``representation`` and ``claim_representation`` retain their original
    meaning: the unnormalized final stream and its final-valid-position slice.
    Both heads consume their normalized versions. Typed support still comes
    exclusively from the unchanged evidence lane and policy-aware decoder;
    raw byte outputs remain uncertified.

    Forward is intentionally explicit to make normalization placement auditable.
    Validation, masking and routing mirror the bound original source; no hooks
    or monkey patches change the model computation.
    """

    normalization_recipe = NORMALIZATION_RECIPE
    normalization_epsilon = RMS_EPSILON

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, *,
                observed_event_ids: Tensor | None = None, trusted_event_mask: Tensor | None = None,
                policies=None) -> TypedLanguageOutput:
        if (not isinstance(input_ids, Tensor) or input_ids.ndim != 2
                or input_ids.dtype not in (torch.int32, torch.int64)
                or not input_ids.shape[0] or not input_ids.shape[1]
                or bool(((input_ids < 0) | (input_ids >= VOCAB_SIZE)).any())):
            raise ValueError("input_ids must be nonempty int32/int64 [batch,time] in byte vocabulary [0,258]")
        if (observed_event_ids is not None or trusted_event_mask is not None) and attention_mask is None:
            raise ValueError("an event lane requires an explicit application-owned attention_mask")
        valid = input_ids != PAD_ID if attention_mask is None else attention_mask
        if (not isinstance(valid, Tensor) or valid.dtype != torch.bool
                or valid.shape != input_ids.shape or valid.device != input_ids.device):
            raise ValueError("attention_mask must be bool with input_ids shape and device")
        positions = torch.where(valid, torch.arange(input_ids.shape[1], device=input_ids.device), -1).amax(-1)
        if bool((positions < 0).any()):
            raise ValueError("each row must have at least one valid slot")
        evidence = self.evidence_lane(input_ids, observed_event_ids, trusted_event_mask, valid)
        stream = self.embedding(input_ids)
        core_states, native_activations = [], []
        for layer in range(self.num_layers):
            projection = self.semantic_projections[0 if self.projection_layout == "shared" else layer]
            semantic = projection(evidence.flatten(2).to(stream.dtype))
            inputs = stream + semantic * (self.semantic_mode == "internal")
            core = (self.recurrent_layers[layer](inputs, mask=valid)[0] if self.recurrent_kind == "ssm"
                    else _gru_sequence(self.recurrent_layers[layer], inputs, valid))
            core_states.append(core)
            residual = core.to(self.native_gates[layer].weight.dtype)
            if self.recurrent_residual:
                residual = residual + stream.to(residual.dtype)
            if layer == self.num_layers - 1:
                residual = residual + semantic * (self.semantic_mode == "output_only")
            normalized = parameter_free_rms(residual)
            native = F.silu(self.native_gates[layer](normalized)) * self.native_values[layer](normalized)
            native_activations.append(native)
            stream = residual + self.native_downs[layer](native)
        normalized_stream = parameter_free_rms(stream)
        raw_logits = self.lm_head(normalized_stream)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        claim_representation = stream[rows, positions]
        proposal_logits = self.claim_head(normalized_stream[rows, positions])
        final_evidence = evidence[rows, positions].bool()
        grounded = (None if policies is None else
                    self.evidence_decoder(final_evidence, proposal_logits, policies=policies))
        return TypedLanguageOutput(raw_logits, evidence, tuple(core_states), tuple(native_activations),
                                   stream, claim_representation, proposal_logits, grounded,
                                   final_evidence, positions)
