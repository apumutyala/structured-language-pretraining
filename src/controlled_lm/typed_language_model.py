"""Fresh byte language model with a disjoint, application-trusted evidence lane.

Raw next-byte prose is always uncertified. Only the optional fixed typed claim
interface has the observed-evidence support guarantee, conditional on trusted
event provenance and current authenticated workspace-key policies.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .evidence_decoder import CLAIM_COUNT, EvidenceDecoder, GroundedCategorical
from .invariant_ssm import InvariantSelectiveSSM
from .parallel_ssm import ParallelSelectiveSSM
from .typed_evidence_lane import TypedEvidenceLane

PAD_ID, BOS_ID, EOS_ID, BYTE_START, VOCAB_SIZE = 0, 1, 2, 3, 259
SEMANTIC_MODES = ("none", "internal", "output_only")


@dataclass
class TypedLanguageOutput:
    raw_logits: Tensor
    evidence_trace: Tensor
    core_states: tuple[Tensor, ...]
    native_activations: tuple[Tensor, ...]
    representation: Tensor
    claim_representation: Tensor
    proposal_logits: Tensor
    grounded: GroundedCategorical | None
    final_evidence: Tensor
    final_positions: Tensor

    @property
    def logits(self):
        """Conventional LM alias; these raw byte logits remain uncertified."""
        return self.raw_logits


def _gru_sequence(recurrent, inputs, valid):
    """Skip masked transitions; retain the fused path for prefix padding."""
    if not bool((~valid[:, :-1] & valid[:, 1:]).any()):
        hidden, _ = recurrent(inputs)
        previous = torch.where(valid, torch.arange(valid.shape[1], device=valid.device), -1)
        previous = previous.cummax(1).values
        held = hidden.gather(1, previous.clamp_min(0)[..., None].expand_as(hidden))
        return torch.where((previous >= 0)[..., None], held, 0.)
    state = inputs.new_zeros(1, inputs.shape[0], inputs.shape[2])
    history = []
    for position in range(inputs.shape[1]):
        _, proposed = recurrent(inputs[:, position:position + 1], state)
        state = torch.where(valid[:, position][None, :, None], proposed, state)
        history.append(state[0])
    return torch.stack(history, 1)


class TypedLanguageModel(nn.Module):
    """One to four recurrent/SwiGLU layers, initialized jointly from scratch.

    Byte IDs are 3+byte (0..255), with PAD=0, BOS=1 and EOS=2. No constructor
    loads or copies learned weights. Every semantic mode allocates the same
    tensors in the same order for a chosen core/configuration and random seed.

    ``internal`` adds an independently learned 32-to-H evidence projection
    before every recurrent layer. ``output_only`` adds only the final such
    projection after the last recurrence, before its nonlinear SwiGLU. This
    allows query/evidence interactions without recurrent evidence access.
    ``none`` computes the same projections with zero routing coefficients.
    The exact workspace is identical across modes for the same event lane.
    Allocated tensors and initialization match, but active task capacity does
    not: internal activates every semantic projection, output_only only the
    last, and none activates none. Earlier output_only projections have zero
    task gradients. Architecture comparisons must disclose this distinction.

    ``projection_layout='shared'`` instead allocates one bias-free map reused
    at all internal sites or the output site. Empty evidence injects zero and
    internal/output-only active parameter counts match. Repeated internal use
    changes computation and gradients; it does not prove a shared meaning of
    the learned layers. The default per_layer layout preserves earlier receipts.

    ``recurrent_residual=True`` adds the PRE-semantic layer stream to its
    post-core residual, without parameters. Internal evidence still enters
    through recurrence, never through this skip. This preserves each core's
    own state contract; it gives no full-stack contraction or output bound.
    This wiring flag must be recorded alongside a checkpoint, like backend.

    SSM core states retain at least float32 recurrence precision even when
    the learned modules use lower precision. Each core's stated cube property
    remains conditional on its finite numerical domain. It does not establish
    contraction of the full stack, FFNs or heads.

    Language IDs never act as evidence events, including IDs 12..79. Events
    require a separate observed_event_ids tensor AND an explicit application-
    authenticated trusted_event_mask. A trust mask cannot authenticate itself.
    Supplying either event argument also requires an explicit application-owned
    attention_mask, so language PAD values cannot suppress observed events.
    No supplied recurrent state, oracle answer, or generated-event feedback API
    exists. Every call starts empty; the application may pass longer prefixes
    of generated language through the language lane alone.

    When policies are supplied, allowed_operands explicitly denotes workspace
    key IDs 0..3. Their authentication/currentness is an application assumption.
    Typed claims concern observed evidence, not external truth, query relevance
    or arbitrary text. Raw byte generation always remains uncertified.
    """

    def __init__(self, hidden_size: int = 128, native_size: int = 256, *,
                 num_layers: int = 2, recurrent_kind: str = "ssm",
                 semantic_mode: str = "internal", rho: float = .99,
                 ssm_backend: str = "sequential", projection_layout: str = "per_layer",
                 recurrent_residual: bool = False):
        super().__init__()
        if (type(hidden_size) is not int or hidden_size < 1
                or type(native_size) is not int or native_size < 1
                or type(num_layers) is not int or not 1 <= num_layers <= 4):
            raise ValueError("positive integer hidden/native sizes and one to four layers are required")
        if recurrent_kind not in ("ssm", "gru") or semantic_mode not in SEMANTIC_MODES:
            raise ValueError("invalid recurrent_kind or semantic_mode")
        if ssm_backend not in ("sequential", "parallel") or (recurrent_kind == "gru" and ssm_backend != "sequential"):
            raise ValueError("invalid SSM backend for this recurrent kind")
        if projection_layout not in ("per_layer", "shared"):
            raise ValueError("projection_layout must be per_layer or shared")
        if type(recurrent_residual) is not bool:
            raise ValueError("recurrent_residual must be bool")
        self.hidden_size, self.native_size, self.num_layers = hidden_size, native_size, num_layers
        self.recurrent_kind, self.semantic_mode = recurrent_kind, semantic_mode
        self.ssm_backend = ssm_backend
        self.projection_layout = projection_layout
        self.recurrent_residual = recurrent_residual
        # Backend is provenance, not learned state. Parallel association has a
        # measured FP32 cube overshoot (~8e-7 at saturation), not a strict FP
        # state certificate. Exact Boolean evidence/typed support is independent
        # of that numerical bound and still fails closed on nonfinite proposals.
        ssm_type = ParallelSelectiveSSM if ssm_backend == "parallel" else InvariantSelectiveSSM
        self.embedding = nn.Embedding(VOCAB_SIZE, hidden_size, padding_idx=PAD_ID)
        self.recurrent_layers = nn.ModuleList([
            ssm_type(hidden_size, rho=rho) if recurrent_kind == "ssm"
            else nn.GRU(hidden_size, hidden_size, batch_first=True)
            for _ in range(num_layers)])
        # One bias-free shared map prevents empty evidence from injecting an
        # unconditional learned bias. Its active count matches internal/output
        # routing; repeated internal uses still change gradient exposure.
        self.semantic_projections = nn.ModuleList(
            [nn.Linear(32, hidden_size, bias=False)] if projection_layout == "shared"
            else [nn.Linear(32, hidden_size) for _ in range(num_layers)])
        self.native_gates = nn.ModuleList([nn.Linear(hidden_size, native_size) for _ in range(num_layers)])
        self.native_values = nn.ModuleList([nn.Linear(hidden_size, native_size) for _ in range(num_layers)])
        self.native_downs = nn.ModuleList([nn.Linear(native_size, hidden_size, bias=False) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, VOCAB_SIZE)
        self.claim_head = nn.Linear(hidden_size, CLAIM_COUNT)
        self.evidence_lane = TypedEvidenceLane()
        self.evidence_decoder = EvidenceDecoder()

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
        # Text-only calls may infer padding. With an event lane, only an
        # explicit application mask owns validity, including PAD placeholders.
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
            # Keep the core capture in recurrence precision; narrow only the
            # bridge to the learned FFN and the following layer's projection.
            residual = core.to(self.native_gates[layer].weight.dtype)
            if self.recurrent_residual:
                residual = residual + stream.to(residual.dtype)
            if layer == self.num_layers - 1:
                residual = residual + semantic * (self.semantic_mode == "output_only")
            native = F.silu(self.native_gates[layer](residual)) * self.native_values[layer](residual)
            native_activations.append(native)
            stream = residual + self.native_downs[layer](native)
        raw_logits = self.lm_head(stream)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        claim_representation = stream[rows, positions]
        proposal_logits = self.claim_head(claim_representation)
        final_evidence = evidence[rows, positions].bool()
        grounded = (None if policies is None else
                    self.evidence_decoder(final_evidence, proposal_logits, policies=policies))
        return TypedLanguageOutput(raw_logits, evidence, tuple(core_states), tuple(native_activations),
                                   stream, claim_representation, proposal_logits, grounded,
                                   final_evidence, positions)
