"""CPU diagnostic impulses in the actual four-layer parallel-SSM byte LM.

For fixed inputs at one layer, h_t=a_t*h_(t-1)+b_t. Adding delta to b_tau
after validity masking is, in exact real arithmetic, h_tau += delta; later
differences at that layer are delta*product(a_(tau+1)..a_t). Other layers
are recomputed through their actual cores, residual skips, FFNs and heads.
Their nonlinear/input-dependent dynamics have no such contraction claim.

The copied scan below pins the inspected forward-source versions and keeps
the original [B,T,H] projection and Hillis-Steele association. Every call
checks zero-impulse identity against actual model.forward before measuring a
patch. This is CPU FP32/FP64 numerical evidence, not a compiler/CUDA refinement
proof or a semantic-circuit claim. The exact observed evidence and current
application policies are held at SOURCE. No labels or oracle answers enter.

No clipping is performed. Finite impulses can leave the numerical state cube;
this diagnostic does not certify admissibility. A later matched campaign must
check its chosen state envelope and realized L2 controls separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .parallel_ssm import ParallelSelectiveSSM
from .typed_language_model import PAD_ID, TypedLanguageModel, TypedLanguageOutput


CORE_FORWARD_SHA256 = "22a792e27ac3ec1e0e14e3648ee7eecbb7c5eec1d46b4fc0ae231354209fb366"
MODEL_FORWARD_SHA256 = "49b93a7a5ce2cbe5658221230e1598674c774f67eb09cba7e8dc1d566b25aa8a"
IDENTITY_TOLERANCES = {torch.float64: 1e-12, torch.float32: 1e-5}


@dataclass
class LanguageStateIntervention:
    source: TypedLanguageOutput
    replay: TypedLanguageOutput
    layer: int
    time_indices: Tensor
    requested_impulse: Tensor
    realized_impulse: Tensor
    requested_l2: Tensor
    realized_l2: Tensor
    identity_max_abs_error: float
    identity_tolerance: float
    source_versions: dict[str, str]


def _source_hash(function):
    return hashlib.sha256(inspect.getsource(function).replace("\r\n", "\n").encode()).hexdigest()


def _configuration(model, input_ids):
    if (type(model) is not TypedLanguageModel or model.num_layers != 4 or model.recurrent_kind != "ssm"
            or model.ssm_backend != "parallel" or model.projection_layout != "shared"
            or model.recurrent_residual is not True
            or any(type(core) is not ParallelSelectiveSSM for core in model.recurrent_layers)):
        raise ValueError("only four-layer parallel SSM/shared projection/recurrent residual TypedLanguageModel is supported")
    if model.training:
        raise ValueError("model must already be in eval mode")
    if (not isinstance(input_ids, Tensor) or input_ids.device.type != "cpu"
            or any(parameter.device.type != "cpu" for parameter in model.parameters())):
        raise ValueError("diagnostic replay is CPU only")
    dtypes = {parameter.dtype for parameter in model.parameters()}
    if len(dtypes) != 1 or next(iter(dtypes)) not in IDENTITY_TOLERANCES or torch.is_autocast_enabled("cpu"):
        raise ValueError("use uniform CPU float32/float64 parameters without autocast")
    versions = {"core_forward_sha256": _source_hash(ParallelSelectiveSSM.forward),
                "model_forward_sha256": _source_hash(TypedLanguageModel.forward),
                "replay_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if versions["core_forward_sha256"] != CORE_FORWARD_SHA256 or versions["model_forward_sha256"] != MODEL_FORWARD_SHA256:
        raise RuntimeError("unsupported model/core source version; inspect scan and continuation correspondence before updating pins")
    return next(iter(dtypes)), versions


def _impulse_scan(core, inputs, valid, times, impulse):
    """Mirror the pinned core.forward graph, with one explicit leaf-b edit."""
    batch, time, _ = inputs.shape
    projected = core.input_projection(inputs)
    dtype = torch.float64 if projected.dtype == torch.float64 else torch.float32
    gate, candidate = projected.to(dtype).chunk(2, dim=-1)
    a = core.rho * gate.sigmoid()
    b = (1 - a) * candidate.tanh()
    initial = torch.zeros(batch, core.hidden_size, device=inputs.device, dtype=dtype)
    a = torch.where(valid[:, :, None], a, 1)
    b = torch.where(valid[:, :, None], b, 0)
    b = b.clone()
    rows = torch.arange(batch)
    b[rows, times] = b[rows, times] + impulse
    offset = 1
    while offset < time:
        next_b = torch.cat((b[:, :offset], b[:, offset:] + a[:, offset:] * b[:, :-offset]), dim=1)
        a = torch.cat((a[:, :offset], a[:, offset:] * a[:, :-offset]), dim=1)
        b = next_b
        offset *= 2
    return b + a * initial[:, None]


def _continue(model, input_ids, valid, source, layer, times, impulse, policies):
    evidence = source.evidence_trace.clone()  # Protected SOURCE, never donor evidence.
    stream = model.embedding(input_ids)
    states, natives = [], []
    for index, core in enumerate(model.recurrent_layers):
        semantic = model.semantic_projections[0](evidence.flatten(2).to(stream.dtype))
        inputs = stream + semantic * (model.semantic_mode == "internal")
        state = (_impulse_scan(core, inputs, valid, times, impulse) if index == layer
                 else core(inputs, mask=valid)[0])
        states.append(state)
        residual = state.to(model.native_gates[index].weight.dtype)
        residual = residual + stream.to(residual.dtype)
        if index == model.num_layers - 1:
            residual = residual + semantic * (model.semantic_mode == "output_only")
        native = F.silu(model.native_gates[index](residual)) * model.native_values[index](residual)
        natives.append(native)
        stream = residual + model.native_downs[index](native)
    raw_logits = model.lm_head(stream)
    rows = torch.arange(input_ids.shape[0])
    representation = stream[rows, source.final_positions]
    proposals = model.claim_head(representation)
    final_evidence = evidence[rows, source.final_positions].bool()
    grounded = model.evidence_decoder(final_evidence, proposals, policies=policies)
    output = TypedLanguageOutput(raw_logits, evidence, tuple(states), tuple(natives), stream,
                                representation, proposals, grounded, final_evidence, source.final_positions.clone())
    if not all(torch.isfinite(value).all() for value in [*states, *natives, stream, raw_logits, proposals]):
        raise FloatingPointError("replay produced nonfinite state/representation/logits")
    return output


def _identity(source, replay, tolerance):
    for name in ("evidence_trace", "final_evidence", "final_positions"):
        if not torch.equal(getattr(source, name), getattr(replay, name)):
            raise AssertionError(f"zero-impulse identity changed protected {name}")
    if not torch.equal(source.grounded.allowed, replay.grounded.allowed):
        raise AssertionError("zero-impulse identity changed typed support")
    pairs = [(source.raw_logits, replay.raw_logits), (source.proposal_logits, replay.proposal_logits),
             (source.representation, replay.representation), (source.claim_representation, replay.claim_representation),
             (source.grounded.probabilities, replay.grounded.probabilities)]
    pairs += list(zip(source.core_states, replay.core_states)) + list(zip(source.native_activations, replay.native_activations))
    error = 0.
    for actual, continued in pairs:
        if not torch.isfinite(actual).all() or not torch.allclose(actual, continued, atol=tolerance, rtol=0):
            raise AssertionError("zero-impulse full-layout identity failed at the fixed precision tolerance")
        error = max(error, float((actual.double() - continued.double()).abs().max()))
    return error


@torch.no_grad()
def replay_language_state_impulse(model, input_ids, attention_mask, *, layer, time, impulse,
                                  observed_event_ids=None, trusted_event_mask=None, policies):
    """Inject [B,H] into local b at a valid post-token boundary, then continue.

    `time` is a zero-based integer scalar or [B] vector, with every selected
    slot valid. `impulse` must already match the core's CPU FP32/FP64 dtype.
    The return includes source/replayed full trajectories, actual heads, and
    requested versus realized post-core displacement and L2 (measured in FP64).
    Policies are the same application-owned current snapshots for both passes.
    Identity thresholds are absolute (rtol=0): 1e-12 FP64, 1e-5 FP32.
    No tolerance override, clipping, donor workspace or oracle-label input is
    available. Registered hooks or arbitrary Python/module mutation are outside
    the source-version contract; zero-identity still runs before every patch.
    """
    dtype, versions = _configuration(model, input_ids)
    if type(layer) is not int or not 0 <= layer < 4:
        raise ValueError("layer must be integer0..3")
    if (not isinstance(impulse, Tensor) or impulse.device.type != "cpu" or impulse.dtype != dtype
            or input_ids.ndim != 2 or impulse.shape != (input_ids.shape[0], model.hidden_size)
            or not torch.isfinite(impulse).all()):
        raise ValueError("impulse must be finite CPU [batch,hidden] in the model dtype")
    requested_l2 = impulse.double().norm(dim=-1)
    if not torch.isfinite(requested_l2).all():
        raise ValueError("impulse L2 must be finite")
    source = model(input_ids, attention_mask, observed_event_ids=observed_event_ids,
                   trusted_event_mask=trusted_event_mask, policies=policies)
    if source.grounded is None:
        raise ValueError("provide current policies to measure the actual typed output")
    valid = input_ids != PAD_ID if attention_mask is None else attention_mask
    if isinstance(time, Tensor) and time.device.type != "cpu":
        raise ValueError("intervention time must be on CPU")
    times = torch.as_tensor(time, device="cpu")
    if times.ndim == 0:
        times = times.expand(input_ids.shape[0])
    if times.shape != (input_ids.shape[0],) or times.dtype not in (torch.int32, torch.int64):
        raise ValueError("time must be an integer scalar or vector[batch]")
    rows = torch.arange(input_ids.shape[0])
    if (bool(((times < 0) | (times > source.final_positions)).any()) or not bool(valid[rows, times].all())):
        raise ValueError("every intervention time must select a valid observed slot")
    tolerance = IDENTITY_TOLERANCES[dtype]
    identity = _continue(model, input_ids, valid, source, layer, times, torch.zeros_like(impulse), policies)
    identity_error = _identity(source, identity, tolerance)
    replay = _continue(model, input_ids, valid, source, layer, times, impulse, policies)
    if not torch.equal(source.evidence_trace, replay.evidence_trace):
        raise AssertionError("impulse edited the protected source evidence")
    realized = replay.core_states[layer][rows, times].double() - source.core_states[layer][rows, times].double()
    realized_l2 = realized.norm(dim=-1)
    if not torch.isfinite(realized_l2).all():
        raise FloatingPointError("realized impulse L2 is nonfinite")
    return LanguageStateIntervention(source, replay, layer, times.clone(), impulse.clone(), realized,
                                     requested_l2, realized_l2, identity_error, tolerance, versions)
