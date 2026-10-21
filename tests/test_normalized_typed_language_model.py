"""Small CPU forward/backward checks only; no optimizer steps or GPU access."""
import importlib
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from controlled_lm.evidence_decoder import ABSTAIN_ID
from controlled_lm.normalized_typed_language_model import (
    NormalizedTypedLanguageModel, parameter_free_rms,
)
from controlled_lm.policy import READ, Policy
from controlled_lm.structured_decision import canonical_state_dict_digest
from controlled_lm.typed_language_model import SEMANTIC_MODES, TypedLanguageModel

ROOT = Path(__file__).resolve().parents[1]
ALLOW = Policy(caps=READ, budget=8, witness_valid=True, allowed_operands=(0, 1, 2, 3))
DENY = Policy(caps=0, budget=8, witness_valid=True, allowed_operands=(0, 1, 2, 3))


def make_model(mode="internal", backend="parallel", layers=3):
    torch.manual_seed(2026102131)
    return NormalizedTypedLanguageModel(8, 13, num_layers=layers, semantic_mode=mode,
        ssm_backend=backend, projection_layout="shared", recurrent_residual=True)


def event_call(model, event=19, policies=None):
    ids = torch.tensor([[1, 68, 69, 2]])
    return model(ids, torch.ones_like(ids, dtype=torch.bool),
        observed_event_ids=torch.tensor([[event, 0, 0, 0]]),
        trusted_event_mask=torch.tensor([[True, False, False, False]]), policies=policies)


@pytest.mark.parametrize("backend", ["sequential", "parallel"])
def test_no_added_learned_parameters_and_paired_fresh_initialization(backend):
    models = [make_model(mode, backend) for mode in SEMANTIC_MODES]
    torch.manual_seed(2026102131)
    original = TypedLanguageModel(8, 13, num_layers=3, semantic_mode="none", ssm_backend=backend,
        projection_layout="shared", recurrent_residual=True)
    assert len({canonical_state_dict_digest(model) for model in [original, *models]}) == 1
    assert all(list(model.state_dict()) == list(original.state_dict()) for model in models)
    assert all(list(dict(model.named_parameters())) == list(dict(original.named_parameters())) for model in models)
    assert all(sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in original.parameters()) for model in models)
    ids = torch.tensor([[1, *range(12, 80), 2]])
    outputs = [model(ids, policies=(ALLOW,)) for model in models]
    assert all(torch.equal(outputs[0].raw_logits, output.raw_logits) for output in outputs[1:])
    for output in outputs:
        assert output.evidence_trace.count_nonzero() == 0
        assert output.final_evidence.count_nonzero() == 0
        assert output.grounded.allowed[0, :32].count_nonzero() == 0


@pytest.mark.parametrize("mode", SEMANTIC_MODES)
def test_explicit_ffn_and_both_head_normalization_equations(mode):
    model = make_model(mode).double()
    ids = torch.tensor([[1, 68, 69, 2]])
    output = event_call(model)
    stream = model.embedding(ids)
    for layer in range(model.num_layers):
        semantic = model.semantic_projections[0](output.evidence_trace.flatten(2).double())
        inputs = stream + semantic * (mode == "internal")
        core = model.recurrent_layers[layer](inputs, mask=torch.ones_like(ids, dtype=torch.bool))[0]
        torch.testing.assert_close(core, output.core_states[layer], rtol=1e-13, atol=1e-13)
        residual = core + stream
        if layer == model.num_layers - 1:
            residual = residual + semantic * (mode == "output_only")
        expected = residual / (residual.square().mean(-1, keepdim=True) + 1e-5).sqrt()
        native = F.silu(model.native_gates[layer](expected)) * model.native_values[layer](expected)
        torch.testing.assert_close(native, output.native_activations[layer], rtol=1e-13, atol=1e-13)
        stream = residual + model.native_downs[layer](native)
    torch.testing.assert_close(output.representation, stream, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(output.raw_logits, model.lm_head(parameter_free_rms(stream)), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(output.proposal_logits, model.claim_head(parameter_free_rms(stream[:, -1])), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("mode", SEMANTIC_MODES)
def test_gradient_routes_and_output_only_recurrent_isolation(mode):
    model = make_model(mode)
    left, right = event_call(model, 19), event_call(model, 20)
    if mode in ("none", "output_only"):
        assert all(torch.equal(a, b) for a, b in zip(left.core_states, right.core_states))
    if mode == "none":
        assert torch.equal(left.raw_logits, right.raw_logits)
    else:
        assert not torch.equal(left.raw_logits, right.raw_logits)
    if mode == "output_only":
        assert all(torch.equal(a, b) for a, b in zip(left.native_activations[:-1], right.native_activations[:-1]))
    loss = F.cross_entropy(left.raw_logits.flatten(0, 1), torch.tensor([68, 69, 70, 2]))
    loss = loss + F.cross_entropy(left.proposal_logits, torch.tensor([7]))
    loss.backward()
    semantic_gradient = model.semantic_projections[0].weight.grad
    assert semantic_gradient is not None
    assert (semantic_gradient.count_nonzero() == 0) if mode == "none" else (semantic_gradient.abs().sum() > 0)
    for module in [model.embedding, *model.recurrent_layers, *model.native_gates, *model.native_values,
                   *model.native_downs, model.lm_head, model.claim_head]:
        gradients = [parameter.grad for parameter in module.parameters()]
        assert gradients and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_internal_evidence_cannot_bypass_disabled_recurrences_via_residual():
    model = make_model()
    with torch.no_grad():
        for core in model.recurrent_layers:
            core.input_projection.weight.zero_()
            core.input_projection.bias.zero_()
    left, right = event_call(model, 19), event_call(model, 20)
    assert all(core.count_nonzero() == 0 for core in left.core_states)
    assert torch.equal(left.representation, right.representation)
    assert torch.equal(left.raw_logits, right.raw_logits)


@pytest.mark.parametrize("mode", SEMANTIC_MODES)
def test_typed_boundary_and_event_authentication_assumptions_are_preserved(mode):
    model = make_model(mode)
    ids = torch.tensor([[1, 19, 19, 2]])
    empty = model(ids, policies=(ALLOW,))
    assert empty.final_evidence.count_nonzero() == 0
    for kwargs in ({"observed_event_ids": ids}, {"trusted_event_mask": ids.bool()}):
        with pytest.raises(ValueError, match="explicit application-owned attention_mask"):
            model(ids, **kwargs)
    untrusted = model(ids, torch.ones_like(ids, dtype=torch.bool), observed_event_ids=ids,
                      trusted_event_mask=torch.zeros_like(ids, dtype=torch.bool), policies=(ALLOW,))
    assert torch.equal(empty.evidence_trace, untrusted.evidence_trace)
    allowed, denied = event_call(model, policies=(ALLOW,)), event_call(model, policies=(DENY,))
    assert allowed.final_evidence[0, 0, 7]
    assert allowed.grounded.probabilities[~allowed.grounded.allowed].count_nonzero() == 0
    assert denied.grounded.probabilities[0, ABSTAIN_ID] == 1
    assert denied.grounded.probabilities[0, :ABSTAIN_ID].count_nonzero() == 0
    with torch.no_grad():
        model.claim_head.weight[0, 0] = torch.nan
    failed = event_call(model, policies=(ALLOW,))
    assert failed.grounded.probabilities[0, ABSTAIN_ID] == 1
    assert failed.grounded.probabilities[0, :ABSTAIN_ID].count_nonzero() == 0


@pytest.mark.parametrize("backend", ["sequential", "parallel"])
def test_padding_holes_and_prefix_causality(backend):
    model = make_model(backend=backend).double()
    padded = model(torch.tensor([[1, 0, 68, 2, 0]]))
    compact = model(torch.tensor([[1, 68, 2]]))
    assert padded.final_positions.tolist() == [3]
    torch.testing.assert_close(padded.raw_logits[:, [0, 2, 3]], compact.raw_logits, rtol=1e-13, atol=1e-13)
    long = model(torch.tensor([[1, 68, 2, 80, 81]]))
    torch.testing.assert_close(long.raw_logits[:, :3], compact.raw_logits, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_rms_formula_zero_gradients_and_finite_domain(dtype):
    values = torch.tensor([[0., 0., 0.], [1e-9, -2e-9, 3e-9], [1e6, -2e6, 3e6]], dtype=dtype, requires_grad=True)
    result = parameter_free_rms(values)
    work = values.double() if dtype == torch.float64 else values.float()
    reference = (work / (work.square().mean(-1, keepdim=True) + 1e-5).sqrt()).to(dtype)
    torch.testing.assert_close(result, reference)
    assert result.dtype == dtype and torch.isfinite(result).all()
    assert result[0].count_nonzero() == 0
    assert (result.float().square().mean(-1) <= 1.01).all()
    result.sum().backward()
    assert torch.isfinite(values.grad).all()
    for nonfinite in (float("nan"), float("inf"), -float("inf")):
        assert not torch.isfinite(parameter_free_rms(torch.tensor([[nonfinite, 1.]]))).all()
    for invalid in (torch.tensor([[1, 2]]), torch.empty(2, 0), torch.tensor(1.)):
        with pytest.raises(ValueError, match="RMS input"):
            parameter_free_rms(invalid)


def test_scaled_rms_handles_extreme_finite_values_without_square_overflow_zeroing():
    tiny = torch.nextafter(torch.tensor(0.), torch.tensor(1.))
    values = torch.tensor([[1e38, -1e38, 5e37], [tiny, -tiny, 0.], [0., 0., 0.]], requires_grad=True)
    output = parameter_free_rms(values)
    assert torch.isfinite(output).all()
    assert output[0].count_nonzero() == 3
    torch.testing.assert_close(output[0].square().mean(), torch.tensor(1.), atol=1e-6, rtol=1e-6)
    assert output[1, :2].count_nonzero() == 2
    assert output[2].count_nonzero() == 0
    output.sum().backward()
    assert torch.isfinite(values.grad).all()
    huge_double = parameter_free_rms(torch.tensor([[1e300, -1e300]], dtype=torch.float64))
    torch.testing.assert_close(huge_double, torch.tensor([[1., -1.]], dtype=torch.float64))


def test_scaled_rms_gradcheck_and_moderate_formula_gradient_equivalence():
    values = torch.tensor([[.2, -.7, 1.3], [1e-4, -2e-4, 3e-4]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(parameter_free_rms, (values,), eps=1e-7, atol=1e-5, rtol=1e-4)
    actual = parameter_free_rms(values)
    reference = values / (values.square().mean(-1, keepdim=True) + 1e-5).sqrt()
    torch.testing.assert_close(actual, reference, rtol=1e-13, atol=1e-13)
    actual_gradient = torch.autograd.grad(actual.square().sum(), values, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference.square().sum(), values)[0]
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-10, atol=1e-10)


def test_normalized_wrapper_binds_private_trainer_without_training_or_global_mutation(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    wrapper = importlib.import_module("train_normalized_language_pilot")
    original = importlib.import_module("train_language_pilot")
    original_class, original_sources, original_probe = original.TypedLanguageModel, original.sources, original.training_probe
    config = json.loads((ROOT / "configs/normalized_language_local_20261021.json").read_text())
    private = wrapper.load_bound_trainer(config, "cuda")
    assert private is not original and private.TypedLanguageModel is NormalizedTypedLanguageModel
    assert original.TypedLanguageModel is original_class and original.sources is original_sources and original.training_probe is original_probe
    source_paths = {path.replace("\\", "/") for path in private.sources()}
    assert {"tools/train_language_pilot.py", "tools/train_normalized_language_pilot.py",
            "src/controlled_lm/typed_language_model.py", "src/controlled_lm/normalized_typed_language_model.py"} <= source_paths
    for edits in ({"resume_reference_report": "unused.json"}, {"precision": "bf16_autocast"},
                  {"semantic_modes": ["output_only"]}, {"updates": 1001},
                  {"normalization_epsilon": 1e-4}, {"max_allocated_bytes": 9 * 1024 ** 3}):
        with pytest.raises(ValueError):
            wrapper.validate_config(dict(config, **edits), "cuda")
    for name, device in (("cpu_enablement", "cpu"), ("cuda_qualification", "cuda")):
        candidate = json.loads((ROOT / f"configs/normalized_language_{name}_20261021.json").read_text())
        wrapper.validate_config(candidate, device)
