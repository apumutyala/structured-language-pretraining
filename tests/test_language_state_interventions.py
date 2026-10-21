"""CPU-only full-layout neural-state impulses with protected SOURCE evidence."""
import pytest
import torch

from controlled_lm.language_state_interventions import replay_language_state_impulse
from controlled_lm.policy import Policy, READ
from controlled_lm.structured_decision import canonical_state_dict_digest
from controlled_lm.typed_language_model import TypedLanguageModel


def fixture(dtype=torch.float64, mode="internal", **kwargs):
    torch.manual_seed(9103)
    config = dict(num_layers=4, recurrent_kind="ssm", ssm_backend="parallel",
                  projection_layout="shared", recurrent_residual=True)
    config.update(kwargs)
    model = TypedLanguageModel(5, 8, semantic_mode=mode, **config).to(dtype).eval()
    ids = torch.tensor([[1, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 0, 0],
                        [1, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]])
    valid = ids != 0
    valid[0, 3] = False
    events = torch.zeros_like(ids)
    trust = torch.zeros_like(ids, dtype=torch.bool)
    events[:, [1, 2, 5, 7]] = torch.tensor([19, 23, 30, 35])
    trust[:, [1, 2, 5, 7]] = True
    allow = Policy(caps=READ, budget=8, witness_valid=True, allowed_operands=(0, 1, 2, 3))
    return model, ids, valid, dict(observed_event_ids=events, trusted_event_mask=trust, policies=(allow, allow))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("mode", ["none", "internal", "output_only"])
@pytest.mark.parametrize("layer", range(4))
def test_zero_impulse_identity_all_modes_layers_and_precisions(dtype, mode, layer):
    model, ids, valid, lane = fixture(dtype, mode)
    before = canonical_state_dict_digest(model)
    result = replay_language_state_impulse(model, ids, valid, layer=layer, time=torch.tensor([5, 8]),
                                            impulse=torch.zeros(2, 5, dtype=dtype), **lane)
    tolerance = 1e-12 if dtype == torch.float64 else 1e-5
    assert result.identity_tolerance == tolerance
    assert result.identity_max_abs_error <= tolerance
    torch.testing.assert_close(result.source.raw_logits, result.replay.raw_logits, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(result.source.proposal_logits, result.replay.proposal_logits, atol=tolerance, rtol=tolerance)
    assert torch.equal(result.source.evidence_trace, result.replay.evidence_trace)
    assert torch.count_nonzero(result.realized_l2) == 0
    assert canonical_state_dict_digest(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_nonzero_impulse_actual_radius_causality_source_bank_and_full_shape():
    model, ids, valid, lane = fixture()
    saved_ids, saved_events = ids.clone(), lane["observed_event_ids"].clone()
    digest = canonical_state_dict_digest(model)
    impulse = torch.tensor([[.1, 0., 0., 0., 0.], [-.2, 0., 0., 0., 0.]], dtype=torch.float64)
    times = torch.tensor([5, 8])
    shapes = []
    handle = model.recurrent_layers[0].input_projection.register_forward_pre_hook(
        lambda module, arguments: shapes.append(tuple(arguments[0].shape)))
    try:
        result = replay_language_state_impulse(model, ids, valid, layer=0, time=times, impulse=impulse, **lane)
    finally:
        handle.remove()
    assert shapes == [(2, 13, 5)] * 3
    torch.testing.assert_close(result.realized_impulse, impulse, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(result.realized_l2, impulse.norm(dim=-1), atol=1e-14, rtol=1e-12)
    for row, time in enumerate(times.tolist()):
        assert torch.equal(result.source.raw_logits[row, :time], result.replay.raw_logits[row, :time])
        for source, patched in zip(result.source.core_states, result.replay.core_states):
            assert torch.equal(source[row, :time], patched[row, :time])
    assert not torch.equal(result.source.raw_logits, result.replay.raw_logits)
    assert torch.equal(result.source.evidence_trace, result.replay.evidence_trace)
    assert torch.equal(result.source.grounded.allowed, result.replay.grounded.allowed)
    assert result.replay.grounded.probabilities[~result.replay.grounded.allowed].count_nonzero() == 0
    assert canonical_state_dict_digest(model) == digest
    assert torch.equal(ids, saved_ids) and torch.equal(lane["observed_event_ids"], saved_events)


def test_real_affine_impulse_formula_and_no_hidden_state_clamp():
    model, ids, valid, lane = fixture()
    impulse = torch.full((2, 5), .15, dtype=torch.float64)
    result = replay_language_state_impulse(model, ids, valid, layer=0, time=2, impulse=impulse, **lane)
    inputs = model.embedding(ids) + model.semantic_projections[0](result.source.evidence_trace.flatten(2).double())
    gate = model.recurrent_layers[0].input_projection(inputs).chunk(2, -1)[0]
    coefficients = model.recurrent_layers[0].rho * gate.sigmoid()
    coefficients = torch.where(valid[..., None], coefficients, 1.)
    for time in range(2, ids.shape[1]):
        expected = impulse * coefficients[:, 3:time + 1].prod(1)
        actual = result.replay.core_states[0][:, time] - result.source.core_states[0][:, time]
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-11)
    large = replay_language_state_impulse(model, ids, valid, layer=3, time=torch.tensor([10, 12]),
                                          impulse=torch.full((2, 5), 2., dtype=torch.float64), **lane)
    assert large.replay.core_states[3].abs().max() > 1


def test_float32_reports_realized_quantization_instead_of_requested_radius():
    model, ids, valid, lane = fixture(torch.float32)
    result = replay_language_state_impulse(model, ids, valid, layer=0, time=2,
                                          impulse=torch.full((2, 5), 1e-12, dtype=torch.float32), **lane)
    assert torch.all(result.requested_l2 > 0)
    assert torch.count_nonzero(result.realized_l2) == 0
    assert torch.count_nonzero(result.realized_impulse) == 0


def test_unsupported_source_version_rejects_before_replay(monkeypatch):
    import controlled_lm.language_state_interventions as module
    model, ids, valid, lane = fixture()
    monkeypatch.setattr(module, "CORE_FORWARD_SHA256", "unreviewed")
    with pytest.raises(RuntimeError, match="unsupported model/core source version"):
        replay_language_state_impulse(model, ids, valid, layer=0, time=2,
                                      impulse=torch.zeros(2, 5, dtype=torch.float64), **lane)


@pytest.mark.parametrize("kwargs", [{"num_layers": 2}, {"recurrent_kind": "gru", "ssm_backend": "sequential"},
    {"ssm_backend": "sequential"}, {"projection_layout": "per_layer"}, {"recurrent_residual": False}])
def test_other_configurations_explicitly_rejected(kwargs):
    model, ids, valid, lane = fixture(**kwargs)
    with pytest.raises(ValueError, match="only four-layer"):
        replay_language_state_impulse(model, ids, valid, layer=0, time=2, impulse=torch.zeros(2, 5, dtype=torch.float64), **lane)


def test_malformed_and_nonfinite_impulses_and_boundaries_reject():
    model, ids, valid, lane = fixture()
    for impulse in [torch.zeros(5, dtype=torch.float64), torch.zeros(2, 5),
                    torch.full((2, 5), float("nan")), torch.full((2, 5), float("inf"))]:
        with pytest.raises(ValueError, match="impulse"):
            replay_language_state_impulse(model, ids, valid, layer=0, time=2, impulse=impulse, **lane)
    with pytest.raises(ValueError, match="L2"):
        replay_language_state_impulse(model, ids, valid, layer=0, time=2,
                                      impulse=torch.full((2, 5), 1e200, dtype=torch.float64), **lane)
    for time in [-1, 13, 3, 1.5, torch.tensor([True, False]), torch.empty(2, device="meta")]:
        with pytest.raises(ValueError, match="time"):
            replay_language_state_impulse(model, ids, valid, layer=0, time=time,
                                          impulse=torch.zeros(2, 5, dtype=torch.float64), **lane)
    model.train()
    with pytest.raises(ValueError, match="eval"):
        replay_language_state_impulse(model, ids, valid, layer=0, time=2,
                                      impulse=torch.zeros(2, 5, dtype=torch.float64), **lane)
