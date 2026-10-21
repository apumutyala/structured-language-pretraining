"""Fresh CPU byte LM: learned prose and a disjoint protected event lane."""
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from controlled_lm.evidence_decoder import ABSTAIN_ID, EvidenceClaim
from controlled_lm.policy import READ, Policy
from controlled_lm.structured_decision import canonical_state_dict_digest
from controlled_lm.typed_language_model import SEMANTIC_MODES, TypedLanguageModel


ALLOW = Policy(caps=READ, budget=8, witness_valid=True, allowed_operands=(0, 1, 2, 3))


def test_parameter_free_recurrent_skip_preserves_language_path_but_cannot_bypass_internal_evidence_route():
    models = []
    for enabled in (False, True):
        torch.manual_seed(947)
        models.append(TypedLanguageModel(8, 13, num_layers=3, semantic_mode="internal",
                                        projection_layout="shared", recurrent_residual=enabled))
    assert canonical_state_dict_digest(models[0]) == canonical_state_dict_digest(models[1])
    assert sum(p.numel() for p in models[0].parameters()) == sum(p.numel() for p in models[1].parameters())
    ids = torch.tensor([[1, 68, 69, 70]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    trust = torch.tensor([[True, False, False, False]])
    for model in models:
        with torch.no_grad():
            for core in model.recurrent_layers:
                core.input_projection.weight.zero_(); core.input_projection.bias.zero_()
            for down in model.native_downs:
                down.weight.zero_()
        outputs = [model(ids, valid, observed_event_ids=torch.tensor([[event, 0, 0, 0]]),
                         trusted_event_mask=trust, policies=(ALLOW,)) for event in (12, 13)]
        # A skip from recurrent INPUTS (including semantic injection) would
        # violate this routing isolation when the core ignores its input.
        assert torch.equal(outputs[0].representation, outputs[1].representation)
        assert all(state.count_nonzero() == 0 for state in outputs[0].core_states)
        expected = model.embedding(ids) if model.recurrent_residual else torch.zeros_like(outputs[0].representation)
        torch.testing.assert_close(outputs[0].representation, expected, rtol=0, atol=0)
        assert outputs[0].grounded.probabilities[~outputs[0].grounded.allowed].count_nonzero() == 0
    with pytest.raises(ValueError, match="recurrent_residual must be bool"):
        TypedLanguageModel(recurrent_residual=1)


@pytest.mark.parametrize("kind", ["ssm", "gru"])
def test_shared_bias_free_map_matches_empty_evidence_paths_and_initialization(kind):
    models = []
    for mode in SEMANTIC_MODES:
        torch.manual_seed(937)
        models.append(TypedLanguageModel(8, 13, num_layers=3, recurrent_kind=kind,
                                        semantic_mode=mode, projection_layout="shared"))
    assert len({canonical_state_dict_digest(model) for model in models}) == 1
    ids = torch.tensor([[1, 68, 69, 70, 2]])
    plain = [model(ids).raw_logits for model in models]
    assert all(torch.equal(plain[0], output) for output in plain[1:])
    for model in models:
        assert len(model.semantic_projections) == 1
        assert model.semantic_projections[0].bias is None
        model.zero_grad(set_to_none=True)
        output = model(ids, torch.ones_like(ids, dtype=torch.bool),
                       observed_event_ids=torch.tensor([[0, 12, 0, 0, 0]]),
                       trusted_event_mask=torch.tensor([[False, True, False, False, False]]))
        output.raw_logits.square().mean().backward()
        magnitude = model.semantic_projections[0].weight.grad.abs().sum()
        assert (magnitude == 0) if model.semantic_mode == "none" else (magnitude > 0)


@pytest.mark.parametrize("kind", ["ssm", "gru"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_output_only_evidence_changes_last_nonlinearity_without_changing_recurrent_states(kind, dtype):
    torch.manual_seed(941)
    model = TypedLanguageModel(8, 13, num_layers=3, recurrent_kind=kind,
                              semantic_mode="output_only", projection_layout="shared").to(dtype)
    ids = torch.tensor([[1, 68, 69, 70, 2]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    trust = torch.tensor([[False, True, False, False, False]])
    outputs = [model(ids, valid, observed_event_ids=torch.tensor([[0, event, 0, 0, 0]]),
                     trusted_event_mask=trust, policies=(ALLOW,)) for event in (12, 13)]
    left, right = outputs
    assert all(torch.equal(a, b) for a, b in zip(left.core_states, right.core_states))
    assert all(torch.equal(a, b) for a, b in zip(left.native_activations[:-1], right.native_activations[:-1]))
    assert not torch.equal(left.native_activations[-1], right.native_activations[-1])
    for output in outputs:
        residual = output.core_states[-1].to(dtype) + model.semantic_projections[0](output.evidence_trace.flatten(2).to(dtype))
        native = F.silu(model.native_gates[-1](residual)) * model.native_values[-1](residual)
        torch.testing.assert_close(output.native_activations[-1], native, rtol=0, atol=0)
        torch.testing.assert_close(output.representation, residual + model.native_downs[-1](native), rtol=0, atol=0)
        torch.testing.assert_close(output.raw_logits, model.lm_head(output.representation), rtol=0, atol=0)
        torch.testing.assert_close(output.proposal_logits, model.claim_head(output.claim_representation), rtol=0, atol=0)
        assert output.grounded.probabilities[~output.grounded.allowed].count_nonzero() == 0
    left.raw_logits.float().square().mean().backward()
    assert model.semantic_projections[0].weight.grad.abs().sum() > 0


def test_output_only_can_interact_nonlinearly_with_query_instead_of_only_adding_a_fixed_logit_shift():
    torch.manual_seed(943)
    model = TypedLanguageModel(8, 13, num_layers=2, semantic_mode="output_only", projection_layout="shared").double()
    # Same last byte, different preceding query context; the direct evidence
    # contribution to a post-FFN linear head would be query-independent.
    ids = torch.tensor([[1, 68, 70], [1, 69, 70]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    trust = torch.tensor([[True, False, False], [True, False, False]])
    predictions = [model(ids, valid, observed_event_ids=torch.tensor([[event, 0, 0], [event, 0, 0]]),
                         trusted_event_mask=trust).raw_logits[:, -1] for event in (12, 13)]
    effect = predictions[0] - predictions[1]
    assert (effect[0] - effect[1]).abs().max() > 1e-8


def test_parallel_language_backend_uses_same_initial_tensors_and_nearby_computation():
    torch.manual_seed(57)
    sequential = TypedLanguageModel(8, 13, num_layers=2, projection_layout="shared")
    torch.manual_seed(57)
    parallel = TypedLanguageModel(8, 13, num_layers=2, projection_layout="shared", ssm_backend="parallel")
    assert canonical_state_dict_digest(sequential) == canonical_state_dict_digest(parallel)
    ids = torch.tensor([[1, 68, 69, 70, 2]])
    assert torch.allclose(sequential(ids).raw_logits, parallel(ids).raw_logits, atol=1e-6, rtol=1e-5)


def model_for(kind="ssm", mode="internal", layers=2, seed=37):
    torch.manual_seed(seed)
    return TypedLanguageModel(8, 13, num_layers=layers, recurrent_kind=kind, semantic_mode=mode)


@pytest.mark.parametrize("kind", ["ssm", "gru"])
@pytest.mark.parametrize("layers", [1, 2, 3, 4])
def test_modes_match_allocated_parameters_initialization_and_fresh_protected_lane(kind, layers):
    models = [model_for(kind, mode, layers) for mode in SEMANTIC_MODES]
    assert len({canonical_state_dict_digest(model) for model in models}) == 1
    assert list(dict(models[0].named_parameters())) == list(dict(models[1].named_parameters()))
    assert len({sum(parameter.numel() for parameter in model.parameters()) for model in models}) == 1
    assert not list(models[0].evidence_lane.parameters())
    assert not list(models[0].evidence_lane.buffers())
    assert not list(models[0].evidence_decoder.parameters())
    assert models[0].embedding.num_embeddings == 259


@pytest.mark.parametrize("kind", ["ssm", "gru"])
@pytest.mark.parametrize("mode", SEMANTIC_MODES)
def test_prefix_causality_of_prose_core_claims_and_events(kind, mode):
    model = model_for(kind, mode).double()
    prefix = torch.tensor([[1, 12, 79, 258]])
    event_prefix, trust_prefix = torch.tensor([[0, 19, 0, 23]]), torch.tensor([[False, True, False, True]])
    first = model(torch.cat((prefix, torch.tensor([[45, 2]])), 1), torch.ones(1, 6, dtype=torch.bool),
                  observed_event_ids=torch.cat((event_prefix, torch.tensor([[76, 0]])), 1),
                  trusted_event_mask=torch.cat((trust_prefix, torch.tensor([[True, False]])), 1))
    second = model(torch.cat((prefix, torch.tensor([[33, 2]])), 1), torch.ones(1, 6, dtype=torch.bool),
                   observed_event_ids=torch.cat((event_prefix, torch.tensor([[44, 0]])), 1),
                   trusted_event_mask=torch.cat((trust_prefix, torch.tensor([[True, False]])), 1))
    truncated = model(prefix, torch.ones_like(prefix, dtype=torch.bool),
                      observed_event_ids=event_prefix, trusted_event_mask=trust_prefix)
    torch.testing.assert_close(first.raw_logits[:, :4], second.raw_logits[:, :4], rtol=0, atol=0)
    torch.testing.assert_close(first.raw_logits[:, :4], truncated.raw_logits, rtol=1e-13, atol=1e-13)
    for core, other in zip(first.core_states, truncated.core_states):
        torch.testing.assert_close(core[:, :4], other, rtol=1e-13, atol=1e-13)
    assert torch.equal(first.evidence_trace[:, :4], truncated.evidence_trace)
    torch.testing.assert_close(truncated.proposal_logits,
                               model.claim_head(truncated.claim_representation), rtol=0, atol=0)


def test_language_token_collisions_and_generated_prefixes_never_write_evidence():
    model = model_for()
    ids = torch.tensor([[1, *range(12, 80), 2, 258]])
    first = model(ids, policies=(ALLOW,))
    assert first.evidence_trace.count_nonzero() == 0
    assert first.final_evidence.count_nonzero() == 0
    assert first.grounded.allowed[0, :32].count_nonzero() == 0
    # Even a raw head explicitly emitting event-like language ID19 cannot
    # write when its generated bytes are passed as a longer language prefix.
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[19] = 10.
    result = model(torch.tensor([[1, 19, 19]]), policies=(ALLOW,))
    assert result.raw_logits[0, -1].argmax() == 19
    assert result.evidence_trace.count_nonzero() == 0


def test_event_lane_requires_app_owned_validity_and_language_pad_cannot_suppress_facts():
    model = model_for()
    language_byte, language_pad = torch.tensor([[1, 3, 2]]), torch.tensor([[1, 0, 2]])
    events, trust = torch.tensor([[0, 19, 0]]), torch.tensor([[False, True, False]])
    valid = torch.ones_like(language_byte, dtype=torch.bool)
    byte = model(language_byte, valid, observed_event_ids=events, trusted_event_mask=trust)
    pad = model(language_pad, valid, observed_event_ids=events, trusted_event_mask=trust)
    assert torch.equal(byte.evidence_trace, pad.evidence_trace)
    assert byte.final_evidence[0, 0, 7] and pad.final_evidence[0, 0, 7]
    for ids in (language_byte, language_pad):
        for kwargs in ({'observed_event_ids': events}, {'trusted_event_mask': trust},
                       {'observed_event_ids': events, 'trusted_event_mask': trust}):
            with pytest.raises(ValueError, match='explicit application-owned attention_mask'):
                model(ids, **kwargs)


@pytest.mark.parametrize("kind", ["ssm", "gru"])
def test_trusted_events_and_independent_masks_only_write_at_valid_trusted_slots(kind):
    model = model_for(kind)
    ids = torch.tensor([[1, 19, 76, 45, 0, 258]])
    events = torch.tensor([[19, 44, 76, 23, 77, -999]])
    trust = torch.tensor([[True, True, True, True, True, False]])
    valid = torch.tensor([[True, False, False, True, False, True]])
    output = model(ids, valid, observed_event_ids=events, trusted_event_mask=trust, policies=(ALLOW,))
    assert output.final_positions.item() == 5
    assert output.final_evidence[0, 0, 7] and output.final_evidence[0, 1, 3]
    assert output.final_evidence.sum() == 2
    assert output.grounded.probabilities[0, 0] == 0
    changed_text = model(torch.tensor([[258, 44, 12, 7, 2, 1]]), valid,
                         observed_event_ids=events, trusted_event_mask=trust)
    assert torch.equal(output.evidence_trace, changed_text.evidence_trace)
    assert not torch.equal(output.raw_logits, changed_text.raw_logits)


@pytest.mark.parametrize("kind", ["ssm", "gru"])
def test_padding_holes_and_final_valid_position_match_compressed_sequences(kind):
    model = model_for(kind).double()
    ids = torch.tensor([[1, 0, 71, 2, 0], [1, 70, 2, 0, 0]])
    output = model(ids)
    assert output.final_positions.tolist() == [3, 2]
    compressed = model(torch.tensor([[1, 71, 2]]))
    torch.testing.assert_close(output.raw_logits[0, [0, 2, 3]], compressed.raw_logits[0],
                               rtol=1e-13, atol=1e-13)
    for core, shorter in zip(output.core_states, compressed.core_states):
        torch.testing.assert_close(core[0, [0, 2, 3]], shorter[0], rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(core[0, 4], core[0, 3], rtol=0, atol=0)
    # An explicit valid event-only slot may use a PAD language placeholder.
    event_only = model(torch.tensor([[0]]), torch.tensor([[True]]),
                       observed_event_ids=torch.tensor([[19]]), trusted_event_mask=torch.tensor([[True]]))
    assert event_only.final_evidence[0, 0, 7]




def test_low_precision_ssm_stack_preserves_float32_core_and_bridges_both_heads():
    model = model_for(layers=4).bfloat16()
    result = model(torch.tensor([[1, 68, 69, 2]]), policies=(ALLOW,))
    assert all(state.dtype == torch.float32 and state.abs().max() <= 1 for state in result.core_states)
    assert result.representation.dtype == torch.bfloat16
    assert result.raw_logits.dtype == result.proposal_logits.dtype == torch.bfloat16
    assert result.evidence_trace.dtype == torch.float32
    assert torch.isfinite(result.raw_logits).all()
    assert result.grounded.probabilities.dtype == torch.float64


def test_arbitrary_claim_and_raw_heads_do_not_turn_uncertified_prose_into_evidence():
    model = model_for()
    with torch.no_grad():
        model.claim_head.weight.zero_()
        model.claim_head.bias.zero_()
        model.claim_head.bias[0] = 1000.
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[ord('X') + 3] = 1000.
    output = model(torch.tensor([[1, 19, 2]]), torch.ones(1, 3, dtype=torch.bool),
                   observed_event_ids=torch.tensor([[19, 0, 0]]),
                   trusted_event_mask=torch.tensor([[True, False, False]]), policies=(ALLOW,))
    assert output.proposal_logits.argmax().item() == 0
    assert output.raw_logits[0, -1].argmax().item() == ord('X') + 3
    assert output.final_evidence[0, 0, 7]
    assert output.grounded.probabilities[0, 0] == 0
    assert output.grounded.allowed.gather(1, output.grounded.sample_ids(1000)).all()
    with torch.no_grad():
        model.claim_head.bias[0] = float('nan')
    failed = model(torch.tensor([[1, 2]]), policies=(ALLOW,))
    assert failed.grounded.argmax() == (EvidenceClaim('abstain'),)


def test_optional_policies_explicit_scope_and_no_oracle_feedback_interfaces():
    model = model_for()
    ids = torch.tensor([[1, 2]])
    assert model(ids).grounded is None
    scoped = model(ids, policies=(replace(ALLOW, allowed_operands=(1,)),))
    assert set(scoped.grounded.allowed[0].nonzero().flatten().tolist()) == {33, ABSTAIN_ID}
    for kwargs in ({'oracle_answer': 7}, {'semantic_state': torch.zeros(1, 4, 8)},
                   {'generated_event_ids': torch.tensor([[19, 0]])}, {'hx': torch.zeros(1, 1, 8)}):
        with pytest.raises(TypeError):
            model(ids, **kwargs)
    with pytest.raises(ValueError):
        model(ids, observed_event_ids=torch.tensor([[19, 0]]))
    with pytest.raises(ValueError):
        model(torch.tensor([[0, 0]]))
    for bad in (torch.tensor([[259]]), torch.tensor([[-1]]), torch.tensor([[1.]])):
        with pytest.raises(ValueError):
            model(bad)


def test_small_fresh_model_parameter_budget_and_configuration_bounds():
    model = TypedLanguageModel(32, 64, num_layers=2)
    assert 10000 < sum(parameter.numel() for parameter in model.parameters()) < 100000
    assert canonical_state_dict_digest(model) != canonical_state_dict_digest(TypedLanguageModel(32, 64, num_layers=2))
    for layers in (0, 5, True):
        with pytest.raises(ValueError):
            TypedLanguageModel(num_layers=layers)


@pytest.mark.parametrize("kind", ["ssm", "gru"])
@pytest.mark.parametrize("mode", SEMANTIC_MODES)
def test_allocated_matching_does_not_imply_equal_active_semantic_projection_capacity(kind, mode):
    model = model_for(kind, mode, layers=3)
    ids = torch.tensor([[1, 68, 69, 2]])
    output = model(ids, torch.ones_like(ids, dtype=torch.bool),
                   observed_event_ids=torch.tensor([[19, 0, 23, 0]]),
                   trusted_event_mask=torch.tensor([[True, False, True, False]]))
    loss = F.cross_entropy(output.raw_logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten())
    loss = loss + F.cross_entropy(output.proposal_logits, torch.tensor([7]))
    loss.backward()
    active = [projection.weight.grad is not None and bool(projection.weight.grad.abs().sum() > 0)
              for projection in model.semantic_projections]
    assert active == ([True, True, True] if mode == 'internal' else
                      [False, False, True] if mode == 'output_only' else [False, False, False])
