import torch

from controlled_lm.record_learning_comparison import RecordProposer, action_loss, generate_world, oracle_trajectory
from controlled_lm.record_placement_diagnostic import RecordPlacementProposer


def test_new_internal_exactly_reproduces_frozen_small_model():
    torch.set_num_threads(1)
    observation, _ = oracle_trajectory(generate_world(3, "comparison", 1, 2))[0]
    torch.manual_seed(4)
    baseline = RecordProposer(8, 16)
    torch.manual_seed(4)
    model = RecordPlacementProposer(8, 16)
    baseline_outputs, baseline_state, baseline_lengths = baseline([observation], "internal")
    outputs, state, lengths = model([observation], "internal")
    assert lengths == baseline_lengths
    assert torch.equal(state, baseline_state)
    assert all(torch.equal(a, b) for a, b in zip(outputs, baseline_outputs))


def test_both_routes_active_but_only_internal_workspace_influences_carry():
    observation, action = oracle_trajectory(generate_world(7, "comparison", 1, 2))[0]
    for arm in ("internal", "typed_readout"):
        torch.manual_seed(12)
        model = RecordPlacementProposer(8, 16)
        outputs, state, lengths = model([observation], arm)
        carry_gradient = torch.autograd.grad(state.sum(), model.semantic_projection.weight,
                                             retain_graph=True, allow_unused=True)[0]
        loss = action_loss(outputs, [action])
        loss.backward()
        assert torch.isfinite(loss)
        assert model.semantic_projection.weight.grad.abs().sum() > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert (carry_gradient.abs().sum() > 0) if arm == "internal" else (carry_gradient.abs().sum() == 0)
        assert lengths == [len(observation["question"])]


def test_no_semantic_projection_makes_routes_identical():
    observation, _ = oracle_trajectory(generate_world(7, "comparison", 1, 2))[0]
    model = RecordPlacementProposer(8, 16)
    with torch.no_grad():
        model.semantic_projection.weight.zero_()
    internal = model([observation], "internal")
    readout = model([observation], "typed_readout")
    assert all(torch.equal(a, b) for a, b in zip(internal[0], readout[0]))
    assert torch.equal(internal[1], readout[1])
    assert internal[2] == readout[2]
