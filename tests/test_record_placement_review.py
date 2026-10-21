"""Independent forward/gradient checks; no optimizer updates or GPU use."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from controlled_lm.record_learning_comparison import RecordProposer, generate_world
from controlled_lm.record_placement_diagnostic import RecordPlacementProposer


def observation():
    return generate_world(2026102161, "train", 1, 2).episode().structured_observation()


def test_subclass_has_exact_same_random_initialization_and_parameter_schema():
    torch.manual_seed(17)
    reference = RecordProposer(8, 16)
    torch.manual_seed(17)
    candidate = RecordPlacementProposer(8, 16)
    assert list(reference.state_dict()) == list(candidate.state_dict())
    for name, value in reference.state_dict().items():
        other = candidate.state_dict()[name]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other), name
        else:
            assert value == other, name


def test_zero_workspace_projection_removes_all_arm_differences():
    torch.manual_seed(19)
    model = RecordPlacementProposer(8, 16)
    with torch.no_grad():
        model.semantic_projection.weight.zero_()
    obs = observation()
    internal = model([obs], "internal")
    readout = model([obs], "typed_readout")
    assert internal[2] == readout[2] == [len(obs["question"].encode("ascii"))]
    assert torch.equal(internal[1], readout[1])
    for left, right in zip(internal[0], readout[0]):
        assert torch.equal(left, right)


def test_workspace_changes_core_state_only_for_recurrent_placement():
    torch.manual_seed(23)
    model = RecordPlacementProposer(8, 16)
    first = observation()
    second = copy.deepcopy(first)
    second["state"]["records"][0]["claim"]["value"] = not first["state"]["records"][0]["claim"]["value"]
    for arm, core_should_change in (("internal", True), ("typed_readout", False)):
        left, right = model([first], arm), model([second], arm)
        assert left[2] == right[2]
        assert (not torch.equal(left[1], right[1])) == core_should_change
        assert any(not torch.equal(a, b) for a, b in zip(left[0], right[0]))


def test_all_same_parameters_are_active_with_finite_gradients_in_both_placements():
    for arm in ("internal", "typed_readout"):
        torch.manual_seed(29)
        model = RecordPlacementProposer(8, 16)
        outputs, hx, lengths = model([observation()], arm)
        sum(head.square().sum() for head in outputs).backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, (arm, name)
            assert torch.isfinite(parameter.grad).all(), (arm, name)
            assert parameter.grad.abs().sum() > 0, (arm, name)
        assert lengths == [11]
        assert hx.shape == (1, 1, 8)


def test_fixed_final_recipe_rejects_checkpoint_count_seed_and_data_drift(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    spec = importlib.util.spec_from_file_location("_independent_placement_trainer", root / "tools/train_record_placement_diagnostic.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    config = json.loads((root / "configs/record_placement_fit_20261021.json").read_text())
    trainer.validate_placement(config)
    for field in ("updates", "seed", "data_seed", "train_worlds", "hidden_size", "native_size"):
        changed = copy.deepcopy(config)
        changed[field] += 1
        with pytest.raises(ValueError):
            trainer.validate_placement(changed)
