import copy

import torch

from controlled_lm.record_learning_comparison import (
    FEATURE_SIZE, RECORD_IDS, RecordProposer, action_loss, expected_answer,
    generate_world, make_dataset, oracle_trajectory, rollout, typed_workspace,
)


def test_world_splits_and_oracle_real_kernel():
    dataset = make_dataset(17, 24, 12)
    signatures = [world.signature() for worlds in dataset.values() for world in worlds]
    assert len(signatures) == len(set(signatures))
    for split, worlds in dataset.items():
        for world in worlds:
            trajectory = oracle_trajectory(world)
            episode = world.episode()
            assert 1 <= len(trajectory) <= 4
            for observation, action in trajectory:
                assert len(typed_workspace(observation)) == FEATURE_SIZE
                assert episode.attempt(action)["feedback"]["accepted"]
            if world.world_id.endswith(".1"):
                assert len(trajectory) == (4 if split == "composition" else 3)
                assert expected_answer(world).startswith(world.query_key + "=")


def test_inputs_have_no_oracle_or_answer_and_external_has_full_snapshot():
    world = generate_world(4, "audit", 1, 2)
    episode = world.episode()
    observation = episode.structured_observation()
    import json
    assert json.loads(episode.external_observation()) == observation
    assert set(observation) == {"schema", "question", "state", "remaining_operations", "grammar", "feedback"}
    observation2 = copy.deepcopy(observation)
    observation2["state"]["rulebook"]["rules"][0]["conclusion"]["value"] ^= True
    assert typed_workspace(observation) != typed_workspace(observation2)
    for _, action in oracle_trajectory(world)[:-1]:
        before = typed_workspace(episode.structured_observation())
        episode.attempt(action)
        assert typed_workspace(episode.structured_observation()) != before


def test_matched_initialization_gradients_and_stateful_external():
    torch.set_num_threads(1)
    observation, action = oracle_trajectory(generate_world(7, "test", 1, 1))[0]
    torch.manual_seed(123)
    first = RecordProposer(8, 16)
    torch.manual_seed(123)
    second = RecordProposer(8, 16)
    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
        assert torch.equal(a, b), name
    for arm, model in (("internal", first), ("external", second)):
        outputs, state, _ = model([observation], arm)
        loss = action_loss(outputs, [action])
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        continued = model([observation], arm, state)[1]
        assert continued.shape == state.shape
        assert torch.isfinite(continued).all()
        # A contracting SSM may numerically forget its initial state over the
        # long external serialization. Carrying hx does not guarantee retention.
        if arm == "internal":
            assert not torch.equal(continued, state)
        gradient = model.semantic_projection.weight.grad
        assert (gradient.abs().sum() > 0) if arm == "internal" else (gradient.abs().sum() == 0)


def test_free_running_does_not_call_training_oracle_for_action(monkeypatch):
    world = generate_world(7, "test", 1, 1)
    model = RecordProposer(8, 16)
    # Force a bad release: raw pointer can name absent record, not an oracle mask.
    with torch.no_grad():
        for head in model.heads:
            head.weight.zero_()
            head.bias.zero_()
        model.heads[0].bias[2] = 10
        model.heads[3].bias[RECORD_IDS.index("step.s3")] = 10
    result = rollout(model, world, "external")
    assert len(result["history"]) == 4
    assert result["released"] is None
    assert all(not step["feedback"]["accepted"] for step in result["history"])
    assert all(step["action"]["record_id"] == "step.s3" for step in result["history"])
