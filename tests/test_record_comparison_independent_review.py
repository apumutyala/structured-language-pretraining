"""Independent bounded checks: exact labels, observed feedback and rollout scoring."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from controlled_lm import record_learning_comparison as comparison


def review_world():
    return comparison.World(
        "review.0", (("f0", "k0", True), ("f1", "k7", False)),
        (("r0", "k0", True, "k1", True),)
        + tuple((f"r{i}", "k7", False, "k7", False) for i in range(1, 5)),
        ("k1", "k7"), "k1", False,
    )


class ScriptedHeads:
    """Proposal heads with no access to world, host, oracle or expected answer."""
    def __init__(self, choices):
        self.choices = choices
        self.seen = []

    def __call__(self, observations, arm, hx):
        self.seen.append((copy.deepcopy(observations[0]), arm, hx))
        choice = self.choices[len(self.seen) - 1]
        outputs = []
        for size, selected in zip((3, 5, 6, 6), choice):
            head = torch.full((1, size), -10.0)
            head[0, selected] = 10.0
            outputs.append(head)
        return tuple(outputs), torch.tensor([float(len(self.seen))]), [1]


@pytest.mark.parametrize("train_worlds,eval_worlds", [(24, 8), (256, 128)])
def test_labels_match_independent_fixed_point_not_private_actual_world(train_worlds, eval_worlds):
    dataset = comparison.make_dataset(2026102143, train_worlds, eval_worlds)
    signatures = []
    for worlds in dataset.values():
        for world in worlds:
            signatures.append(world.signature())
            claims = {(key, value) for _, key, value in world.facts}
            while True:
                extended = claims | {(q, qv) for _, p, pv, q, qv in world.rules if (p, pv) in claims}
                if extended == claims:
                    break
                claims = extended
            answers = [f"{key}={str(value).lower()}" for key, value in claims
                       if key == world.query_key and key in world.release_keys]
            assert len(answers) <= 1
            assert comparison.expected_answer(world) == (answers[0] if answers else None)
    assert len(signatures) == len(set(signatures))


def test_v2_removes_constant_source_id_and_fact_count_shortcuts():
    causal_ids = set()
    shortcut_misses = 0
    for worlds in comparison.make_dataset(2026102143, 24, 8).values():
        no_answer = 0
        for world in worlds:
            public = world.episode().structured_observation()
            assert len(public["state"]["records"]) == 2
            shortcut = (any(r["record_id"] == "fact.f0" for r in public["state"]["records"])
                        and world.query_key in public["state"]["release_keys"])
            expected = comparison.expected_answer(world)
            shortcut_misses += shortcut != (expected is not None)
            trajectory = comparison.oracle_trajectory(world)
            if expected is not None:
                causal_ids.add(trajectory[0][1]["premises"][0])
            no_answer += expected is None
        assert no_answer / len(worlds) == 0.375
    assert shortcut_misses > 0
    assert causal_ids == {"fact.f0", "fact.f1"}


def test_rollout_uses_own_rejections_and_carries_state_before_expected_label(monkeypatch):
    # First action cites an absent future record; second sees rejection then abstains.
    model = ScriptedHeads([(1, 0, 5, 0), (0, 0, 0, 0)])
    def forbidden_oracle(*args):
        raise AssertionError("oracle consulted in neural rollout")
    monkeypatch.setattr(comparison, "oracle_action", forbidden_oracle)
    def expected_after_rollout(world):
        assert len(model.seen) == 2
        return "k1=true"
    monkeypatch.setattr(comparison, "expected_answer", expected_after_rollout)
    row = comparison.rollout(model, review_world(), "external")
    assert model.seen[0][2] is None
    assert model.seen[1][2].item() == 1
    second = model.seen[1][0]
    assert second["feedback"]["accepted"] is False
    assert second["remaining_operations"] == 3
    assert [r["record_id"] for r in second["state"]["records"]] == ["fact.f0", "fact.f1"]
    assert row["history"][0]["feedback"]["reason"] == "missing or wrong number of premises"
    assert comparison.summarize([row])["task_exact"] == 0


def test_supported_permitted_wrong_query_release_is_not_task_success():
    # Kernel accepts a true permitted distractor; task metric must still reject it.
    model = ScriptedHeads([(2, 0, 0, 1)])
    row = comparison.rollout(model, review_world(), "internal")
    assert row["released"] == "k7=false"
    assert row["history"][0]["feedback"]["accepted"] is True
    summary = comparison.summarize([row])
    assert summary["task_exact"] == 0
    assert summary["supported_answer_recall"] == 0
    assert summary["query_wrong_release_count"] == 1


def test_oracle_derived_record_is_not_inserted_during_neural_rollout():
    # Four failed release attempts must not acquire an oracle derivation/fallback.
    model = ScriptedHeads([(2, 0, 0, 2)] * 4)
    row = comparison.rollout(model, review_world(), "internal")
    assert len(row["history"]) == 4
    assert all(step["feedback"]["accepted"] is False for step in row["history"])
    assert row["released"] is None and not row["explicit_abstain"]
    assert comparison.summarize([row])["task_exact"] == 0


def test_same_actions_produce_same_internal_and_external_kernel_feedback():
    choices = [(1, 0, 0, 0), (2, 0, 0, 2)]
    left = comparison.rollout(ScriptedHeads(choices), review_world(), "internal")
    right = comparison.rollout(ScriptedHeads(choices), review_world(), "external")
    assert left == right
    assert left["released"] == "k1=true"


def test_primary_paired_interval_excludes_negative_abstention_rewards():
    source = Path(__file__).resolve().parents[1] / "tools" / "train_record_comparison.py"
    spec = importlib.util.spec_from_file_location("_independent_record_trainer", source)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    left = [{"world_id": "positive", "expected": "k1=true", "released": "k1=true", "explicit_abstain": False},
            {"world_id": "negative", "expected": None, "released": None, "explicit_abstain": False}]
    right = [{"world_id": "positive", "expected": "k1=true", "released": None, "explicit_abstain": True},
             {"world_id": "negative", "expected": None, "released": None, "explicit_abstain": True}]
    estimate = trainer.bootstrap_paired(left, right, 7)
    assert estimate["primary_supported_answer_recall"]["worlds"] == 1
    assert estimate["primary_supported_answer_recall"]["internal_minus_external"] == 1
    assert estimate["primary_supported_answer_recall"]["world_bootstrap_95_interval"] == [1, 1]
    assert estimate["secondary_task_exact"]["internal_minus_external"] == 0
    corrupted = copy.deepcopy(right)
    corrupted[0]["expected"] = "k1=false"
    with pytest.raises(ValueError, match="expected labels differ"):
        trainer.bootstrap_paired(left, corrupted, 7)
    with pytest.raises(ValueError, match="unpaired"):
        trainer.bootstrap_paired(left, list(reversed(right)), 7)
