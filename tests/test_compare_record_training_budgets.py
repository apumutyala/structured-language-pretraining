"""Test rejection of mismatched replay and accounting of paid-in-time prefixes."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools" / "compare_record_training_budgets.py"
SPEC = importlib.util.spec_from_file_location("_budget_comparison_review", SOURCE)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def fixture_run(path, updates, positive_correct):
    path.mkdir()
    config = {"seeds": [7], "updates": updates, "precision": "fp32", "data_seed": 11}
    env = {"python": "3.12", "torch": "2.14.1+cu130", "torch_cuda": "13.0",
           "device": "cuda", "device_name": "local GPU", "dtype": "torch.float32",
           "compute_capability": [12, 0], "nvidia_smi": "local GPU, 610.88, 12 GiB, variable",
           "source_sha256": {"model.py": "bound-model"}}
    write(path / "preregistered.json", {"config": config, "sources": env,
          "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()})
    worlds = {split: [{"world_id": "positive"}, {"world_id": "negative"}] for split in comparison.SPLITS}
    write(path / "world_manifest.json", worlds)
    positive = {"world_id": "positive", "expected": "k1=true", "released": "k1=true" if positive_correct else None,
                "explicit_abstain": not positive_correct, "history": [{"input_tokens": 11, "feedback": {"accepted": True}}]}
    negative = {"world_id": "negative", "expected": None, "released": None, "explicit_abstain": True,
                "history": [{"input_tokens": 11, "feedback": {"accepted": True}}]}
    rows = {split: [copy.deepcopy(positive), copy.deepcopy(negative)] for split in comparison.SPLITS}
    arms = {}
    for arm in comparison.ARMS:
        checkpoint = f"seed-7-{arm}.pt"
        (path / checkpoint).write_bytes(b"fixed checkpoint fixture")
        result = {"initial_state_sha256": "same-fresh-initialization", "checkpoint_file": checkpoint,
                  "checkpoint_sha256": comparison.digest(path / checkpoint),
                  "training": [{"update": i + 1, "loss": 1 / (i + 1), "gradient_l2": 2., "input_tokens": 11} for i in range(updates)],
                  "training_input_tokens": 11 * updates, "free_running": rows,
                  "metrics": {split: comparison.raw_metrics(value) for split, value in rows.items()},
                  "training_wall_seconds": updates * 2., "evaluation_wall_seconds": 1.,
                  "peak_cuda_allocated_bytes": 1000, "peak_cuda_reserved_bytes": 2000}
        arms[arm] = result
        write(path / f"seed-7-{arm}.json", result)
    write(path / "report.json", {"status": "enablement_passed", "seeds": [{"seed": 7, "arms": arms}],
          "dataset_manifest_sha256": comparison.digest(path / "world_manifest.json"), "total_wall_seconds": updates * 4. + 2.})


@pytest.fixture
def paired_runs(tmp_path):
    before, after = tmp_path / "small", tmp_path / "large"
    fixture_run(before, 2, False)
    fixture_run(after, 4, True)
    return before, after


def test_exact_prefix_and_full_replay_cost_are_preserved(paired_runs):
    result = comparison.compare_runs(*paired_runs)
    assert result["all_checks_pass"]
    assert result["cost"]["additional_after_run_wall_seconds_charged_in_full"] == 18
    assert result["cost"]["both_runs_total_wall_seconds"] == 28
    assert result["cost"]["after_run_replayed_prefix_updates"] == 4
    assert result["cost"]["all_charged_training_updates"] == 12
    group = result["descriptive_seed_groups"][0]
    assert group["unique_supported_worlds"] == group["supported_seed_world_trials"] == 1
    assert group["seed_mean_recall_change"] == 1
    assert str(paired_runs[0].parent) not in json.dumps(result)


def test_changed_prefix_is_reported_even_when_all_final_metrics_pass(paired_runs):
    before, after = paired_runs
    arm = comparison.read_json(after / "seed-7-internal.json")
    arm["training"][0]["loss"] += .01
    report = comparison.read_json(after / "report.json")
    report["seeds"][0]["arms"]["internal"] = arm
    write(after / "seed-7-internal.json", arm)
    write(after / "report.json", report)
    result = comparison.compare_runs(before, after)
    assert not result["all_checks_pass"]
    assert result["checks"]["seed_7_internal_iid_metrics"]
    assert not result["checks"]["seed_7_internal_exact_training_prefix"]


@pytest.mark.parametrize("changed", ["torch_cuda", "source_sha256", "nvidia_smi"])
def test_runtime_source_and_driver_mismatches_cannot_pass(paired_runs, changed):
    before, after = paired_runs
    prereg = comparison.read_json(after / "preregistered.json")
    prereg["sources"][changed] = {"model.py": "different"} if changed == "source_sha256" else "different"
    write(after / "preregistered.json", prereg)
    assert not comparison.compare_runs(before, after)["all_checks_pass"]
