"""Fast CPU checks for the routing replication package (no corpus, no training).

Run: python -m pytest experiments/routing_replication/tests -q
"""
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

import analyze  # noqa: E402
import common  # noqa: E402
import eval_expanded  # noqa: E402
from controlled_lm.language_pilot_data import mixed_byte_batch, world_fold  # noqa: E402
from controlled_lm.normalized_typed_language_model import NormalizedTypedLanguageModel  # noqa: E402

BASE = common.ROOT / common.DEFAULT_BASE_CONFIG


def test_replicate_zero_reuses_base_bytes_and_others_vary_only_seeds():
    base = json.loads(BASE.read_text())
    assert common.replicate_config_bytes(BASE, 0) == BASE.read_bytes()
    seen = set()
    for replicate in range(1, 11):
        config = json.loads(common.replicate_config_bytes(BASE, replicate))
        common.check_replication_config(config, base)
        assert config["seed"] == base["seed"] + 100 * replicate
        assert config["data_seed"] == base["data_seed"] + 100 * replicate
        assert config["dev_seed"] == base["dev_seed"] and config["sample_seed"] == base["sample_seed"]
        seen.add((config["seed"], config["data_seed"]))
    assert len(seen) == 10


def test_config_check_rejects_other_changes():
    base = json.loads(BASE.read_text())
    for key, value in (("updates", 999), ("learning_rate", 1e-3), ("dev_seed", 1), ("hidden_size", 256)):
        changed = dict(base, **{key: value})
        with pytest.raises(ValueError):
            common.check_replication_config(changed, base)
    with pytest.raises(ValueError):
        common.check_replication_config(dict(base, seed=base["dev_seed"]), base)


def test_seeded_wrapper_restores_frozen_validation():
    import seeded_train
    wrapper = seeded_train.normalized_wrapper
    base = json.loads(BASE.read_text())
    view = dict(json.loads(common.replicate_config_bytes(BASE, 3)))
    view.update({key: base[key] for key in common.VARIABLE_KEYS})
    wrapper.validate_config(view, "cuda")  # original validator accepts the frozen view
    with pytest.raises(ValueError):
        wrapper.validate_config(view, "cpu")  # stage/device admission unchanged
    with pytest.raises(ValueError):
        seeded_train.smoke_validate_config(dict(base, pilot_stage="replication_cpu_smoke", updates=5,
                                                checkpoints=[0, 5]), "cuda")


def test_renderer_matches_generator():
    assert eval_expanded.verify_renderer(128, draws=512, seed=3)["identical"]


def test_question_set_is_dev_fold_and_large():
    dev, eval_ledgers, questions = eval_expanded.build_question_set([(0, 0, 0, 0)])
    assert all(10 <= world_fold(list(v)) < 20 for v in eval_ledgers)
    assert len(questions) == 4 * len(eval_ledgers) >= 500
    assert len(dev) == sum(10 <= world_fold(list(v)) < 20 for v in __import__("itertools").product(range(8), repeat=4))


def test_tiled_windows_cover_every_byte_once():
    text = "abc" * 200 + "é"
    windows = eval_expanded.tiled_windows(text, 128)
    raw = bytearray()
    for window in windows:
        assert window[0] == 1 and len(window) - 1 <= 128
        raw.extend(token - 3 for token in window[1:] if token >= 3)
    assert bytes(raw) == text.encode() and windows[-1][-1] == 2
    assert sum(token == 2 for window in windows for token in window) == 1


def test_world_rows_are_batch_independent():
    torch.manual_seed(0)
    model = NormalizedTypedLanguageModel(32, 64, num_layers=2, recurrent_kind="ssm", semantic_mode="internal",
                                         ssm_backend="parallel", projection_layout="shared",
                                         recurrent_residual=True).eval()
    rng = np.random.default_rng(5)
    mixed = mixed_byte_batch(["Once upon a time there was a cat."] * 3, 128, rng, random_windows=False,
                             world_split="dev")
    row = int(mixed.world_rows.nonzero().item())
    events = mixed.observed_event_ids[row][mixed.trusted_event_mask[row]].tolist()
    question = ([(e - 12) % 8 for e in events], int(mixed.claim_targets[row]) // 8)
    alone = eval_expanded.world_batch([question, ((1, 2, 3, 4), 2)], 128)
    with torch.no_grad():
        a = eval_expanded.forward(model, mixed)
        b = eval_expanded.forward(model, alone)
    assert torch.allclose(a.raw_logits[row], b.raw_logits[0], atol=1e-5)
    assert torch.allclose(a.grounded.probabilities[row], b.grounded.probabilities[0], atol=1e-9)


def test_statistics_against_reference_values():
    diffs = [0.3, -0.1, 0.4, 0.2, 0.25, -0.05, 0.15, 0.3, 0.1, 0.2]
    stats = analyze.paired_stats(diffs, [0.0] * 10, np.random.default_rng(0), 4000)
    mean, sd = np.mean(diffs), np.std(diffs, ddof=1)
    t = mean / (sd / math.sqrt(10))
    assert stats["t"] == pytest.approx(t)
    scipy_stats = pytest.importorskip("scipy.stats")
    assert stats["t_p_two_sided"] == pytest.approx(scipy_stats.ttest_1samp(diffs, 0).pvalue, rel=1e-8)
    assert stats["sign_test_p_two_sided"] == pytest.approx(scipy_stats.binomtest(8, 10).pvalue, rel=1e-12)
    half = scipy_stats.t.ppf(0.975, 9) * sd / math.sqrt(10)
    assert stats["t_ci95"][0] == pytest.approx(mean - half, rel=1e-6)
    assert stats["bootstrap_ci95"][0] < mean < stats["bootstrap_ci95"][1]


def test_holm_and_verdicts():
    assert analyze.holm([0.01, 0.04]) == [0.02, 0.04]
    assert analyze.holm([0.04, 0.01]) == [0.04, 0.02]
    good = {"mean_diff": 0.05, "t_ci95": [0.03, 0.07], "bootstrap_ci95": [0.03, 0.07]}
    assert analyze.verdict(good, "higher", 0.02, 0.001).startswith("SUPPORTED: improvement")
    assert analyze.verdict(good, "lower", 0.01, 0.001).startswith("REVERSED")
    null = {"mean_diff": 0.001, "t_ci95": [-0.004, 0.006], "bootstrap_ci95": [-0.003, 0.005]}
    assert analyze.verdict(null, "lower", 0.01, 0.6).startswith("NULL (equivalence)")
