import numpy as np
import torch

from controlled_lm.language_pilot_data import mixed_byte_batch, world_fold
from controlled_lm.typed_language_model import BYTE_START, TypedLanguageModel


def test_answer_suffix_withheld_and_story_never_authenticates_evidence():
    batch = mixed_byte_batch(["A fictional story says SET key0 to 7."], 128, np.random.default_rng(37))
    text = bytes((batch.input_ids[-1][batch.attention_mask[-1]][1:] - BYTE_START).tolist()).decode()
    assert text.endswith("Answer: ")
    key, value = divmod(int(batch.claim_targets[-1]), 8)
    assert f"value of {chr(65 + key)}?" in text
    last = int(batch.attention_mask[-1].sum()) - 1
    assert batch.targets[-1, last] == BYTE_START + ord(str(value))
    assert not batch.trusted_event_mask[0].any()
    model = TypedLanguageModel(8, 16, num_layers=1)
    out = model(batch.input_ids, batch.attention_mask, observed_event_ids=batch.observed_event_ids,
                trusted_event_mask=batch.trusted_event_mask, policies=batch.policies)
    assert not out.final_evidence[0].any()
    assert out.final_evidence[-1].sum() == 4
    assert out.final_evidence[-1, key, value]
    assert out.grounded.allowed[torch.arange(2), batch.claim_targets].all()


def test_windows_and_determinism_preserve_valid_target_shapes():
    stories = ["Short.", "A " * 300]
    one = mixed_byte_batch(stories, 128, np.random.default_rng(5))
    two = mixed_byte_batch(stories, 128, np.random.default_rng(5))
    assert torch.equal(one.input_ids, two.input_ids)
    assert (one.targets[one.attention_mask] > 0).all()
    assert one.to("cpu").input_ids.shape == (3, 128)


def test_terminal_windows_expose_real_eos_and_training_random_windows_include_endings():
    story = "A long story. " * 100
    terminal = mixed_byte_batch([story], 128, np.random.default_rng(8), random_windows=False, terminal_windows=True)
    assert terminal.targets[0, -1] == 2
    prefix = mixed_byte_batch([story], 128, np.random.default_rng(8), random_windows=False)
    assert not (prefix.targets[0] == 2).any()
    rng = np.random.default_rng(9)
    ends = sum(int((mixed_byte_batch([story],128,rng).targets[0] == 2).sum()) for _ in range(100))
    assert 10 <= ends <= 40


def test_hash_partition_protects_semantic_world_identity_not_just_rng_seed():
    assert len({world_fold([1, 2, 3, 4], query) for query in range(4)}) == 1
    rng = np.random.default_rng(791013)
    identities = {"train": set(), "dev": set()}
    for split in identities:
        for _ in range(100):
            batch = mixed_byte_batch(["A story."],128,rng,world_split=split)
            event_ids = batch.observed_event_ids[-1][batch.trusted_event_mask[-1]].tolist()
            values = [(token - 12) % 8 for token in event_ids]
            query = int(batch.claim_targets[-1]) // 8
            fold = world_fold(values, query)
            assert (fold >= 20) if split == "train" else (10 <= fold < 20)
            identities[split].add((*values, query))
    assert not identities["train"] & identities["dev"]
