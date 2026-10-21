"""Initial byte-LM examples and separate trusted finite-world observations."""
from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from .policy import Policy, READ
from .typed_language_model import BOS_ID, EOS_ID, BYTE_START


@dataclass
class LanguageBatch:
    input_ids: torch.Tensor
    targets: torch.Tensor
    attention_mask: torch.Tensor
    observed_event_ids: torch.Tensor
    trusted_event_mask: torch.Tensor
    claim_targets: torch.Tensor
    world_rows: torch.Tensor
    policies: tuple

    def to(self, device):
        return LanguageBatch(*(getattr(self, name).to(device) if name != "policies" else self.policies
                               for name in self.__dataclass_fields__))


def world_fold(values, query=None):
    # All questions about an identical ledger remain in the same fold.
    # query is accepted for caller compatibility but cannot affect membership.
    identity = ",".join(map(str, values))
    return int(hashlib.sha256(("controlled-language-world-v1|" + identity).encode()).hexdigest(), 16) % 100


def mixed_byte_batch(stories, length, rng, *, random_windows=True,
                     terminal_windows=False, world_split=None):
    """One world row plus story rows; world answer NEVER appears in its inputs.

    Stories are fictional language practice, not evidence. The deterministic
    world renderer owns its independent observed-event provenance annotations.
    """
    if length < 72 or not stories:
        raise ValueError("length>=72 and at least one story row required")
    if world_split not in (None, "train", "dev"):
        raise ValueError("world_split must be train/dev or legacy None; reserved TEST is not opened")
    batch = len(stories) + 1
    ids = torch.zeros(batch, length, dtype=torch.long)
    targets, events = torch.zeros_like(ids), torch.zeros_like(ids)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    trusted = torch.zeros_like(valid)
    claim_targets = torch.full((batch,), 40, dtype=torch.long)
    world_rows = torch.zeros(batch, dtype=torch.bool)
    policies = []
    for row, story in enumerate(stories):
        raw = story.encode("utf-8")
        if not raw:
            raise ValueError("empty story")
        terminal = terminal_windows or (random_windows and rng.random() < .25)
        offset = (max(0, len(raw) - (length - 1)) if terminal else
                  int(rng.integers(max(1, len(raw) - length + 1))) if random_windows else 0)
        segment = raw[offset:offset + length]
        ending = [EOS_ID] if offset + len(segment) == len(raw) else []
        tokens = [BOS_ID, *(BYTE_START + byte for byte in segment), *ending]
        count = min(length, len(tokens) - 1)
        ids[row, :count], targets[row, :count] = torch.tensor(tokens[:count]), torch.tensor(tokens[1:count + 1])
        valid[row, :count] = True
        policies.append(Policy())  # Fiction cannot authorize an evidence claim.
    row = batch - 1
    for _ in range(1000):
        values = rng.integers(8, size=4).tolist()
        query = int(rng.integers(4))
        fold = world_fold(values, query)
        if (world_split is None or world_split == "train" and fold >= 20
                or world_split == "dev" and 10 <= fold < 20):
            break
    else:
        raise RuntimeError("world split rejection budget exhausted")
    prefix = bytearray(b"Facts: ")
    positions = []
    for key, value in enumerate(values):
        prefix.extend(f"{chr(65 + key)}={value}; ".encode())
        # +1 accounts for BOS. Index is the last observed byte of this fact.
        positions.append(len(prefix))
    prefix.extend(f"\nQuestion: value of {chr(65 + query)}?\nAnswer: ".encode())
    context = [BOS_ID, *(BYTE_START + byte for byte in prefix)]
    if len(context) > length:
        raise ValueError("world prefix exceeds registered length")
    answer = BYTE_START + ord(str(values[query]))
    count = len(context)
    ids[row, :count] = torch.tensor(context)
    targets[row, :count] = torch.tensor([*context[1:], answer])
    valid[row, :count] = True
    for key, position in enumerate(positions):
        events[row, position] = 12 + key * 8 + values[key]
        trusted[row, position] = True
    claim_targets[row] = query * 8 + values[query]
    world_rows[row] = True
    policies.append(Policy(caps=READ, budget=8, witness_valid=True, allowed_operands=(0, 1, 2, 3)))
    return LanguageBatch(ids, targets, valid, events, trusted, claim_targets, world_rows, tuple(policies))
