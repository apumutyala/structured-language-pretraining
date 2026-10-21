"""Fixed, causal evidence recurrence for the finite event-token grammar.

This workspace records observed tokens, not truth or permission. The neural
capability stream may read its features; no neural tensor is a write input.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

KEYS, VALUES, VOCAB_SIZE = 4, 8, 112


def _fixed_weights():
    keep, write = [], []
    for token in range(VOCAB_SIZE):
        retained, inserted = [True] * 32, [False] * 32
        if 12 <= token < 44:
            key, value = divmod(token - 12, VALUES)
            retained[key * 8:(key + 1) * 8] = [False] * 8
            inserted[key * 8 + value] = True
        elif 44 <= token < 76:
            key, value = divmod(token - 44, VALUES)
            # Clear the coordinate before writing 1: no addition/saturation.
            retained[key * 8 + value] = False
            inserted[key * 8 + value] = True
        elif 76 <= token < 80:
            key = token - 76
            retained[key * 8:(key + 1) * 8] = [False] * 8
        keep.append(tuple(retained))
        write.append(tuple(inserted))
    return tuple(keep), tuple(write)


# Immutable Python values, deliberately neither nn.Parameters nor mutable
# checkpoint-loaded buffers. Materialized tensors are local to each operation.
KEEP_WEIGHTS, WRITE_WEIGHTS = _fixed_weights()


def _tokens(input_ids, ndim):
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != ndim:
        raise ValueError(f"input_ids must be a rank-{ndim} integer tensor")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must have int32 or int64 dtype; soft/learned tokens are forbidden")
    if bool(((input_ids < 0) | (input_ids >= VOCAB_SIZE)).any()):
        raise ValueError("input_ids outside the grammar vocabulary [0, 111]")


def _mask(attention_mask, input_ids):
    if attention_mask is None:
        return None
    if (not isinstance(attention_mask, torch.Tensor) or attention_mask.dtype != torch.bool
            or attention_mask.shape != input_ids.shape or attention_mask.device != input_ids.device):
        raise ValueError("attention_mask must be bool with input_ids shape and device")
    return attention_mask


def _bank(device):
    return (torch.tensor(KEEP_WEIGHTS, dtype=torch.bool, device=device).view(VOCAB_SIZE, KEYS, VALUES),
            torch.tensor(WRITE_WEIGHTS, dtype=torch.bool, device=device).view(VOCAB_SIZE, KEYS, VALUES))


def _transition(state, tokens, keep, write, valid=None):
    result = (state & keep[tokens.long()]) | write[tokens.long()]
    return result if valid is None else torch.where(valid[:, None, None], result, state)


class FixedSemanticTransitions(nn.Module):
    """Exact bitset recurrence, returning post-token float32 features [B,T,4,8].

    Forward always starts empty and accepts only discrete token history and an
    optional observed-position mask. Masked positions preserve the previous
    state. All in-vocabulary non-event tokens (including padding) are identity.
    """

    def forward(self, input_ids, attention_mask=None):
        _tokens(input_ids, 2)
        valid = _mask(attention_mask, input_ids)
        keep, write = _bank(input_ids.device)
        state = torch.zeros((input_ids.shape[0], KEYS, VALUES), dtype=torch.bool, device=input_ids.device)
        history = []
        for position in range(input_ids.shape[1]):
            state = _transition(state, input_ids[:, position], keep, write,
                                None if valid is None else valid[:, position])
            history.append(state)
        if not history:
            return torch.empty((input_ids.shape[0], 0, KEYS, VALUES), dtype=torch.float32,
                               device=input_ids.device)
        return torch.stack(history, dim=1).to(torch.float32)

    def step(self, state, token_ids, valid_mask=None):
        """One recurrent step for integration; exact bool state, no learned writes.

        The caller must use the empty initial state and previous returned state.
        This low-level primitive is not an authorization boundary against a
        Python caller deliberately fabricating state or replacing the code.
        """
        _tokens(token_ids, 1)
        valid = _mask(valid_mask, token_ids)
        if (not isinstance(state, torch.Tensor) or state.dtype != torch.bool
                or state.shape != (token_ids.shape[0], KEYS, VALUES)
                or state.device != token_ids.device):
            raise ValueError("state must be bool [batch,4,8] on the token device")
        return _transition(state, token_ids, *_bank(token_ids.device), valid)


class EvidenceStatus(IntEnum):
    UNKNOWN = 0
    SUPPORTED = 1
    CONFLICT = 2


@dataclass(frozen=True, slots=True)
class EvidenceQuery:
    """Application-typed request; no permission is inferred from these fields."""
    kind: str
    keys: tuple[int, ...]

    def __post_init__(self):
        if self.kind not in ("retrieve", "modularsum") or type(self.kind) is not str:
            raise ValueError("query kind must be retrieve or modularsum")
        if (type(self.keys) is not tuple or not self.keys
                or any(type(key) is not int or not 0 <= key < KEYS for key in self.keys)
                or len(set(self.keys)) != len(self.keys)):
            raise ValueError("query keys must be distinct exact integers in [0,3]")
        if (self.kind == "retrieve" and len(self.keys) != 1
                or self.kind == "modularsum" and not 2 <= len(self.keys) <= KEYS):
            raise ValueError("retrieve needs one key; modularsum needs two to four distinct keys")


@dataclass(frozen=True)
class EvidenceInterpretation:
    status: torch.Tensor
    value: torch.Tensor  # -1 unless supported; never a permission or raw-LM answer.


def interpret_evidence(state, query):
    """Separate exact typed interpretation; unknown takes precedence over conflict.

    Accepts arbitrary leading batch/time dimensions and exact binary features.
    Permission must be checked independently through the authenticated contract.
    """
    if type(query) is not EvidenceQuery:
        raise TypeError("query must be an EvidenceQuery")
    if (not isinstance(state, torch.Tensor) or state.ndim < 2 or state.shape[-2:] != (KEYS, VALUES)
            or state.requires_grad or state.dtype not in (torch.bool, torch.float32)
            or bool(((state != 0) & (state != 1)).any())):
        raise ValueError("state must be exact non-learned binary bool/float32 [...,4,8]")
    selected = state[..., list(query.keys), :].bool()
    count = selected.sum(-1)
    unknown, conflict = (count == 0).any(-1), (count > 1).any(-1)
    status = torch.where(unknown, int(EvidenceStatus.UNKNOWN),
                         torch.where(conflict, int(EvidenceStatus.CONFLICT), int(EvidenceStatus.SUPPORTED)))
    singleton_values = (selected.long() * torch.arange(VALUES, device=state.device)).sum(-1)
    value = singleton_values.sum(-1).remainder(VALUES)
    return EvidenceInterpretation(status, torch.where(status == int(EvidenceStatus.SUPPORTED), value, -1))


def prove_transition_induction():
    """Exhaust each 8-bit row/token lemma; row locality lifts it to all 2**32 states.

    Base is the all-zero workspace. The implementation is coordinatewise and
    has no cross-row reads: each row is exhaustively checked for every possible
    old bitset and vocabulary token against an independent integer-set oracle.
    This is a finite transition proof, not a general tensor-program verifier.
    """
    bits = torch.arange(VALUES)
    row_states = ((torch.arange(256)[:, None] >> bits) & 1).bool()
    # Every row gets every possible state; componentwise independence permits
    # simultaneous testing. No assumption that actual rows have equal values.
    states = row_states[:, None, :].expand(256, KEYS, VALUES)
    states = states.repeat(VOCAB_SIZE, 1, 1)
    tokens = torch.arange(VOCAB_SIZE).repeat_interleave(256)
    actual = FixedSemanticTransitions().step(states, tokens)
    expected = torch.empty_like(actual)
    for token in range(VOCAB_SIZE):
        for old in range(256):
            for key in range(KEYS):
                values = {value for value in range(VALUES) if old & (1 << value)}
                if 12 <= token <= 43 and (token - 12) // 8 == key:
                    values = {(token - 12) % 8}
                elif 44 <= token <= 75 and (token - 44) // 8 == key:
                    values.add((token - 44) % 8)
                elif token == 76 + key:
                    values.clear()
                expected[token * 256 + old, key] = torch.tensor([v in values for v in range(VALUES)])
    if not torch.equal(actual, expected):
        raise AssertionError("finite semantic transition lemma failed")
    # Check the fixed Boolean bank formula with 32 independent symbolic bits.
    # This verifies the formula, not the Python/PyTorch compiler or backend.
    import z3
    symbolic = z3.BitVec("observed_evidence_32bits", 32)
    for token in range(VOCAB_SIZE):
        keep_mask = sum(int(bit) << index for index, bit in enumerate(KEEP_WEIGHTS[token]))
        write_mask = sum(int(bit) << index for index, bit in enumerate(WRITE_WEIGHTS[token]))
        formula = (symbolic & keep_mask) | write_mask
        reference = symbolic
        if 12 <= token < 44:
            key, value = divmod(token - 12, VALUES)
            reference = (symbolic & ~(255 << (key * 8))) | (1 << (key * 8 + value))
        elif 44 <= token < 76:
            key, value = divmod(token - 44, VALUES)
            reference = symbolic | (1 << (key * 8 + value))
        elif 76 <= token < 80:
            reference = symbolic & ~(255 << ((token - 76) * 8))
        solver = z3.Solver()
        solver.add(formula != reference)
        if solver.check() != z3.unsat:
            raise AssertionError(f"symbolic transition formula failed for token {token}")
    weights = json.dumps([KEEP_WEIGHTS, WRITE_WEIGHTS], separators=(",", ":")).encode()
    return {"passed": True, "method": "exhaustive_per_key_tensor_checks_and_symbolic_32bit_bank_induction",
            "row_bitsets": 256, "keys": KEYS, "vocabulary_tokens": VOCAB_SIZE,
            "checked_row_transitions": 256 * KEYS * VOCAB_SIZE,
            "global_state_space_covered_by_product_lemma": 2 ** 32,
            "symbolic_bank_global_states": 2 ** 32, "symbolic_token_checks_unsat": VOCAB_SIZE,
            "solver": "Z3 " + z3.get_version_string(),
            "tensor_correspondence_assumption": "Inspected coordinatewise AND/OR tensor implementation matches the symbolic bank formula; finite tests do not verify all tensor-program executions.",
            "event_tokens": 68, "identity_tokens": VOCAB_SIZE - 68,
            "base": "empty evidence bitset", "induction": "exact post-token observed evidence for each finite prefix",
            "arithmetic": "CPU torch.bool AND/OR recurrence; float32 output casts exactly 0/1",
            "bank_sha256": hashlib.sha256(weights).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "exclusions": ["raw LM output correctness", "source truth", "authentication or authorization",
                           "arbitrary Python mutation", "GPU/backend implementation verification",
                           "machine-checked PyTorch-to-Boolean-formula correspondence",
                           "unavailable full8 or L0-L4 specifications"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-output", type=Path)
    args = parser.parse_args()
    result = prove_transition_induction()
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.proof_output:
        args.proof_output.parent.mkdir(parents=True, exist_ok=True)
        args.proof_output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
