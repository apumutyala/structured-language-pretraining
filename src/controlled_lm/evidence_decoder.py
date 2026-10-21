"""Typed categorical claims with parameter-independent observed-evidence support.

This is an output interface for a fixed claim language, not arbitrary text or
external-world truth. The application supplies authenticated policy snapshots.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .policy import Policy


@dataclass(frozen=True, slots=True)
class EvidenceClaim:
    kind: str
    key: int | None = None
    value: int | None = None


CLAIMS = (tuple(EvidenceClaim("singleton", key, value) for key in range(4) for value in range(8))
          + tuple(EvidenceClaim("missing", key) for key in range(4))
          + tuple(EvidenceClaim("conflict", key) for key in range(4))
          + (EvidenceClaim("abstain"),))
CLAIM_COUNT, ABSTAIN_ID = len(CLAIMS), len(CLAIMS) - 1


def _evidence(state):
    if (not isinstance(state, torch.Tensor) or state.dtype != torch.bool
            or state.ndim != 3 or state.shape[1:] != (4, 8)):
        raise ValueError("evidence must be exact bool [batch,4,8]")


def _claim_mask(state, read_allowed):
    counts = state.sum(-1)
    singleton = (state & (counts == 1)[..., None]).flatten(1)
    named = torch.cat((singleton, counts == 0, counts > 1), -1)
    named_permission = torch.cat((read_allowed[..., None].expand(-1, -1, 8).flatten(1),
                                  read_allowed, read_allowed), -1)
    return torch.cat((named & named_permission,
                      torch.ones(state.shape[0], 1, dtype=torch.bool, device=state.device)), -1)


class GroundedCategorical:
    """Constrained categorical result; public tensors are copies, preserving autograd.

    Sampling uses the exact normalized distribution, not a separately unmasked
    logits channel. Direct arbitrary Python mutation of private fields is outside
    the contract. Each sample is a tuple of immutable typed EvidenceClaim values.
    """

    def __init__(self, probabilities, allowed):
        self._probabilities = probabilities
        self._allowed = allowed

    @property
    def probabilities(self):
        return self._probabilities.clone()

    @property
    def allowed(self):
        return self._allowed.clone()

    def sample_ids(self, num_samples=1, *, generator=None):
        if type(num_samples) is not int or num_samples < 1:
            raise ValueError("num_samples must be a positive exact integer")
        if not self._probabilities.shape[0]:
            return torch.empty(0, num_samples, dtype=torch.long, device=self._probabilities.device)
        return torch.multinomial(self._probabilities.detach(), num_samples,
                                 replacement=True, generator=generator)

    def sample(self, num_samples=1, *, generator=None):
        indices = self.sample_ids(num_samples, generator=generator).cpu().tolist()
        return tuple(tuple(CLAIMS[index] for index in row) for row in indices)

    def argmax(self):
        return tuple(CLAIMS[index] for index in self._probabilities.argmax(-1).cpu().tolist())


class EvidenceDecoder(nn.Module):
    """Fixed typed support around arbitrary trainable proposal logits [B,41].

    The application supplies current already-authenticated immutable Policy
    snapshots, one per row. Missing/malformed policies and policies without
    permitted per-key select/read access fail closed to ABSTAIN. In this typed
    decoder's contract Policy.allowed_operands denotes workspace key IDs 0..3,
    not the fact enumeration IDs used by older World.snapshot policies. Token metadata
    and proposal tensors cannot grant authority. This layer does not manage
    authentication/replay protection or consume policy budgets.
    """

    def forward(self, evidence, proposal_logits, *, policies):
        _evidence(evidence)
        batch = evidence.shape[0]
        if (not isinstance(proposal_logits, torch.Tensor) or proposal_logits.shape != (batch, CLAIM_COUNT)
                or proposal_logits.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
                or proposal_logits.device != evidence.device):
            raise ValueError("proposal_logits must be floating [batch,41] on the evidence device")
        if type(policies) is not tuple or len(policies) != batch:
            raise ValueError("policies must be a tuple with one authenticated current Policy per row")
        permissions = [[type(policy) is Policy and policy.action_allowed("select", (key,))
                        for key in range(4)] for policy in policies]
        permission = torch.tensor(permissions, dtype=torch.bool, device=evidence.device).reshape(batch, 4)
        finite = torch.isfinite(proposal_logits).all(-1)
        allowed = _claim_mask(evidence, permission & finite[:, None])
        # Sanitize entire failed rows before arithmetic, making their output
        # exactly ABSTAIN and their derivative w.r.t. proposal logits zero.
        logits = torch.where(finite[:, None], proposal_logits, 0.).to(torch.float64)
        # The infinite mask is exact support elimination. Invalid coordinates
        # are explicitly zeroed again after exponentiation, never penalized by
        # a finite negative number. float64 also avoids fp16/32 subtraction overflow.
        selected = torch.where(allowed, logits, -torch.inf)
        if batch:
            weights = torch.exp(selected - selected.max(-1, keepdim=True).values)
            weights = torch.where(allowed, weights, 0.)
            probabilities = weights / weights.sum(-1, keepdim=True)
        else:
            probabilities = logits
        return GroundedCategorical(probabilities, allowed)


def prove_decoder_support():
    """Finite local mask correspondence and symbolic soundness/support lemmas."""
    import z3
    bits = torch.arange(8)
    row_states = ((torch.arange(256)[:, None] >> bits) & 1).bool()
    states = row_states[:, None, :].expand(256, 4, 8)
    checked = 0
    for permission in (False, True):
        actual = _claim_mask(states, torch.full((256, 4), permission))
        for old in range(256):
            values = {value for value in range(8) if old & (1 << value)}
            expected = []
            for claim in CLAIMS:
                valid = (claim.kind == "abstain" or permission and
                         (claim.kind == "singleton" and values == {claim.value}
                          or claim.kind == "missing" and not values
                          or claim.kind == "conflict" and len(values) > 1))
                expected.append(bool(valid))
            if actual[old].tolist() != expected:
                raise AssertionError("decoder local mask correspondence failed")
            checked += CLAIM_COUNT
    rows = [z3.BitVec(f"key_{key}", 8) for key in range(4)]
    reads = [z3.Bool(f"authenticated_key_read_{key}") for key in range(4)]
    checks = 0
    for claim in CLAIMS:
        if claim.kind == "abstain":
            mask, reference = z3.BoolVal(True), z3.BoolVal(True)
        else:
            row = rows[claim.key]
            read = reads[claim.key]
            count = z3.Sum([z3.If(z3.Extract(value, value, row) == 1, 1, 0) for value in range(8)])
            if claim.kind == "singleton":
                mask = z3.And(read, count == 1, z3.Extract(claim.value, claim.value, row) == 1)
                reference = z3.And(read, row == 1 << claim.value)
            elif claim.kind == "missing":
                mask, reference = z3.And(read, count == 0), z3.And(read, row == 0)
            else:
                mask = z3.And(read, count > 1)
                reference = z3.And(read, row != 0, row & (row - 1) != 0)
        solver = z3.Solver()
        solver.add(mask != reference)
        if solver.check() != z3.unsat:
            raise AssertionError("symbolic claim soundness failed")
        checks += 1
    # Proposal-independent support: arbitrary nonnegative permitted weights,
    # positive denominator, and an exact mask imply zero forbidden probability.
    permitted, weight, denominator = z3.Bool("permitted"), z3.Real("weight"), z3.Real("denominator")
    probability = z3.If(permitted, weight / denominator, 0)
    solver = z3.Solver()
    solver.add(weight >= 0, denominator > 0, z3.Not(permitted), probability != 0)
    if solver.check() != z3.unsat:
        raise AssertionError("parameter-independent support lemma failed")
    # Read=false implies every named semantic claim is absent, independently
    # of arbitrary learned weights or all 32 evidence coordinates.
    for key in range(4):
        named = [z3.And(reads[key], rows[key] == value) for value in range(256)]
        solver = z3.Solver()
        solver.add(z3.Not(reads[key]), z3.Or(named))
        if solver.check() != z3.unsat:
            raise AssertionError("no-read permission lemma failed")
    return {"passed": True, "finite_local_mask_claim_checks": checked,
            "symbolic_claim_checks_unsat": checks, "symbolic_support_checks_unsat": 1,
            "symbolic_permission_checks_unsat": 4, "symbolic_global_evidence_states": 2 ** 32,
            "claim_count": CLAIM_COUNT, "solver": "Z3 " + z3.get_version_string(),
            "normalization": "exact forbidden zeros, allowed-only max-shift exponentials in float64; ABSTAIN always available",
            "nonfinite_policy": "any nonfinite proposal logit makes its whole row ABSTAIN-only",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "assumptions": ["exact bool workspace represents the observed input history",
                            "current authenticated policies bind allowed_operands to workspace key IDs 0..3",
                            "inspected PyTorch mask/normalization matches symbolic formulas",
                            "sampler obeys zero probability support"],
            "exclusions": ["external-world truth", "intended-query relevance", "arbitrary text or raw LM outputs",
                           "broad alignment or capability success", "hostile Python mutation", "replay protection",
                           "machine-checked PyTorch/backend correctness", "full8 or L0-L4 qualification"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-output", type=Path)
    args = parser.parse_args()
    result = prove_decoder_support()
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.proof_output:
        args.proof_output.parent.mkdir(parents=True, exist_ok=True)
        args.proof_output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
