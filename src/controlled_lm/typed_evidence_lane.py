"""Separate observed evidence events from arbitrary language token namespaces.

Language token VALUES are never evidence: only shape/device/time are used.
The application authenticates event provenance and supplies the explicit trust
mask. A Boolean mask does not authenticate itself. Generated text must remain
in the language lane or carry trust=False; this research API cannot identify
a caller deliberately relabeling generated events as trusted observations.

The output records observed event-token history, not external truth, query
relevance, authorization or unconstrained learned-answer correctness. The
existing fixed Boolean bank is reused without trainable/checkpoint state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from .parallel_semantic import ParallelSemanticTransitions
from .semantic_transitions import KEEP_WEIGHTS, WRITE_WEIGHTS


def _aligned(value, language, name, *, boolean=False):
    if (not isinstance(value, Tensor) or value.shape != language.shape
            or value.device != language.device):
        raise ValueError(f"{name} must have language shape [batch,time] and device")
    accepted = (torch.bool,) if boolean else (torch.int32, torch.int64)
    if value.dtype not in accepted:
        raise ValueError(f"{name} must have {'bool' if boolean else 'int32 or int64'} dtype")


class TypedEvidenceLane(nn.Module):
    """Return exact post-slot evidence features [B,T,4,8], starting empty.

    `observed_event_ids` and `trusted_event_mask` must appear together. Only
    trusted event IDs are range-checked by the fixed bank. Untrusted values,
    including negative/huge fake events, map to identity ID0 before validation
    or indexing. Trusted invalid IDs reject even at padded positions; valid
    mask independently prevents an otherwise valid trusted event from writing.
    No neural proposal, external state, or generated-feedback API exists.
    """

    def __init__(self):
        super().__init__()
        self.transitions = ParallelSemanticTransitions()

    def forward(self, language_token_ids: Tensor, observed_event_ids: Tensor | None = None,
                trusted_event_mask: Tensor | None = None,
                valid_mask: Tensor | None = None) -> Tensor:
        if (not isinstance(language_token_ids, Tensor) or language_token_ids.ndim != 2
                or language_token_ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("language_token_ids must be int32 or int64 [batch,time]")
        if valid_mask is not None:
            _aligned(valid_mask, language_token_ids, "valid_mask", boolean=True)
        if (observed_event_ids is None) != (trusted_event_mask is None):
            raise ValueError("observed_event_ids require an explicit trusted_event_mask and vice versa")
        if observed_event_ids is None:
            return torch.zeros((*language_token_ids.shape, 4, 8), dtype=torch.float32,
                               device=language_token_ids.device)
        _aligned(observed_event_ids, language_token_ids, "observed_event_ids")
        _aligned(trusted_event_mask, language_token_ids, "trusted_event_mask", boolean=True)
        safe_events = torch.where(trusted_event_mask, observed_event_ids,
                                  torch.zeros_like(observed_event_ids))
        return self.transitions(safe_events, valid_mask)


def prove_typed_lane() -> dict:
    """Check scalar-lane Boolean/bitvector lemmas, lifted by prefix induction.

    K/W are arbitrary 32-bit transition banks except identity at ID0. This
    proves sanitization/masking/noninterference, not the event bank's meaning.
    Source-to-formula and PyTorch/backend refinement remain assumptions.
    """
    import z3
    if not all(KEEP_WEIGHTS[0]) or any(WRITE_WEIGHTS[0]):
        raise AssertionError("fixed bank token0 must be identity")
    state = z3.BitVec("evidence", 32)
    event_a, event_b, text_a, text_b = z3.Ints("event_a event_b text_a text_b")
    trust, valid = z3.Bools("trusted valid")
    keep = z3.Function("keep", z3.IntSort(), z3.BitVecSort(32))
    write = z3.Function("write", z3.IntSort(), z3.BitVecSort(32))
    bank_identity = [keep(0) == z3.BitVecVal(2 ** 32 - 1, 32), write(0) == 0]

    def step(language_id, event_id, trusted, active):
        # language_id deliberately has no read path into the protected step.
        safe = z3.If(trusted, event_id, 0)
        return z3.If(active, (state & keep(safe)) | write(safe), state)

    checks = [
        ("language_values_noninterference", [],
         step(text_a, event_a, trust, valid) != step(text_b, event_a, trust, valid)),
        ("untrusted_event_values_noninterference", [z3.Not(trust)],
         step(text_a, event_a, trust, valid) != step(text_a, event_b, trust, valid)),
        ("untrusted_slot_stutters", [z3.Not(trust)],
         step(text_a, event_a, trust, valid) != state),
        ("invalid_slot_stutters", [z3.Not(valid)],
         step(text_a, event_a, trust, valid) != state),
    ]
    results = []
    for name, preconditions, violation in checks:
        solver = z3.Solver()
        solver.add(*bank_identity, *preconditions, violation)
        status = solver.check()
        if status != z3.unsat:
            raise AssertionError(f"typed lane {name}: expected unsat, got {status}")
        results.append({"name": name, "counterexample_query": str(status)})
    return {"passed": True, "solver": "Z3 " + z3.get_version_string(), "unsat_checks": len(results),
            "checks": results, "arithmetic": "exact Boolean and 32-bit bitvector operations",
            "base": "all-zero workspace",
            "induction": "Same event/trust/valid lane gives identical full traces despite arbitrary language ID changes. Inserted untrusted or invalid slots retain state; aligned original positions retain their original states.",
            "trust_assumption": "Application authenticates observed-event provenance and trust mask; this API cannot authenticate a self-asserted Boolean.",
            "trusted_domain": "trusted event IDs in [0,111], including at padded positions; language IDs and untrusted event values impose no evidence-domain bounds",
            "bank_identity_verified": "KEEP[0] all true and WRITE[0] all false; other bank semantics reuse existing semantic-transition proof",
            "bank_sha256": hashlib.sha256(json.dumps([KEEP_WEIGHTS, WRITE_WEIGHTS], separators=(",", ":")).encode()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "parallel_source_sha256": hashlib.sha256(Path(__file__).with_name("parallel_semantic.py").read_bytes()).hexdigest(),
            "implementation_correspondence_assumption": "Inspected tensor sanitization/identity-bank composition matches symbolic step; hashes bind reviewed sources but do not prove tensor/compiler/backend refinement.",
            "exclusions": ["authentication supplied by the mask itself", "external truth or semantic safety",
                           "raw language correctness", "arbitrary caller/module mutation",
                           "machine-checked PyTorch/CUDA refinement", "floating-point learned dynamics",
                           "full8 or L0-L4 compliance"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-output", type=Path)
    args = parser.parse_args()
    receipt = prove_typed_lane()
    text = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    if args.proof_output:
        args.proof_output.parent.mkdir(parents=True, exist_ok=True)
        args.proof_output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
