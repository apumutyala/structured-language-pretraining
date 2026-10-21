"""Same-information interface plumbing, not a trained external-workspace baseline.

Both interfaces expose identical current facts, ground rules, intermediate
records, permissions, operation budget and failure feedback after every step.
The host owns the kernel and evidence updates. This is not process isolation.
"""
from __future__ import annotations

from dataclasses import asdict
import json

from .supported_state import Derivation, ProposerChannel

GRAMMAR = {
    "derive": ["op", "record_id", "rule_id", "premises"],
    "release": ["op", "record_id", "evidence_epoch", "policy_epoch"],
    "abstain": ["op"],
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


class ComparisonEpisode:
    def __init__(self, proposer: ProposerChannel, question: str, operation_budget: int):
        if type(proposer) is not ProposerChannel:
            raise ValueError("host-supplied proposer channel required")
        if type(question) is not str or len(question) > 2048:
            raise ValueError("bounded question string required")
        if type(operation_budget) is not int or not 1 <= operation_budget <= 128:
            raise ValueError("operation budget must be an integer in 1..128")
        self._proposer = proposer
        self.question = question
        self.remaining = operation_budget
        self.feedback = {"accepted": None, "reason": "initial observation", "released_text": None}
        self.attempts = 0

    def structured_observation(self):
        state = asdict(self._proposer.snapshot())
        # Session nonces are administrative binding, not a semantic input.
        state.pop("kernel_id")
        state.pop("certificates")
        return json.loads(canonical({
            "schema": "supported-state-comparison-v1", "question": self.question,
            "state": state, "remaining_operations": self.remaining,
            "grammar": GRAMMAR, "feedback": self.feedback,
        }))

    def external_observation(self):
        return canonical(self.structured_observation())

    def _reject(self, reason):
        self.feedback = {"accepted": False, "reason": reason, "released_text": None}
        return self.structured_observation()

    def attempt(self, proposal):
        """Every attempted operation, including malformed input, consumes budget.

        Returned feedback is available immediately to BOTH interface arms.
        An exhausted episode performs no further semantic operation.
        """
        if self.remaining == 0:
            return self._reject("operation budget exhausted")
        self.remaining -= 1
        self.attempts += 1
        if type(proposal) is str:
            if len(proposal) > 8192:
                return self._reject("proposal too long")
            try:
                proposal = json.loads(proposal, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
            except (ValueError, RecursionError):
                return self._reject("invalid JSON proposal")
        if type(proposal) is not dict or type(proposal.get("op")) is not str:
            return self._reject("exact operation object required")
        op = proposal["op"]
        if op not in GRAMMAR or set(proposal) != set(GRAMMAR[op]):
            return self._reject("operation outside declared grammar")
        if op == "abstain":
            self.feedback = {"accepted": True, "reason": "abstained", "released_text": None}
        elif op == "derive":
            premises = proposal["premises"]
            if type(premises) is not list or len(premises) > 16:
                return self._reject("bounded premise list required")
            result = self._proposer.step(Derivation(proposal["record_id"], proposal["rule_id"], tuple(premises)))
            self.feedback = {"accepted": result.accepted, "reason": result.reason, "released_text": None}
        else:
            state = self._proposer.snapshot()
            if (type(proposal["record_id"]) is not str
                    or type(proposal["evidence_epoch"]) is not int
                    or type(proposal["policy_epoch"]) is not int):
                return self._reject("malformed release request")
            if (proposal["evidence_epoch"] != state.evidence_epoch
                    or proposal["policy_epoch"] != state.policy_epoch):
                return self._reject("stale release request")
            certificate = next((c for c in state.certificates if c.record.record_id == proposal["record_id"]), None)
            if certificate is None:
                return self._reject("unknown supported record")
            result = self._proposer.release(certificate)
            self.feedback = {"accepted": result.accepted, "reason": result.reason, "released_text": result.text}
        return self.structured_observation()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON member")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("nonstandard JSON numeric constant")
