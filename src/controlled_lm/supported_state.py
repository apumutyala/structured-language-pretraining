"""Finite, exact supported RECORD state; no embeddings, training, or Lean proof.

The trusted host owns facts, the ground rulebook and release policy. A proposer
can only name a declared rule and existing premise records. Separate channels
express application ownership, not isolation from hostile code in this process.
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

MAX_KEYS = 16
MAX_RULES = 64
MAX_RECORDS = 128
MAX_EVIDENCE = 64


def _name(value, limit=64):
    if (type(value) is not str or not 1 <= len(value) <= limit
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in value)):
        raise ValueError("bounded ASCII identifier required")


@dataclass(frozen=True, slots=True)
class Claim:
    key: str
    value: bool


def _claim(claim):
    if type(claim) is not Claim or type(claim.value) is not bool:
        raise ValueError("exact Boolean Claim required")
    _name(claim.key)


@dataclass(frozen=True, slots=True)
class Rule:
    rule_id: str
    premises: tuple[Claim, ...]
    conclusion: Claim


@dataclass(frozen=True, slots=True)
class Rulebook:
    keys: tuple[str, ...]
    rules: tuple[Rule, ...]

    def __post_init__(self):
        if type(self.keys) is not tuple or not 1 <= len(self.keys) <= MAX_KEYS:
            raise ValueError("finite immutable key universe required")
        for key in self.keys:
            _name(key)
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("duplicate key")
        if type(self.rules) is not tuple or len(self.rules) > MAX_RULES:
            raise ValueError("bounded immutable rulebook required")
        ids = set()
        for rule in self.rules:
            if type(rule) is not Rule:
                raise ValueError("exact ground Rule required")
            _name(rule.rule_id)
            if rule.rule_id in ids:
                raise ValueError("duplicate rule id")
            ids.add(rule.rule_id)
            if type(rule.premises) is not tuple or not 1 <= len(rule.premises) <= MAX_KEYS:
                raise ValueError("nonempty bounded immutable premises required")
            for claim in rule.premises + (rule.conclusion,):
                _claim(claim)
                if claim.key not in self.keys:
                    raise ValueError("claim outside declared universe")
            if len(set(rule.premises)) != len(rule.premises):
                raise ValueError("duplicate rule premise")
            by_key = {claim.key: claim.value for claim in rule.premises}
            if len(by_key) != len(rule.premises):
                raise ValueError("contradictory rule premises")


@dataclass(frozen=True, slots=True)
class Record:
    record_id: str
    claim: Claim
    source: str  # evidence or rule
    source_id: str
    premises: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Certificate:
    kernel_id: str
    evidence_epoch: int
    policy_epoch: int
    record: Record


@dataclass(frozen=True, slots=True)
class Derivation:
    record_id: str
    rule_id: str
    premises: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Snapshot:
    kernel_id: str
    evidence_epoch: int
    policy_epoch: int
    rulebook: Rulebook
    release_keys: tuple[str, ...]
    records: tuple[Record, ...]
    certificates: tuple[Certificate, ...]


@dataclass(frozen=True, slots=True)
class StepResult:
    accepted: bool
    reason: str
    certificate: Certificate | None
    snapshot: Snapshot


@dataclass(frozen=True, slots=True)
class Release:
    accepted: bool
    reason: str
    text: str | None = None
    claim: Claim | None = None


class _Core:
    def __init__(self, rulebook, release_keys):
        if type(rulebook) is not Rulebook:
            raise ValueError("host-owned exact Rulebook required")
        rulebook.__post_init__()
        self.rulebook = rulebook
        self.kernel_id = uuid4().hex  # session binding only, not an authentication secret
        self.evidence_epoch = 0
        self.policy_epoch = 0
        self.evidence = {}
        self.records = {}
        self.release_keys = self.policy(release_keys)

    def policy(self, keys):
        if (type(keys) is not tuple or any(type(key) is not str for key in keys)
                or len(set(keys)) != len(keys) or any(key not in self.rulebook.keys for key in keys)):
            raise ValueError("immutable release keys must belong to universe")
        return tuple(sorted(keys))

    def certificate(self, record):
        return Certificate(self.kernel_id, self.evidence_epoch, self.policy_epoch, record)

    def snapshot(self):
        records = tuple(self.records[key] for key in sorted(self.records))
        return Snapshot(self.kernel_id, self.evidence_epoch, self.policy_epoch,
                        self.rulebook, self.release_keys, records,
                        tuple(self.certificate(record) for record in records))

    def reset_records(self):
        # Conservative dependency invalidation, including unchanged replacements.
        self.evidence_epoch += 1
        self.records = {"fact." + key: Record("fact." + key, claim, "evidence", key)
                        for key, claim in self.evidence.items()}


class HostChannel:
    """Application-only channel. Never pass it to a learned/untrusted proposer."""
    def __init__(self, core):
        self._core = core

    def admit(self, evidence_id: str, claim: Claim) -> Snapshot:
        """Add or replace evidence atomically; conflicts with OTHER facts reject.

        Derived records do not veto a host revision: all are discarded. The
        host authenticates evidence and asserts its truth before calling this.
        """
        _name(evidence_id)
        _claim(claim)
        core = self._core
        if claim.key not in core.rulebook.keys:
            raise ValueError("claim outside declared universe")
        if evidence_id not in core.evidence and len(core.evidence) >= MAX_EVIDENCE:
            raise ValueError("evidence capacity")
        if any(key != evidence_id and old.key == claim.key and old.value != claim.value
               for key, old in core.evidence.items()):
            raise ValueError("conflicting evidence rejected")
        core.evidence[evidence_id] = claim
        core.reset_records()
        return core.snapshot()

    replace = admit

    def revoke(self, evidence_id: str) -> Snapshot:
        _name(evidence_id)
        core = self._core
        if evidence_id not in core.evidence:
            raise ValueError("unknown evidence id")
        del core.evidence[evidence_id]
        core.reset_records()
        return core.snapshot()

    def set_release_keys(self, release_keys: tuple[str, ...]) -> Snapshot:
        core = self._core
        core.release_keys = core.policy(release_keys)
        core.policy_epoch += 1
        return core.snapshot()


class ProposerChannel:
    """Only snapshot, checked rule application and checked release are exposed.

    Snapshots intentionally expose all facts/rules/policy to both comparison
    arms. This interface makes no secrecy or confidentiality claim.
    """
    def __init__(self, core):
        self._core = core

    def snapshot(self) -> Snapshot:
        return self._core.snapshot()

    def step(self, proposal: Derivation) -> StepResult:
        core = self._core
        try:
            if type(proposal) is not Derivation:
                raise ValueError("exact Derivation required; facts and policy are host-only")
            _name(proposal.record_id, 69)
            _name(proposal.rule_id)
            if type(proposal.premises) is not tuple or len(proposal.premises) > MAX_KEYS:
                raise ValueError("bounded immutable premise ids required")
            for record_id in proposal.premises:
                _name(record_id, 69)
            if proposal.record_id in proposal.premises:
                raise ValueError("cyclic derivation")
            if proposal.record_id in core.records:
                raise ValueError("record id already exists")
            if len(core.records) >= MAX_RECORDS:
                raise ValueError("record capacity")
            rules = {rule.rule_id: rule for rule in core.rulebook.rules}
            if proposal.rule_id not in rules:
                raise ValueError("undeclared rule")
            rule = rules[proposal.rule_id]
            if (len(proposal.premises) != len(rule.premises)
                    or any(key not in core.records for key in proposal.premises)):
                raise ValueError("missing or wrong number of premises")
            if tuple(core.records[key].claim for key in proposal.premises) != rule.premises:
                raise ValueError("premises do not match declared rule")
            if any(old.claim.key == rule.conclusion.key and old.claim.value != rule.conclusion.value
                   for old in core.records.values()):
                raise ValueError("conflicting derivation rejected")
            record = Record(proposal.record_id, rule.conclusion, "rule", rule.rule_id, proposal.premises)
            core.records[record.record_id] = record
            return StepResult(True, "declared rule checked", core.certificate(record), core.snapshot())
        except ValueError as error:
            return StepResult(False, str(error), None, core.snapshot())

    def release(self, certificate: Certificate) -> Release:
        core = self._core
        try:
            if (type(certificate) is not Certificate or type(certificate.record) is not Record
                    or type(certificate.kernel_id) is not str
                    or type(certificate.evidence_epoch) is not int or type(certificate.policy_epoch) is not int):
                raise ValueError("malformed certificate")
            supplied = certificate.record
            _name(supplied.record_id, 69)
            _claim(supplied.claim)
            _name(supplied.source_id)
            if (type(supplied.source) is not str or supplied.source not in ("evidence", "rule")
                    or type(supplied.premises) is not tuple or len(supplied.premises) > MAX_KEYS):
                raise ValueError("malformed certificate record")
            for record_id in supplied.premises:
                _name(record_id, 69)
        except ValueError:
            return Release(False, "malformed certificate")
        record = core.records.get(certificate.record.record_id)
        if record is None or certificate != core.certificate(record):
            return Release(False, "stale or foreign certificate")
        if record.claim.key not in core.release_keys:
            return Release(False, "release denied by host policy")
        text = record.claim.key + "=" + ("true" if record.claim.value else "false")
        return Release(True, "current support and policy checked", text, record.claim)


def create_supported_state(rulebook: Rulebook, release_keys: tuple[str, ...] = ()):
    """Return (trusted host channel, proposer channel); no untrusted deserializer.

    Calls must be serialized by the host. Authentication, truthful evidence,
    sound declared rules, channel ownership and correct deployment are assumed.
    """
    core = _Core(rulebook, release_keys)
    return HostChannel(core), ProposerChannel(core)
