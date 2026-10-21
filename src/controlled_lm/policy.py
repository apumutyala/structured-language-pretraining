"""Exact finite policy and explicit trusted protocol boundary.

No neural tensor is stored here. Python capability identity protects the public
interface from accidental data-to-authority conversion; it is not an in-process
security sandbox against a caller able to alter objects or execute arbitrary code.
"""
from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping, Sequence

ACTIONS = ("select", "count", "sum", "cite_supported_fact", "request_evidence", "clarify", "abstain", "halt")
ACTION_TO_ID = {name: index for index, name in enumerate(ACTIONS)}
HALT_ID = 7
READ, AGGREGATE, CITE, PLAN = 1, 2, 4, 8


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an exact integer in [{lower}, {upper}]")
    return value


@dataclass(frozen=True, slots=True)
class Policy:
    caps: int = 0
    budget: int = 0
    revoked: bool = False
    policy_version: int = 0
    prefix_position: int = 0
    witness_valid: bool = False
    principal: str = ""
    tenant: str = ""
    allowed_operands: tuple[int, ...] = ()

    def __post_init__(self):
        for name, upper in (("caps", 15), ("budget", 8), ("policy_version", 1), ("prefix_position", 8)):
            _integer(getattr(self, name), name, 0, upper)
        if type(self.revoked) is not bool or type(self.witness_valid) is not bool:
            raise ValueError("revoked and witness_valid must be exact booleans")
        if type(self.principal) is not str or type(self.tenant) is not str:
            raise ValueError("principal and tenant must be strings")
        if type(self.allowed_operands) is not tuple:
            raise ValueError("allowed_operands must be an immutable tuple")
        for value in self.allowed_operands:
            _integer(value, "operand", 0, 15)
        if len(set(self.allowed_operands)) != len(self.allowed_operands):
            raise ValueError("duplicate allowed operands")

    @classmethod
    def from_mapping(cls, value: Mapping):
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("policy must be a mapping or Policy")
        fields = dict(value)
        for source, target in (("version", "policy_version"), ("prefix", "prefix_position"), ("witness", "witness_valid")):
            if source in fields:
                if target in fields and (fields[target] != fields[source] or type(fields[target]) is not type(fields[source])):
                    raise ValueError(f"conflicting {source}/{target}")
                fields[target] = fields.pop(source)
        if "allowed_operands" in fields:
            operands = fields["allowed_operands"]
            if type(operands) not in (tuple, list):
                raise ValueError("allowed_operands must be a list or tuple")
            fields["allowed_operands"] = tuple(operands)
        return cls(**fields)

    def action_allowed(self, action, operands=None):
        return action_allowed(self, action, operands)


def action_allowed(policy: Policy, action: str, operands: Sequence[int] | None = None) -> bool:
    if type(policy) is not Policy or type(action) is not str or action not in ACTION_TO_ID:
        return False
    if action in ("clarify", "abstain", "halt"):
        return True
    if policy.revoked or policy.budget == 0 or policy.prefix_position == 8:
        return False
    if action == "request_evidence":
        return bool(policy.caps & READ) and not policy.witness_valid
    if policy.prefix_position > 0 and not policy.caps & PLAN:
        return False
    required = READ | (AGGREGATE if action in ("count", "sum") else CITE if action == "cite_supported_fact" else 0)
    if policy.caps & required != required or not policy.witness_valid:
        return False
    if operands is not None:
        if type(operands) not in (list, tuple) or policy.prefix_position + len(operands) > 8:
            return False
        if any(type(operand) is not int or operand not in policy.allowed_operands for operand in operands):
            return False
    return True


check_proposal = action_allowed


def operand_allowed(policy: Policy, operand: int, prefix: Sequence[int] = ()) -> bool:
    return (type(policy) is Policy and type(operand) is int and operand in policy.allowed_operands
            and action_allowed(policy, "select")
            and type(prefix) in (tuple, list) and policy.prefix_position + len(prefix) < 8
            and all(type(item) is int and item in policy.allowed_operands for item in prefix))


class EventKind(Enum):
    REPLACE_POLICY = "replace_policy"


@dataclass(frozen=True, slots=True)
class TrustedEvent:
    kind: EventKind
    policy: Policy
    _issuer: object


class TrustedCapability:
    """Held by the trusted application, never passed into a model forward call."""
    __slots__ = ("_identity",)

    def __init__(self):
        self._identity = object()

    def issue(self, policy: Policy) -> TrustedEvent:
        if type(policy) is not Policy:
            raise TypeError("trusted replacement requires an already validated Policy")
        return TrustedEvent(EventKind.REPLACE_POLICY, policy, self._identity)


@dataclass(frozen=True, slots=True)
class ProtectedState:
    policy: Policy
    revision: int = 0

    def __post_init__(self):
        if type(self.policy) is not Policy or type(self.revision) is not int or self.revision < 0:
            raise ValueError("invalid protected state")

    @classmethod
    def initial(cls, policy):
        return cls(Policy.from_mapping(policy))


def trusted_transition(state: ProtectedState, event: TrustedEvent, capability: TrustedCapability) -> ProtectedState:
    if (type(state) is not ProtectedState or type(event) is not TrustedEvent
            or type(capability) is not TrustedCapability or event._issuer is not capability._identity
            or event.kind is not EventKind.REPLACE_POLICY or type(event.policy) is not Policy):
        raise TypeError("protected writes require the issuing trusted capability")
    return replace(state, policy=event.policy, revision=state.revision + 1)


def execute_action(state: ProtectedState, action: str, operands: Sequence[int], capability: TrustedCapability) -> ProtectedState:
    """Trusted deterministic accounting; a proposal cannot grant permissions."""
    if type(capability) is not TrustedCapability or type(state) is not ProtectedState:
        raise TypeError("execution requires the trusted application capability")
    if not action_allowed(state.policy, action, operands):
        raise ValueError("illegal proposal")
    if action in ("clarify", "abstain", "halt"):
        return state
    policy = replace(state.policy, budget=state.policy.budget - 1,
                     prefix_position=state.policy.prefix_position + (action != "request_evidence"))
    return replace(state, policy=policy, revision=state.revision + 1)


ControlState = ProtectedState
