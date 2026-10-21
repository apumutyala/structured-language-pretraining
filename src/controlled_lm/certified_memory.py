"""Tiny, executable set-inclusion certificates for uncertain workflow memory.

This is a finite reference (1 <= n <= 6), not a trained model or proof assistant.
The environment owns policy and authenticates Receipt/Parity/CountReceipt observations. Neural
or other untrusted code may propose *only* a Memory. See the experiment README
for the transition semantics, induction argument, and asynchronous boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import sys
from typing import TypeAlias


def _integer(value: int, name: str, lo: int, hi: int) -> None:
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")


def _dimension(n: int) -> None:
    _integer(n, "n", 1, 6)


def enumerate_worlds(n: int) -> tuple[int, ...]:
    """World w has completion bit i equal to (w >> i) & 1."""
    _dimension(n)
    return tuple(range(1 << n))


@dataclass(frozen=True, slots=True)
class ExactMemory:
    n: int
    world_bits: int

    def __post_init__(self) -> None:
        _dimension(self.n)
        _integer(self.world_bits, "world_bits", 0, (1 << (1 << self.n)) - 1)

    @classmethod
    def from_worlds(cls, n: int, worlds) -> ExactMemory:
        _dimension(n)
        bits = 0
        for world in worlds:
            _integer(world, "world", 0, (1 << n) - 1)
            bits |= 1 << world
        return cls(n, bits)


@dataclass(frozen=True, slots=True)
class FactorMemory:
    """Independent bits: unset in both masks means unknown."""
    n: int
    known_zero: int = 0
    known_one: int = 0

    def __post_init__(self) -> None:
        _dimension(self.n)
        for name in ("known_zero", "known_one"):
            _integer(getattr(self, name), name, 0, (1 << self.n) - 1)
        if self.known_zero & self.known_one:
            raise ValueError("factor masks conflict")


@dataclass(frozen=True, slots=True)
class Equation:
    mask: int
    rhs: int

    def __post_init__(self) -> None:
        _integer(self.mask, "mask", 0, 63)
        _integer(self.rhs, "rhs", 0, 1)


@dataclass(frozen=True, slots=True)
class AffineMemory:
    """Conjunction of Boolean affine equations, including inconsistent systems."""
    n: int
    equations: tuple[Equation, ...] = ()

    def __post_init__(self) -> None:
        _dimension(self.n)
        if type(self.equations) is not tuple or len(self.equations) > 255:
            raise ValueError("equations must be a tuple of at most 255 rows")
        for equation in self.equations:
            if type(equation) is not Equation or equation.mask >= 1 << self.n:
                raise ValueError("equation outside memory dimension")


Memory: TypeAlias = ExactMemory | FactorMemory | AffineMemory


def concretize(memory: Memory) -> frozenset[int]:
    """Independent exhaustive gamma; deliberately does not use Gaussian elimination."""
    if type(memory) is ExactMemory:
        return frozenset(w for w in enumerate_worlds(memory.n) if memory.world_bits >> w & 1)
    if type(memory) is FactorMemory:
        return frozenset(w for w in enumerate_worlds(memory.n)
                         if not w & memory.known_zero and w & memory.known_one == memory.known_one)
    if type(memory) is AffineMemory:
        return frozenset(w for w in enumerate_worlds(memory.n)
                         if all((w & row.mask).bit_count() % 2 == row.rhs for row in memory.equations))
    raise ValueError("unsupported memory representation")


def factor_hull(n: int, worlds) -> FactorMemory:
    """Smallest independent-factor overapproximation of a nonempty set."""
    exact = ExactMemory.from_worlds(n, worlds)
    concrete = concretize(exact)
    if not concrete:
        raise ValueError("factor hull requires nonempty worlds")
    ones, zeros = (1 << n) - 1, (1 << n) - 1
    for world in concrete:
        ones &= world
        zeros &= ~world
    return FactorMemory(n, zeros, ones)


def gaussian_eliminate(n: int, equations: tuple[Equation, ...]) -> AffineMemory:
    """Canonical reduced GF(2) system; this generator is not trusted by gamma."""
    source = AffineMemory(n, equations)
    basis: dict[int, tuple[int, int]] = {}
    for equation in source.equations:
        mask, rhs = equation.mask, equation.rhs
        while mask:
            pivot = mask.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (mask, rhs)
                break
            old_mask, old_rhs = basis[pivot]
            mask, rhs = mask ^ old_mask, rhs ^ old_rhs
        if not mask and rhs:
            return AffineMemory(n, (Equation(0, 1),))
    for pivot in sorted(basis):
        mask, rhs = basis[pivot]
        for other in sorted(basis):
            if other != pivot and basis[other][0] >> pivot & 1:
                old_mask, old_rhs = basis[other]
                basis[other] = (old_mask ^ mask, old_rhs ^ rhs)
    return AffineMemory(n, tuple(Equation(*basis[p]) for p in sorted(basis, reverse=True)))


def affine_hull(n: int, worlds) -> AffineMemory:
    """Smallest affine overapproximation, found by enumerating true equations."""
    concrete = concretize(ExactMemory.from_worlds(n, worlds))
    if not concrete:
        return AffineMemory(n, (Equation(0, 1),))
    anchor = min(concrete)
    rows = tuple(Equation(mask, (anchor & mask).bit_count() % 2)
                 for mask in range(1, 1 << n)
                 if len({(w & mask).bit_count() % 2 for w in concrete}) == 1)
    return gaussian_eliminate(n, rows)


@dataclass(frozen=True, slots=True)
class ProtectedPolicy:
    """Caller-owned grant mask and remaining submission-attempt budget."""
    n: int
    grants: int
    budget: int

    def __post_init__(self) -> None:
        _dimension(self.n)
        _integer(self.grants, "grants", 0, (1 << self.n) - 1)
        _integer(self.budget, "budget", 0, sys.maxsize)


@dataclass(frozen=True, slots=True)
class CertifiedState:
    memory: Memory
    policy: ProtectedPolicy

    def __post_init__(self) -> None:
        if type(self.policy) is not ProtectedPolicy or self.memory.n != self.policy.n:
            raise ValueError("memory and protected policy dimensions must agree")
        if not concretize(self.memory):
            raise ValueError("certified state must have nonempty concretization")


def initial_state(n: int, *, grants: int | None = None, budget: int | None = None,
                  memory: Memory | None = None, worlds=(0,)) -> CertifiedState:
    """Check initial hidden-world inclusion; default environment starts all-zero."""
    _dimension(n)
    true_worlds = concretize(ExactMemory.from_worlds(n, worlds))
    if not true_worlds:
        raise ValueError("initial worlds must be nonempty")
    proposal = ExactMemory.from_worlds(n, true_worlds) if memory is None else memory
    if proposal.n != n or not true_worlds <= concretize(proposal):
        raise ValueError("initial memory omits possible worlds")
    return CertifiedState(proposal, ProtectedPolicy(n, (1 << n) - 1 if grants is None else grants,
                                                   n if budget is None else budget))


@dataclass(frozen=True, slots=True)
class Submit:
    index: int


@dataclass(frozen=True, slots=True)
class Wait:
    pass


Action: TypeAlias = Submit | Wait


@dataclass(frozen=True, slots=True)
class Timeout:
    """No receipt; never evidence of failure."""


@dataclass(frozen=True, slots=True)
class Receipt:
    """Authenticated observation of current completion (not arbitrary old receipt)."""
    index: int
    success: bool

    def __post_init__(self) -> None:
        if type(self.success) is not bool:
            raise ValueError("success must be bool")


@dataclass(frozen=True, slots=True)
class Parity:
    """Authenticated current-world parity measurement."""
    mask: int
    rhs: int

    def __post_init__(self) -> None:
        _integer(self.mask, "mask", 0, 63)
        _integer(self.rhs, "rhs", 0, 1)


@dataclass(frozen=True, slots=True)
class CountConstraint:
    """Proposed count proposition; constructing it confers no observation authority."""
    mask: int
    count: int

    def __post_init__(self) -> None:
        _integer(self.mask, "mask", 0, 63)
        _integer(self.count, "count", 0, self.mask.bit_count())


@dataclass(frozen=True, slots=True)
class CountReceipt(CountConstraint):
    """Authenticated current completion count on the selected bit mask.

    Exact cardinality is generally non-affine; e.g. exactly one of three bits
    gives {001, 010, 100}, whose affine hull also includes 111.
    """


@dataclass(frozen=True, slots=True)
class UntrustedText:
    text: str


Observation: TypeAlias = Timeout | Receipt | Parity | CountReceipt | UntrustedText | None


def validate_observation(n: int, observation: Observation) -> None:
    """Validate protocol shape/dimension before dispatch; does not authenticate.

    Hosts may call this before external IO. Contradiction with the possible
    worlds is a later semantic check, not a malformed packet.
    """
    _dimension(n)
    if observation is None or type(observation) in (Timeout, UntrustedText):
        return
    if type(observation) is Receipt:
        _integer(observation.index, "receipt index", 0, n - 1)
    elif type(observation) in (Parity, CountReceipt):
        _integer(observation.mask, "observation mask", 0, (1 << n) - 1)
    else:
        raise ValueError("unsupported observation")


def legal_actions(state: CertifiedState) -> tuple[Action, ...]:
    """Robust SUBMIT requires grant, budget and bit=0 in *every* possible world."""
    worlds = concretize(state.memory)
    submits = tuple(Submit(i) for i in range(state.policy.n)
                    if state.policy.budget > 0 and state.policy.grants >> i & 1
                    and all(not w >> i & 1 for w in worlds))
    return (Wait(),) + submits


class IllegalAction(ValueError):
    pass


class InconsistentObservation(ValueError):
    pass


def post_worlds(prior: CertifiedState, action: Action,
                observation: Observation = None) -> frozenset[int]:
    """Exact Post(gamma(prior), action, observation), independently enumerated."""
    validate_observation(prior.policy.n, observation)
    if type(action) is Submit:
        _integer(action.index, "index", 0, prior.policy.n - 1)
    elif type(action) is not Wait:
        raise IllegalAction("unsupported action")
    if action not in legal_actions(prior):
        raise IllegalAction("submission lacks robust zero bit, grant, or budget")
    worlds = concretize(prior.memory)
    if type(action) is Submit:
        worlds = frozenset(outcome for w in worlds for outcome in (w, w | (1 << action.index)))
    if observation is None or type(observation) in (Timeout, UntrustedText):
        return worlds
    if type(observation) is Receipt:
        worlds = frozenset(w for w in worlds if bool(w >> observation.index & 1) == observation.success)
    elif type(observation) is Parity:
        worlds = frozenset(w for w in worlds if (w & observation.mask).bit_count() % 2 == observation.rhs)
    elif type(observation) is CountReceipt:
        worlds = frozenset(w for w in worlds if (w & observation.mask).bit_count() == observation.count)
    else:
        raise ValueError("unsupported observation")
    if not worlds:
        raise InconsistentObservation("observation contradicts all possible worlds")
    return worlds


@dataclass(frozen=True, slots=True)
class UpdateResult:
    state: CertifiedState
    fallback: CertifiedState
    accepted: bool
    event_applied: bool
    reason: str
    omitted_worlds: frozenset[int] = frozenset()


def checked_update(prior: CertifiedState, action: Action, observation: Observation,
                   candidate: Memory) -> UpdateResult:
    """Check Post subset gamma(candidate); rejected proposals cannot change policy.

    A legal event advances even when its summary is rejected. The immutable exact
    post-state fallback is therefore used instead of the stale pre-event memory.
    Illegal actions reject the event. Contradictory observations are discarded,
    but any legal submission still advances the state and consumes its budget;
    rejecting a receipt cannot undo an external side effect.
    """
    validate_observation(prior.policy.n, observation)
    try:
        worlds = post_worlds(prior, action, None)
    except IllegalAction as error:
        return UpdateResult(prior, prior, False, False, str(error))
    policy = prior.policy
    if type(action) is Submit:
        policy = ProtectedPolicy(policy.n, policy.grants, policy.budget - 1)
    try:
        worlds = post_worlds(prior, action, observation)
    except InconsistentObservation as error:
        fallback = CertifiedState(ExactMemory.from_worlds(policy.n, worlds), policy)
        return UpdateResult(fallback, fallback, False, True, str(error))
    fallback = CertifiedState(ExactMemory.from_worlds(policy.n, worlds), policy)
    if type(candidate) not in (ExactMemory, FactorMemory, AffineMemory) or candidate.n != policy.n:
        return UpdateResult(fallback, fallback, False, True, "invalid candidate representation or dimension")
    omitted = worlds - concretize(candidate)
    if omitted:
        return UpdateResult(fallback, fallback, False, True, "candidate omits possible worlds", omitted)
    return UpdateResult(CertifiedState(candidate, policy), fallback, True, True, "inclusion checked")


def grounded_assert(memory: Memory, proposition: Equation | CountConstraint) -> bool:
    """Emit TRUE only for nonempty universal support; False means abstain."""
    if type(proposition) not in (Equation, CountConstraint) or proposition.mask >= 1 << memory.n:
        raise ValueError("proposition outside memory dimension")
    worlds = concretize(memory)
    if type(proposition) is CountConstraint:
        return bool(worlds) and all((w & proposition.mask).bit_count() == proposition.count for w in worlds)
    return bool(worlds) and all((w & proposition.mask).bit_count() % 2 == proposition.rhs for w in worlds)


def pack_memory(memory: Memory) -> bytes:
    """Actual documented encoding incl tag/dimension/count; no compression claim."""
    concretize(memory)  # strict representation validation
    if type(memory) is ExactMemory:
        return bytes((0, memory.n)) + memory.world_bits.to_bytes(((1 << memory.n) + 7) // 8, "little")
    if type(memory) is FactorMemory:
        return bytes((1, memory.n, memory.known_zero, memory.known_one))
    return bytes((2, memory.n, len(memory.equations))) + bytes(
        value for row in memory.equations for value in (row.mask, row.rhs))


def python_storage_bytes(value) -> int:
    """CPython reachable-object size, counting shared identities once per root.

    Includes dataclass slots and referenced integers/tuples/rows, excludes class
    objects, interpreter allocator overhead, protected policy and checker work.
    This is runtime-dependent; encoded size is len(pack_memory(memory)).
    """
    seen: set[int] = set()

    def size(obj) -> int:
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        total = sys.getsizeof(obj)
        if is_dataclass(obj) and not isinstance(obj, type):
            total += sum(size(getattr(obj, field.name)) for field in fields(obj))
        elif isinstance(obj, (tuple, list, frozenset, set)):
            total += sum(size(item) for item in obj)
        return total

    return size(value)
