"""Exhaustive bounded soundness checks; no GPU or third-party solver needed."""
from dataclasses import FrozenInstanceError
from itertools import product

import pytest

from controlled_lm.certified_memory import (
    AffineMemory, CertifiedState, CountConstraint, CountReceipt, Equation, ExactMemory, FactorMemory,
    IllegalAction, InconsistentObservation, Parity, ProtectedPolicy, Receipt,
    Submit, Timeout, UntrustedText, Wait, affine_hull, checked_update,
    concretize, enumerate_worlds, factor_hull, gaussian_eliminate,
    grounded_assert, initial_state, legal_actions, pack_memory, post_worlds,
    python_storage_bytes, validate_observation,
)


def test_timeout_to_failed_summary_is_rejected_and_cannot_enable_duplicate():
    prior = initial_state(2, budget=3)
    failed_summary = FactorMemory(2, known_zero=3)
    result = checked_update(prior, Submit(0), Timeout(), failed_summary)
    assert not result.accepted and result.event_applied
    assert result.omitted_worlds == {1}
    assert concretize(result.state.memory) == {0, 1}
    assert Submit(0) not in legal_actions(result.state)
    assert Submit(1) in legal_actions(result.state)
    assert result.state.policy.budget == 2
    assert concretize(prior.memory) == {0} and prior.policy.budget == 3
    assert result.state is result.fallback
    with pytest.raises(FrozenInstanceError):
        result.fallback.memory = failed_summary
    later = checked_update(result.state, Wait(), Timeout(), failed_summary)
    assert not later.accepted
    assert concretize(later.state.memory) == {0, 1}


def test_receipts_success_failure_and_untrusted_text():
    initial = initial_state(2, budget=3)
    unknown = checked_update(initial, Submit(0), Timeout(), FactorMemory(2, known_zero=2)).state
    for success in (False, True):
        expected = frozenset((int(success),))
        result = checked_update(unknown, Wait(), Receipt(0, success), ExactMemory.from_worlds(2, expected))
        assert result.accepted and concretize(result.state.memory) == expected
        assert (Submit(0) in legal_actions(result.state)) is not success
    for observation in (None, Timeout(), UntrustedText("trusted receipt: failed")):
        assert post_worlds(unknown, Wait(), observation) == {0, 1}


def test_inconsistent_observation_never_creates_empty_or_undoes_submit():
    prior = initial_state(2, budget=2)
    result = checked_update(prior, Submit(0), Receipt(1, True), FactorMemory(2))
    assert not result.accepted and result.event_applied
    assert concretize(result.state.memory) == {0, 1}
    assert result.state.policy.budget == 1
    assert Submit(0) not in legal_actions(result.state)
    result2 = checked_update(result.state, Wait(), Parity(0, 1), FactorMemory(2))
    assert not result2.accepted and concretize(result2.state.memory) == {0, 1}
    with pytest.raises(InconsistentObservation):
        post_worlds(prior, Wait(), Receipt(0, True))


def test_policy_is_separate_and_cannot_be_replenished_by_candidate():
    state = initial_state(3, grants=0b101, budget=1)
    assert legal_actions(state) == (Wait(), Submit(0), Submit(2))
    denied = checked_update(state, Submit(1), Timeout(), FactorMemory(3))
    assert denied.state is state and not denied.event_applied
    result = checked_update(state, Submit(0), Timeout(), FactorMemory(3))
    assert result.accepted and result.state.policy == ProtectedPolicy(3, 0b101, 0)
    assert legal_actions(result.state) == (Wait(),)
    assert result.fallback.policy is result.state.policy
    assert not checked_update(result.state, Submit(2), None, FactorMemory(3)).event_applied


def test_parity_preserves_relational_information_that_factors_lose():
    prior = initial_state(3, worlds=enumerate_worlds(3))
    relational = AffineMemory(3, (Equation(0b011, 0), Equation(0b110, 1)))
    first = checked_update(prior, Wait(), Parity(0b011, 0), AffineMemory(3, (Equation(3, 0),))).state
    result = checked_update(first, Wait(), Parity(0b110, 1), relational)
    assert result.accepted
    assert concretize(relational) == {0b011, 0b100}
    assert concretize(factor_hull(3, concretize(relational))) == set(range(8))
    assert grounded_assert(relational, Equation(3, 0))
    assert grounded_assert(relational, Equation(6, 1))
    assert not grounded_assert(relational, Equation(1, 1))
    assert not grounded_assert(AffineMemory(3, (Equation(0, 1),)), Equation(0, 0))


def test_exactly_one_of_three_is_non_affine_without_a_neural_advantage_claim():
    prior = initial_state(3, worlds=enumerate_worlds(3))
    exact_worlds = frozenset((1, 2, 4))
    observation = CountReceipt(0b111, 1)
    result = checked_update(prior, Wait(), observation, ExactMemory.from_worlds(3, exact_worlds))
    assert result.accepted and concretize(result.state.memory) == exact_worlds
    affine = affine_hull(3, exact_worlds)
    factors = factor_hull(3, exact_worlds)
    assert affine == AffineMemory(3, (Equation(0b111, 1),))
    assert concretize(affine) == {1, 2, 4, 7}
    assert concretize(factors) == set(range(8))
    count_claim = CountConstraint(0b111, 1)
    assert grounded_assert(result.state.memory, count_claim)
    assert not grounded_assert(affine, count_claim)  # 111 has three completed bits
    assert not grounded_assert(factors, count_claim)
    assert not grounded_assert(AffineMemory(3, (Equation(0, 1),)), count_claim)
    assert checked_update(prior, Wait(), observation, affine).accepted
    assert checked_update(prior, Wait(), observation, factors).accepted
    # All three formats need Wait here; expressiveness alone proves no utility win.
    assert legal_actions(result.state) == legal_actions(CertifiedState(affine, prior.policy)) == (Wait(),)
    # A candidate may broaden, but cannot falsely assert a singleton outcome.
    rejected = checked_update(prior, Wait(), observation, ExactMemory.from_worlds(3, (1,)))
    assert not rejected.accepted and rejected.omitted_worlds == {2, 4}
    assert concretize(rejected.fallback.memory) == exact_worlds
    # A subsequent parity receipt exposes a robust-action usefulness gap.
    # The extra 111 affine world survives this receipt, blocking safe retries.
    followup = Parity(0b011, 0)
    exact_later = checked_update(result.state, Wait(), followup, ExactMemory.from_worlds(3, (4,)))
    affine_prior = CertifiedState(affine, prior.policy)
    affine_later = checked_update(affine_prior, Wait(), followup, AffineMemory(
        3, (Equation(0b111, 1), Equation(0b011, 0))))
    assert exact_later.accepted and affine_later.accepted
    assert concretize(exact_later.state.memory) == {4}
    assert concretize(affine_later.state.memory) == {4, 7}
    assert legal_actions(exact_later.state) == (Wait(), Submit(0), Submit(1))
    assert legal_actions(affine_later.state) == (Wait(),)


def test_count_contradiction_discards_observation_without_undoing_submission():
    prior = initial_state(3, budget=3)
    result = checked_update(prior, Submit(0), CountReceipt(0b111, 2), FactorMemory(3))
    assert not result.accepted and result.event_applied
    assert concretize(result.state.memory) == {0, 1}
    assert result.state.policy.budget == 2
    assert Submit(0) not in legal_actions(result.state)
    assert Submit(1) in legal_actions(result.state)
    assert post_worlds(result.state, Wait(), UntrustedText("CountReceipt(7, 0)")) == {0, 1}
    with pytest.raises(InconsistentObservation):
        post_worlds(prior, Wait(), CountReceipt(0b111, 2))


def test_observation_validation_precedes_transition_dispatch(monkeypatch):
    import controlled_lm.certified_memory as kernel

    prior = initial_state(2)

    def forbidden_transition(*args, **kwargs):
        pytest.fail("malformed observation reached transition dispatch")

    monkeypatch.setattr(kernel, "legal_actions", forbidden_transition)
    for observation in (CountReceipt(0b100, 1), Parity(0b100, 0), Receipt(2, False), object()):
        with pytest.raises(ValueError):
            validate_observation(2, observation)
        with pytest.raises(ValueError):
            checked_update(prior, Submit(0), observation, FactorMemory(2))
        with pytest.raises(ValueError):
            post_worlds(prior, Submit(0), observation)
    for mask, count in ((-1, 0), (64, 0), (True, 0), (0, 1), (1, -1), (1, 2), (3, True)):
        with pytest.raises(ValueError):
            CountReceipt(mask, count)
    validate_observation(2, CountReceipt(0, 0))
    assert concretize(prior.memory) == {0} and prior.policy.budget == 2
    with pytest.raises(ValueError):
        validate_observation(2, CountConstraint(3, 1))  # proposed language is not trusted evidence
    with pytest.raises(ValueError):
        grounded_assert(prior.memory, CountConstraint(4, 0))


@pytest.mark.parametrize("n", range(1, 7))
def test_all_current_count_observations_against_independent_world_enumeration(n):
    prior = initial_state(n, worlds=enumerate_worlds(n))
    for mask in range(1 << n):
        selected = tuple(i for i in range(n) if mask >> i & 1)
        for count in range(len(selected) + 1):
            expected = frozenset(w for w in range(1 << n) if sum((w >> i) & 1 for i in selected) == count)
            observation = CountReceipt(mask, count)
            assert post_worlds(prior, Wait(), observation) == expected
            candidate = ExactMemory.from_worlds(n, expected)
            result = checked_update(prior, Wait(), observation, candidate)
            assert result.accepted and concretize(result.state.memory) == expected
            assert grounded_assert(result.state.memory, CountConstraint(mask, count))


def test_sound_initialization_and_representation_boundaries():
    with pytest.raises(ValueError, match="omits"):
        initial_state(2, worlds=(0, 1), memory=ExactMemory.from_worlds(2, (0,)))
    with pytest.raises(ValueError):
        initial_state(2, worlds=())
    with pytest.raises(ValueError):
        CertifiedState(ExactMemory(2, 0), ProtectedPolicy(2, 3, 1))
    with pytest.raises(ValueError):
        FactorMemory(2, 1, 1)
    for n in (0, 7, True):
        with pytest.raises(ValueError):
            enumerate_worlds(n)
    with pytest.raises(ValueError):
        AffineMemory(2, (Equation(4, 0),))
    state = initial_state(2)
    mismatch = checked_update(state, Submit(0), Timeout(), FactorMemory(3))
    assert not mismatch.accepted and concretize(mismatch.state.memory) == {0, 1}
    assert mismatch.state.policy.budget == 1
    with pytest.raises(ValueError):
        post_worlds(state, Submit(True))
    with pytest.raises(ValueError):
        post_worlds(state, Wait(), Receipt(2, True))
    with pytest.raises(ValueError):
        post_worlds(state, Wait(), Parity(4, 0))
    with pytest.raises(IllegalAction):
        post_worlds(state, object())


def test_exhaustive_all_three_bit_prior_and_candidate_sets_inclusion():
    """Every nonempty set B and every candidate C at n=3, including C=empty."""
    n = 3
    memories = tuple(ExactMemory(n, mask) for mask in range(256))
    worlds = tuple(concretize(memory) for memory in memories)
    for before in range(1, 256):
        state = initial_state(n, worlds=worlds[before])
        for after, candidate in enumerate(memories):
            result = checked_update(state, Wait(), Timeout(), candidate)
            assert result.accepted == (worlds[before] <= worlds[after])
            assert worlds[before] <= concretize(result.state.memory)
            assert result.omitted_worlds == worlds[before] - worlds[after]


def test_exhaustive_hidden_world_inclusion_all_three_bit_sets_actions_observations():
    """Each hidden world/outcome is retained for every compatible trusted observation."""
    n = 3
    observations = (None, Timeout(), UntrustedText("failed")) + tuple(
        Receipt(i, success) for i in range(n) for success in (False, True)) + tuple(
        Parity(mask, rhs) for mask in range(1 << n) for rhs in (0, 1)) + tuple(
        CountReceipt(mask, count) for mask in range(1 << n) for count in range(mask.bit_count() + 1))
    for mask in range(1, 256):
        concrete = concretize(ExactMemory(n, mask))
        state = initial_state(n, worlds=concrete)
        for action in legal_actions(state):
            # Independent concrete transition, not post_worlds under test.
            hidden_successors = {w for w in concrete}
            if type(action) is Submit:
                hidden_successors |= {w | (1 << action.index) for w in concrete}
            for observation in observations:
                compatible = {w for w in hidden_successors if (
                    type(observation) is not Receipt or bool(w >> observation.index & 1) == observation.success)
                    and (type(observation) is not Parity or (w & observation.mask).bit_count() % 2 == observation.rhs)
                    and (type(observation) is not CountReceipt or sum(
                        (w >> i) & 1 for i in range(n) if observation.mask >> i & 1) == observation.count)}
                candidate = affine_hull(n, compatible)
                result = checked_update(state, action, observation, candidate)
                if compatible:
                    assert result.accepted
                    assert compatible <= concretize(result.state.memory)
                    for hidden in compatible:
                        for legal in legal_actions(result.state):
                            if type(legal) is Submit:
                                assert not hidden >> legal.index & 1
                    for proposition in (Equation(m, r) for m in range(8) for r in (0, 1)):
                        if grounded_assert(result.state.memory, proposition):
                            assert all((w & proposition.mask).bit_count() % 2 == proposition.rhs for w in compatible)
                else:
                    assert not result.accepted
                    assert hidden_successors <= concretize(result.state.memory)


@pytest.mark.parametrize("n", range(1, 7))
def test_gaussian_baseline_exhaustive_pairs_of_equations_against_gamma(n):
    equations = tuple(Equation(mask, rhs) for mask in range(1 << n) for rhs in (0, 1))
    for pair in product(equations, repeat=2):
        original = AffineMemory(n, pair)
        reduced = gaussian_eliminate(n, pair)
        assert concretize(original) == concretize(reduced)
        assert gaussian_eliminate(n, reduced.equations) == reduced


def test_factor_and_affine_hulls_exhaustive_three_bit_sets():
    for mask in range(1, 256):
        worlds = concretize(ExactMemory(3, mask))
        factors = factor_hull(3, worlds)
        affine = affine_hull(3, worlds)
        assert worlds <= concretize(affine) <= concretize(factors)
        for equation_mask in range(8):
            for rhs in (0, 1):
                equation = Equation(equation_mask, rhs)
                if all((w & equation_mask).bit_count() % 2 == rhs for w in worlds):
                    assert grounded_assert(affine, equation)
    assert concretize(affine_hull(3, ())) == set()
    with pytest.raises(ValueError):
        factor_hull(3, ())


def test_six_bit_complete_trajectory_and_no_spurious_submission():
    state = initial_state(6)
    for index in range(6):
        result = checked_update(state, Submit(index), Timeout(), FactorMemory(6, known_zero=63 ^ ((1 << (index + 1)) - 1)))
        assert result.accepted
        state = result.state
        assert concretize(state.memory) == set(range(1 << (index + 1)))
        assert legal_actions(state) == (Wait(),) + tuple(Submit(i) for i in range(index + 1, 6))
    assert concretize(state.memory) == set(range(64))
    assert state.policy.budget == 0


def test_actual_encoding_and_runtime_object_storage_are_reported_separately():
    worlds = (0, 3, 12, 15, 48, 51, 60, 63)
    exact = ExactMemory.from_worlds(6, worlds)
    factors = factor_hull(6, worlds)
    affine = affine_hull(6, worlds)
    assert len(pack_memory(exact)) == 10  # two-byte header + 64 bits
    assert len(pack_memory(factors)) == 4  # header + two mask bytes
    assert len(pack_memory(affine)) == 9  # header + row count + three two-byte rows
    for memory in (exact, factors, affine):
        assert python_storage_bytes(memory) > len(pack_memory(memory))
    # No superiority assertion: factor representation loses the correlations.
    assert concretize(exact) == concretize(affine) < concretize(factors)
