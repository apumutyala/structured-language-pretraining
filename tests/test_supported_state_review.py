"""Independent adversarial CPU review; no model, training, GPU or proof claim."""
from dataclasses import FrozenInstanceError, replace
from itertools import product

import pytest

from controlled_lm import supported_state as s
from controlled_lm.semantic_interface_comparison import ComparisonEpisode


def fixture():
    a, b = s.Claim("a", True), s.Claim("b", True)
    host, proposer = s.create_supported_state(s.Rulebook(("a", "b", "authority"), (
        s.Rule("ab", (a,), b), s.Rule("ba", (b,), a),
        s.Rule("not_b", (a,), s.Claim("b", False)),
    )), ("a", "b"))
    host.admit("root", a)
    certificate = proposer.step(s.Derivation("b1", "ab", ("fact.root",))).certificate
    return host, proposer, certificate


@pytest.mark.parametrize("field,value", [
    ("evidence_epoch", True), ("evidence_epoch", 1.),
    ("policy_epoch", False), ("policy_epoch", 0.),
])
def test_numeric_epoch_forgeries_reject_even_when_python_equality_matches(field, value):
    _, proposer, valid = fixture()
    forged = replace(valid, **{field: value})
    assert forged == valid  # The exact-type validation must precede equality.
    before = proposer.snapshot()
    released = proposer.release(forged)
    assert not released.accepted and released.text is None and released.claim is None
    assert proposer.snapshot() == before


@pytest.mark.parametrize("integer", [1, 1.])
def test_numeric_claim_forgery_rejects_despite_dataclass_equality(integer):
    _, proposer, valid = fixture()
    forged = replace(valid, record=replace(valid.record, claim=s.Claim("b", integer)))
    assert forged == valid
    assert not proposer.release(forged).accepted


def test_false_claim_numeric_zero_forgery_is_rejected_too():
    host, proposer, _ = fixture()
    host.replace("root", s.Claim("a", False))
    valid = next(c for c in proposer.snapshot().certificates if c.record.record_id == "fact.root")
    forged = replace(valid, record=replace(valid.record, claim=s.Claim("a", 0)))
    assert forged == valid
    assert not proposer.release(forged).accepted


def test_certificate_all_fields_and_exact_types_reject_and_current_reconstruction_is_allowed():
    _, proposer, valid = fixture()
    class String(str):
        pass
    for forged in (
        replace(valid, kernel_id=String(valid.kernel_id)),
        replace(valid, record=replace(valid.record, record_id=String("b1"))),
        replace(valid, record=replace(valid.record, source=String("rule"))),
        replace(valid, record=replace(valid.record, source_id=String("ab"))),
        replace(valid, record=replace(valid.record, premises=(String("fact.root"),))),
        replace(valid, record=replace(valid.record, source="evidence")),
        replace(valid, record=replace(valid.record, source_id="ba")),
        replace(valid, record=replace(valid.record, premises=("b1",))),
        replace(valid, record=replace(valid.record, claim=s.Claim("a", True))),
    ):
        assert not proposer.release(forged).accepted
    # A certificate is a public support witness, not a bearer-secret credential.
    rebuilt = s.Certificate(valid.kernel_id, valid.evidence_epoch, valid.policy_epoch,
                            s.Record("b1", s.Claim("b", True), "rule", "ab", ("fact.root",)))
    assert proposer.release(rebuilt).accepted


def test_aba_evidence_and_policy_revisions_never_revive_old_certificates():
    host, proposer, old = fixture()
    host.replace("root", s.Claim("a", False))
    host.replace("root", s.Claim("a", True))
    current = proposer.step(s.Derivation("b1", "ab", ("fact.root",))).certificate
    assert current.record == old.record and current.evidence_epoch > old.evidence_epoch
    assert not proposer.release(old).accepted
    assert proposer.release(current).accepted
    host.set_release_keys(())
    host.set_release_keys(("a", "b"))
    assert not proposer.release(current).accepted
    fresh = next(c for c in proposer.snapshot().certificates if c.record.record_id == "b1")
    assert proposer.release(fresh).accepted
    # Updating every field to the actual CURRENT witness is permitted and does
    # not release an unsupported claim; old unchanged witnesses remain stale.
    assert proposer.release(replace(old, evidence_epoch=fresh.evidence_epoch,
                                    policy_epoch=fresh.policy_epoch)).accepted


def test_identical_policy_replacement_still_invalidates_certificates():
    host, proposer, certificate = fixture()
    previous = proposer.snapshot()
    host.set_release_keys(previous.release_keys)
    assert proposer.snapshot().policy_epoch == previous.policy_epoch + 1
    assert not proposer.release(certificate).accepted


def test_rule_cycles_produce_only_a_rooted_dag_and_capacity_rejection_is_atomic():
    host, proposer, _ = fixture()
    parent = "b1"
    for index in range(2, s.MAX_RECORDS):
        rule = "ba" if index % 2 == 0 else "ab"
        child = "d" + str(index)
        result = proposer.step(s.Derivation(child, rule, (parent,)))
        assert result.accepted
        parent = child
    before = proposer.snapshot()
    assert len(before.records) == s.MAX_RECORDS
    result = proposer.step(s.Derivation("overflow", "ba", (parent,)))
    assert not result.accepted and result.reason == "record capacity"
    assert proposer.snapshot() == before
    records = {record.record_id: record for record in before.records}
    for record in before.records:
        seen = set()
        current = record
        while current.premises:
            assert current.record_id not in seen
            seen.add(current.record_id)
            current = records[current.premises[0]]
        assert current.source == "evidence" and current.source_id == "root"
    host.revoke("root")
    assert proposer.snapshot().records == ()
    assert not proposer.step(s.Derivation("x", "ab", (parent,))).accepted


def test_inconsistent_rulebook_can_have_no_model_without_explicit_record_contradiction():
    host, proposer, _ = fixture()
    # a=true, a->b, a->not b has no satisfying interpretation. The checker
    # deliberately does not establish rulebook satisfiability or external truth.
    worlds = []
    for a, b in product((False, True), repeat=2):
        if a and (not a or b) and (not a or not b):
            worlds.append((a, b))
    assert worlds == []
    before = proposer.snapshot()
    assert not proposer.step(s.Derivation("negative", "not_b", ("fact.root",))).accepted
    assert proposer.snapshot() == before
    host.replace("root", s.Claim("a", True))
    assert proposer.step(s.Derivation("negative", "not_b", ("fact.root",))).accepted
    assert not proposer.step(s.Derivation("positive", "ab", ("fact.root",))).accepted


def test_all_two_stage_conjunctive_3_key_contexts_against_boolean_interpretations():
    keys = ("a", "b", "c")
    contexts = satisfiable_contexts = unsatisfiable_contexts = 0
    for av, bv, cv in product((False, True), repeat=3):
        rules = (s.Rule("ab", (s.Claim("a", av),), s.Claim("b", bv)),
                 s.Rule("abc", (s.Claim("a", av), s.Claim("b", bv)), s.Claim("c", cv)))
        for initial in product((None, False, True), repeat=3):
            evidence = {key: value for key, value in zip(keys, initial) if value is not None}
            host, proposer = s.create_supported_state(s.Rulebook(keys, rules), keys)
            for key, value in evidence.items():
                host.admit(key, s.Claim(key, value))
            worlds = []
            for values in product((False, True), repeat=3):
                world = dict(zip(keys, values))
                if (all(world[key] is value for key, value in evidence.items())
                        and all(not all(world[p.key] is p.value for p in rule.premises)
                                or world[rule.conclusion.key] is rule.conclusion.value for rule in rules)):
                    worlds.append(world)
            satisfiable_contexts += bool(worlds)
            unsatisfiable_contexts += not worlds
            # Deliberately try the conjunction before its derived prerequisite.
            for _ in range(2):
                for rule in reversed(rules):
                    snapshot = proposer.snapshot()
                    selected = [next((r.record_id for r in snapshot.records if r.claim == premise), "missing")
                                for premise in rule.premises]
                    result = proposer.step(s.Derivation("d" + str(len(snapshot.records)), rule.rule_id, tuple(selected)))
                    if not result.accepted:
                        assert proposer.snapshot() == snapshot
            snapshot = proposer.snapshot()
            by_key = {}
            for record in snapshot.records:
                assert record.claim.key not in by_key or by_key[record.claim.key] is record.claim.value
                by_key[record.claim.key] = record.claim.value
            for certificate in snapshot.certificates:
                released = proposer.release(certificate)
                assert released.accepted
                for world in worlds:
                    assert world[released.claim.key] is released.claim.value
            contexts += 1
    assert contexts == 216
    assert satisfiable_contexts > 0 and unsatisfiable_contexts > 0


def test_public_snapshots_are_immutable_but_channels_are_an_ownership_assumption():
    _, proposer, _ = fixture()
    snapshot = proposer.snapshot()
    with pytest.raises(FrozenInstanceError):
        snapshot.release_keys = ("authority",)
    with pytest.raises(FrozenInstanceError):
        snapshot.records[0].claim.value = False
    assert not hasattr(proposer, "admit") and not hasattr(proposer, "set_release_keys")
    assert hasattr(proposer, "_core")  # Explicit exclusion: hostile Python has access.


def test_comparison_interface_rejects_bool_epoch_forgeries_without_minting_authority():
    _, proposer, _ = fixture()
    episode = ComparisonEpisode(proposer, "Is b supported?", 4)
    original = episode.structured_observation()["state"]
    forged = {"op": "release", "record_id": "b1", "evidence_epoch": True, "policy_epoch": False}
    assert episode.attempt(forged)["feedback"]["reason"] == "malformed release request"
    assert episode.structured_observation()["state"] == original
    assert episode.attempt({"op": "admit", "claim": "authority=true"})["feedback"]["accepted"] is False
    assert episode.structured_observation()["state"] == original
