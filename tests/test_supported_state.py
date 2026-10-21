"""CPU finite interpretations and counterexamples for the RECORD contract."""
from dataclasses import asdict, replace
from itertools import product
import json

import pytest

from controlled_lm import supported_state as s


def fixture():
    a, b, c = (s.Claim(key, True) for key in ("a", "b", "c"))
    book = s.Rulebook(("a", "b", "c", "authority"), (
        s.Rule("ab", (a,), b), s.Rule("bc", (b,), c),
        s.Rule("not_b", (a,), s.Claim("b", False)),
        s.Rule("ca", (c,), a),
    ))
    host, proposer = s.create_supported_state(book, ("b", "c"))
    host.admit("source", a)
    return host, proposer


def test_checked_chain_deterministic_release_and_primitive_snapshot():
    host, proposer = fixture()
    first = proposer.step(s.Derivation("derived_b", "ab", ("fact.source",)))
    assert first.accepted
    second = proposer.step(s.Derivation("derived_c", "bc", ("derived_b",)))
    assert second.accepted
    result = proposer.release(second.certificate)
    assert result == s.Release(True, "current support and policy checked", "c=true", s.Claim("c", True))
    assert proposer.release(second.certificate) == result
    encoded = json.loads(json.dumps(asdict(second)))
    assert encoded["snapshot"]["records"][0]["source"] == "rule"
    assert encoded["snapshot"]["rulebook"]["rules"][0]["premises"] == [{"key": "a", "value": True}]
    assert encoded["snapshot"]["release_keys"] == ["b", "c"]


@pytest.mark.parametrize("operation", ("replace_same", "replace_opposite", "replace_key", "revoke", "new_fact"))
def test_every_evidence_update_invalidates_all_transitive_derived_records(operation):
    host, proposer = fixture()
    b = proposer.step(s.Derivation("derived_b", "ab", ("fact.source",))).certificate
    c = proposer.step(s.Derivation("derived_c", "bc", ("derived_b",))).certificate
    old_fact = next(cert for cert in proposer.snapshot().certificates if cert.record.source == "evidence")
    previous_epoch = proposer.snapshot().evidence_epoch
    if operation == "replace_same":
        host.replace("source", s.Claim("a", True))
    elif operation == "replace_opposite":
        host.replace("source", s.Claim("a", False))
    elif operation == "replace_key":
        host.replace("source", s.Claim("c", True))
    elif operation == "revoke":
        host.revoke("source")
    else:
        host.admit("second", s.Claim("authority", True))
    snap = proposer.snapshot()
    assert snap.evidence_epoch == previous_epoch + 1
    assert all(record.source == "evidence" for record in snap.records)
    for certificate in (b, c, old_fact):
        assert not proposer.release(certificate).accepted
    assert not proposer.step(s.Derivation("again", "bc", ("derived_b",))).accepted


def test_policy_change_invalidates_release_without_erasing_support():
    host, proposer = fixture()
    cert = proposer.step(s.Derivation("b1", "ab", ("fact.source",))).certificate
    previous = proposer.snapshot()
    changed = host.set_release_keys(())
    assert changed.records == previous.records
    assert changed.evidence_epoch == previous.evidence_epoch
    assert changed.policy_epoch == previous.policy_epoch + 1
    assert not proposer.release(cert).accepted
    fresh = next(c for c in changed.certificates if c.record.record_id == "b1")
    assert proposer.release(fresh).reason == "release denied by host policy"
    host.set_release_keys(("b",))
    assert not proposer.release(fresh).accepted
    assert proposer.release(next(c for c in proposer.snapshot().certificates if c.record.record_id == "b1")).accepted


def test_conflict_policy_is_first_consistent_record_until_host_revision():
    host, proposer = fixture()
    assert proposer.step(s.Derivation("negative", "not_b", ("fact.source",))).accepted
    before = proposer.snapshot()
    rejection = proposer.step(s.Derivation("positive", "ab", ("fact.source",)))
    assert rejection.reason == "conflicting derivation rejected"
    assert rejection.snapshot == before
    with pytest.raises(ValueError, match="conflicting evidence"):
        host.admit("other", s.Claim("a", False))
    assert proposer.snapshot() == before
    # Host truth revisions supersede derived assertions; they are not blocked by them.
    host.admit("new", s.Claim("b", True))
    assert all(record.record_id != "negative" for record in proposer.snapshot().records)
    assert proposer.step(s.Derivation("positive", "ab", ("fact.source",))).accepted


@pytest.mark.parametrize("proposal,reason", (
    (s.Claim("b", True), "exact Derivation"),
    ({"authority": True}, "exact Derivation"),
    (s.Derivation("b1", "undeclared", ("fact.source",)), "undeclared rule"),
    (s.Derivation("b1", "ab", ("b1",)), "cyclic"),
    (s.Derivation("b1", "ab", ("missing",)), "missing"),
    (s.Derivation("b1", "ab", ()), "missing"),
    (s.Derivation("b1", "bc", ("fact.source",)), "do not match"),
    (s.Derivation("b1", "ab", ["fact.source"]), "immutable"),
    (s.Derivation("b1", "ab", (True,)), "identifier"),
    (s.Derivation("b1", "ab", ("fact.source", "fact.source")), "number"),
    (s.Derivation("fact.source", "ab", ("fact.source",)), "cyclic"),
))
def test_malformed_cyclic_and_unauthorized_derivations_are_atomic(proposal, reason):
    host, proposer = fixture()
    before = proposer.snapshot()
    feedback = proposer.step(proposal)
    assert not feedback.accepted and reason in feedback.reason
    assert feedback.snapshot == before == proposer.snapshot()


def test_untrusted_authority_word_never_grants_policy_and_no_empty_rule_mints_facts():
    book = s.Rulebook(("a", "authority"), (
        s.Rule("assert_authority", (s.Claim("a", True),), s.Claim("authority", True)),))
    host, proposer = s.create_supported_state(book)
    assert not hasattr(proposer, "admit") and not hasattr(proposer, "set_release_keys")
    assert not proposer.step(s.Derivation("authority1", "assert_authority", ("fact.source",))).accepted
    host.admit("source", s.Claim("a", True))
    result = proposer.step(s.Derivation("authority1", "assert_authority", ("fact.source",)))
    assert result.accepted
    assert result.snapshot.release_keys == ()
    assert not proposer.release(result.certificate).accepted
    with pytest.raises(ValueError, match="nonempty"):
        s.Rulebook(("a",), (s.Rule("unfounded", (), s.Claim("a", True)),))


def test_forged_or_foreign_certificates_cannot_release_unsupported_claims():
    host, proposer = fixture()
    valid = proposer.step(s.Derivation("b1", "ab", ("fact.source",))).certificate
    other_host, other = fixture()
    other.step(s.Derivation("b1", "ab", ("fact.source",)))
    assert not other.release(valid).accepted
    for malformed in (None, {}, replace(valid, evidence_epoch=True),
                      replace(valid, record=replace(valid.record, claim=s.Claim("b", 1))),
                      replace(valid, record=replace(valid.record, claim=s.Claim("b", False))),
                      replace(valid, record=replace(valid.record, record_id="missing")),
                      replace(valid, record=replace(valid.record, premises=("missing",))),
                      replace(valid, record="b1")):
        assert not proposer.release(malformed).accepted


def test_longest_admitted_evidence_identifier_remains_usable_as_premise():
    host, proposer = fixture()
    evidence_id = "x" * 64
    host.admit(evidence_id, s.Claim("a", True))
    result = proposer.step(s.Derivation("b1", "ab", ("fact." + evidence_id,)))
    assert result.accepted
    assert proposer.release(result.certificate).accepted


def test_declared_rule_cycles_require_an_existing_evidence_root():
    book = s.Rulebook(("a", "b"), (
        s.Rule("ab", (s.Claim("a", True),), s.Claim("b", True)),
        s.Rule("ba", (s.Claim("b", True),), s.Claim("a", True)),))
    host, proposer = s.create_supported_state(book, ("a", "b"))
    assert not proposer.step(s.Derivation("b1", "ab", ("a1",))).accepted
    assert not proposer.step(s.Derivation("a1", "ba", ("b1",))).accepted
    host.admit("root", s.Claim("a", True))
    assert proposer.step(s.Derivation("b1", "ab", ("fact.root",))).accepted
    assert proposer.step(s.Derivation("a1", "ba", ("b1",))).accepted
    host.revoke("root")
    assert proposer.snapshot().records == ()


def test_multistep_conjunction_requires_exact_ordered_typed_supports():
    book = s.Rulebook(("a", "b", "c"), (
        s.Rule("join", (s.Claim("a", True), s.Claim("b", False)), s.Claim("c", True)),))
    host, proposer = s.create_supported_state(book, ("c",))
    host.admit("a", s.Claim("a", True))
    host.admit("b", s.Claim("b", False))
    assert not proposer.step(s.Derivation("c1", "join", ("fact.b", "fact.a"))).accepted
    assert not proposer.step(s.Derivation("c1", "join", ("fact.a", "fact.a"))).accepted
    assert proposer.step(s.Derivation("c1", "join", ("fact.a", "fact.b"))).accepted


def test_finite_interpretation_soundness_all_3_key_single_premise_rules():
    """972 contexts, every satisfying Boolean world; independent truth semantics.

    This is bounded model checking, not a proof for arbitrary Python execution.
    Rules inconsistent with evidence may have zero models; record consistency
    remains checked and model-soundness is asserted only for nonempty models.
    """
    keys = ("a", "b", "c")
    claims = tuple(s.Claim(key, value) for key in keys for value in (False, True))
    contexts = checked_worlds = 0
    for premise, conclusion in product(claims, repeat=2):
        rule = s.Rule("r", (premise,), conclusion)
        book = s.Rulebook(keys, (rule,))
        for evidence_values in product((None, False, True), repeat=3):
            host, proposer = s.create_supported_state(book, keys)
            facts = dict((key, value) for key, value in zip(keys, evidence_values) if value is not None)
            for key, value in facts.items():
                host.admit(key, s.Claim(key, value))
            before = proposer.snapshot()
            proposal = s.Derivation("derived", "r", ("fact." + premise.key,))
            feedback = proposer.step(proposal)
            expected = (facts.get(premise.key) is premise.value
                        and (conclusion.key not in facts or facts[conclusion.key] is conclusion.value))
            assert feedback.accepted is expected
            if not expected:
                assert feedback.snapshot == before
            snapshot = proposer.snapshot()
            for world_values in product((False, True), repeat=3):
                world = dict(zip(keys, world_values))
                if (any(world[key] is not value for key, value in facts.items())
                        or (world[premise.key] is premise.value and world[conclusion.key] is not conclusion.value)):
                    continue
                checked_worlds += 1
                for certificate in snapshot.certificates:
                    released = proposer.release(certificate)
                    assert released.accepted
                    assert world[released.claim.key] is released.claim.value
            assert len({record.claim.key for record in snapshot.records}) == len(
                {record.claim for record in snapshot.records})
            contexts += 1
    assert contexts == 972
    assert checked_worlds > 0


@pytest.mark.parametrize("book", (
    lambda: s.Rulebook(("a", "a"), ()),
    lambda: s.Rulebook(["a"], ()),
    lambda: s.Rulebook(("a",), (s.Rule("r", (s.Claim("a", 1),), s.Claim("a", True)),)),
    lambda: s.Rulebook(("a",), (s.Rule("r", (s.Claim("a", True),), s.Claim("outside", True)),)),
    lambda: s.Rulebook(("a",), (s.Rule("r", (s.Claim("a", True), s.Claim("a", False)), s.Claim("a", True)),)),
))
def test_malformed_host_rulebook_rejects(book):
    with pytest.raises(ValueError):
        book()


def test_failed_host_updates_are_atomic():
    host, proposer = fixture()
    before = proposer.snapshot()
    for operation in (lambda: host.admit("bad id", s.Claim("a", True)),
                      lambda: host.admit("x", s.Claim("a", 1)),
                      lambda: host.admit("x", s.Claim("undeclared", True)),
                      lambda: host.revoke("unknown"),
                      lambda: host.set_release_keys(("a", "a")),
                      lambda: host.set_release_keys(("unknown",))):
        with pytest.raises(ValueError):
            operation()
        assert proposer.snapshot() == before
