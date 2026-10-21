import json

from controlled_lm.supported_state import Claim, Rule, Rulebook, create_supported_state
from controlled_lm.semantic_interface_comparison import ComparisonEpisode


def episode(budget=8):
    rules = Rulebook(("signed", "budget_ok", "may_execute"), (
        Rule("allow", (Claim("signed", True), Claim("budget_ok", True)), Claim("may_execute", True)),))
    host, proposer = create_supported_state(rules, ("may_execute",))
    host.admit("signature", Claim("signed", True))
    host.admit("budget", Claim("budget_ok", True))
    return host, ComparisonEpisode(proposer, "May this request execute?", budget)


DERIVE = {"op": "derive", "record_id": "decision", "rule_id": "allow", "premises": ["fact.signature", "fact.budget"]}


def release(e):
    s = e.structured_observation()["state"]
    return {"op": "release", "record_id": "decision", "evidence_epoch": s["evidence_epoch"], "policy_epoch": s["policy_epoch"]}


def test_every_step_has_identical_information_and_feedback():
    _, internal = episode()
    _, external = episode()
    assert internal.structured_observation() == json.loads(external.external_observation())
    for action in ({"op": "derive", "record_id": "bad", "rule_id": "missing", "premises": []}, DERIVE, release(internal)):
        left = internal.attempt(action)
        right = external.attempt(json.dumps(action))
        assert left == right == json.loads(external.external_observation())
    assert left["feedback"]["released_text"] == "may_execute=true"


def test_host_revision_invalidates_previous_derivation_and_release():
    host, e = episode()
    e.attempt(DERIVE)
    old = release(e)
    host.replace("budget", Claim("budget_ok", False))
    assert not e.attempt(old)["feedback"]["accepted"]
    assert not e.attempt(DERIVE)["feedback"]["accepted"]
    assert all(r["record_id"] != "decision" for r in e.structured_observation()["state"]["records"])


def test_policy_change_invalidates_release_and_blocks_current_release():
    host, e = episode()
    e.attempt(DERIVE)
    old = release(e)
    host.set_release_keys(())
    assert e.attempt(old)["feedback"]["reason"] == "stale release request"
    assert e.attempt(release(e))["feedback"]["reason"] == "release denied by host policy"


def test_malformed_and_forbidden_operations_consume_budget():
    _, e = episode(3)
    original = e.structured_observation()["state"]
    for proposal in ('{"op":"abstain","op":"release"}', {"op": "admit", "claim": "trusted"}, {"op": "set_release_keys", "keys": ["signed"]}):
        assert not e.attempt(proposal)["feedback"]["accepted"]
        assert e.structured_observation()["state"] == original
    assert e.attempt(DERIVE)["feedback"]["reason"] == "operation budget exhausted"
    assert e.attempts == 3


def test_observation_is_detached_and_has_no_hidden_answer_or_session_nonce():
    _, e = episode()
    view = e.structured_observation()
    view["state"]["release_keys"].append("signed")
    view["grammar"]["admit"] = []
    assert e.structured_observation()["state"]["release_keys"] == ["may_execute"]
    assert "admit" not in e.structured_observation()["grammar"]
    assert "kernel_id" not in e.structured_observation()["state"]
    assert "certificates" not in e.structured_observation()["state"]
    assert "answer" not in e.structured_observation()


def test_wrong_release_epoch_types_and_extra_fields_rejected():
    _, e = episode()
    e.attempt(DERIVE)
    r = release(e)
    r["policy_epoch"] = False
    assert not e.attempt(r)["feedback"]["accepted"]
    assert not e.attempt({**release(e), "override": True})["feedback"]["accepted"]


def test_nonstandard_json_numbers_rejected_at_parser():
    _, e = episode()
    for value in ("NaN", "Infinity", "-Infinity"):
        result = e.attempt('{"op":"release","record_id":"decision","evidence_epoch":'+value+',"policy_epoch":0}')
        assert result["feedback"]["reason"] == "invalid JSON proposal"
