"""CPU-only deterministic demonstration; no neural training or general safety proof."""
import json
from controlled_lm.supported_state import Claim, Rule, Rulebook, create_supported_state
from controlled_lm.semantic_interface_comparison import ComparisonEpisode

rules = Rulebook(("signed", "budget_ok", "may_execute"), (
    Rule("allow", (Claim("signed", True), Claim("budget_ok", True)), Claim("may_execute", True)),))
host, proposer = create_supported_state(rules, ("may_execute",))
host.admit("signature", Claim("signed", True))
host.admit("budget", Claim("budget_ok", True))
episode = ComparisonEpisode(proposer, "May this request execute?", 6)
assert episode.structured_observation() == json.loads(episode.external_observation())
episode.attempt({"op": "derive", "record_id": "decision", "rule_id": "allow", "premises": ["fact.signature", "fact.budget"]})
state = episode.structured_observation()["state"]
release = {"op": "release", "record_id": "decision", "evidence_epoch": state["evidence_epoch"], "policy_epoch": state["policy_epoch"]}
before = episode.attempt(release)["feedback"]
host.replace("budget", Claim("budget_ok", False))
after = episode.attempt(release)["feedback"]
assert before["released_text"] == "may_execute=true" and not after["accepted"]
assert episode.structured_observation() == json.loads(episode.external_observation())
print(json.dumps({"same_information_roundtrip": True, "before_revision": before,
    "after_revision": after, "attempts": episode.attempts,
    "scope": "Exact Python interface demonstration; neither a Lean theorem nor a trained external-control result"}, indent=2))
