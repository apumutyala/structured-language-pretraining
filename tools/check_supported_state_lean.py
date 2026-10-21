"""Fresh Lean proof/axiom audit and bounded actual-Lean/Python comparison.

The comparison is executable evidence for selected projections, NOT a refinement
proof or a universal runtime, isolation, or external-truth guarantee. CPU only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PROOFS = ROOT / "proofs/supported_state"
sys.path.insert(0, str(ROOT / "src"))
from controlled_lm import supported_state as s  # noqa: E402

THEOREMS = (
    "initial_invariant", "initial_consistent", "revise_invariant", "revision_only_evidence",
    "revise_consistent", "stepOK_sound", "checked_step_invariant",
    "checked_step_consistent", "accepted_premises_preexist", "rejected_no_mutation",
    "proposer_preserves_authority", "policy_preserves_invariant", "evolution_invariant",
    "evolution_consistent", "release_checks_current", "release_supported",
    "released_claim_registered", "old_evidence_certificate_rejected",
    "old_policy_certificate_rejected", "support_has_evidence_root", "supported_semantically_sound",
    "release_semantically_sound", "support_does_not_imply_external_truth",
)
STANDARD_AXIOMS = {"propext", "Quot.sound", "Classical.choice"}
FIELDS = ("accepted", "evidence_epoch", "policy_epoch", "record_count", "claim_mask",
          "current_release_mask", "old_certificate_accepted")
FORMAL_ONLY = ("J 4", "J 6")


def lean_executable() -> Path:
    return Path(os.environ.get("SUPPORTED_STATE_LEAN", str(ROOT / ".tools/lean-4.8.0-windows/bin/lean.exe")))


def freeze_cases() -> tuple[str, ...]:
    """994 distinct fixtures, fixed before invoking Lean; no adaptive selection."""
    singles = tuple(f"S {p} {c} {e}" for p, c, e in product(range(6), range(6), range(27)))
    revisions = tuple(f"R {op} 7" for op in range(5)) + tuple(f"R 5 {mask}" for mask in range(8)) + ("R 6 7",)
    conjunctions = tuple(f"J {mode}" for mode in range(10) if mode not in (4, 6))
    return singles + revisions + conjunctions


def malformed_cases() -> tuple[str, ...]:
    return ("", " ", "S", "S 0 0", "S 0 0 0 0", "S -1 0 0", "S 6 0 0", "S 0 6 0",
            "S 0 0 27", "S 0 0 0.0", "S ０ 0 0", "R 7 0", "R 0 8", "R true 0",
            "J 10", "J -1", "ASSERT 1", "S " + "9" * 129 + " 0 0")


def claim(index: int) -> s.Claim:
    return s.Claim(f"k{index // 2}", bool(index % 2))


def summary(proposer, accepted: bool, old_certificate=None) -> tuple[int, ...]:
    snap = proposer.snapshot()
    mask = release_mask = 0
    for cert in snap.certificates:
        bit = 1 << (int(cert.record.claim.key[1:]) * 2 + int(cert.record.claim.value))
        mask |= bit
        if proposer.release(cert).accepted:
            release_mask |= bit
    return (int(accepted), snap.evidence_epoch, snap.policy_epoch, len(snap.records), mask,
            release_mask, int(old_certificate is not None and proposer.release(old_certificate).accepted))


def python_fixture(line: str):
    tokens = line.split()
    tag, values = tokens[0], tuple(map(int, tokens[1:]))
    keys = ("k0", "k1", "k2")
    old = None
    if tag == "S":
        p, c, evidence = values
        host, proposer = s.create_supported_state(s.Rulebook(keys, (s.Rule("r", (claim(p),), claim(c)),)), keys)
        for key in range(3):
            value = evidence // 3 ** key % 3
            if value:
                host.admit(f"f{key}", s.Claim(keys[key], value == 2))
        accepted = proposer.step(s.Derivation("derived", "r", (f"fact.f{p // 2}",))).accepted
    elif tag == "R":
        op, policy = values
        a, b, c = (s.Claim(key, True) for key in keys)
        book = s.Rulebook(keys, (s.Rule("ab", (a,), b), s.Rule("bc", (b,), c)))
        host, proposer = s.create_supported_state(book, keys)
        host.admit("f0", a)
        assert proposer.step(s.Derivation("derived_b", "ab", ("fact.f0",))).accepted
        result = proposer.step(s.Derivation("derived_c", "bc", ("derived_b",)))
        assert result.accepted
        old = result.certificate
        if op == 0:
            host.replace("f0", a)
        elif op == 1:
            host.replace("f0", s.Claim("k0", False))
        elif op == 2:
            host.replace("f0", c)
        elif op == 3:
            host.revoke("f0")
        elif op == 4:
            host.admit("f1", b)
        elif op == 5:
            host.set_release_keys(tuple(key for i, key in enumerate(keys) if policy & (1 << i)))
        elif op == 6:
            host.set_release_keys(keys)
        else:
            raise ValueError("unknown revision fixture")
        accepted = True
    elif tag == "J":
        mode, = values
        if mode in (4, 6):
            raise ValueError("formal-only resolved-payload perturbation has no Python proposal equivalent")
        a, b, c = s.Claim("k0", True), s.Claim("k1", False), s.Claim("k2", True)
        host, proposer = s.create_supported_state(s.Rulebook(keys, (s.Rule("join", (a, b), c),)), keys)
        host.admit("f0", a)
        host.admit("f1", b)
        if mode == 8:
            host.admit("f2", s.Claim("k2", False))
        premises = {1: ("fact.f1", "fact.f0"), 2: ("fact.f0", "fact.f0"), 3: ("fact.f0",),
                    9: ("fact.missing", "fact.f1")}.get(mode, ("fact.f0", "fact.f1"))
        record_id = "fact.f0" if mode == 5 else "derived"
        rule_id = "unknown" if mode == 7 else "join"
        accepted = proposer.step(s.Derivation(record_id, rule_id, premises)).accepted
    else:
        raise ValueError("unknown fixture")
    return summary(proposer, accepted, old), proposer.snapshot()


def independent_single(line: str) -> tuple[int, ...]:
    """Independent expected behavior for every one-premise, three-key fixture."""
    _, p, c, evidence = line.split()
    p, c, evidence = int(p), int(c), int(evidence)
    facts = {key: value == 2 for key in range(3) if (value := evidence // 3 ** key % 3)}
    accepted = (p // 2 in facts and facts[p // 2] == bool(p % 2)
                and (c // 2 not in facts or facts[c // 2] == bool(c % 2)))
    mask = sum(1 << (2 * key + int(value)) for key, value in facts.items())
    if accepted:
        mask |= 1 << c
    return (int(accepted), len(facts), 0, len(facts) + int(accepted), mask, mask, 0)


def check_semantic_worlds(snapshot) -> int:
    """Independent truth table, conditional on this world's facts AND all rules.

    Zero models is recorded, never used to claim model existence or true evidence.
    Evidence records are the roots because snapshots omit the private evidence map.
    """
    count = 0
    for values in product((False, True), repeat=3):
        def holds(c):
            return values[int(c.key[1:])] == c.value
        facts = (record.claim for record in snapshot.records if record.source == "evidence")
        if not all(holds(c) for c in facts):
            continue
        if not all(not all(holds(c) for c in rule.premises) or holds(rule.conclusion)
                   for rule in snapshot.rulebook.rules):
            continue
        count += 1
        if not all(holds(record.claim) for record in snapshot.records):
            raise AssertionError("record is false in an enumerated satisfying world")
    return count


def audit_axioms(stdout: str) -> dict[str, list[str]]:
    parsed = {}
    for line in stdout.splitlines():
        match = re.fullmatch(r"'SupportedState\.([^']+)' (?:depends on axioms: \[([^]]*)\]|(does not depend on any axioms))", line)
        if match is None:
            raise RuntimeError(f"unexpected Lean audit output: {line!r}")
        name, axioms, _ = match.groups()
        if name in parsed:
            raise RuntimeError(f"duplicate axiom audit: {name}")
        used = [] if axioms is None else [part.strip() for part in axioms.split(",") if part.strip()]
        if not set(used).issubset(STANDARD_AXIOMS):
            raise RuntimeError(f"nonstandard axiom in {name}: {used}")
        parsed[name] = used
    if set(parsed) != set(THEOREMS):
        raise RuntimeError("missing or unexpected theorem in axiom audit")
    return parsed


def compile_current_sources(directory: Path) -> dict:
    exe = lean_executable()
    env = os.environ.copy()
    env["LEAN_PATH"] = str(directory)
    version = subprocess.run([str(exe), "--version"], capture_output=True, text=True,
                             encoding="utf-8", timeout=55, check=True).stdout.strip()
    logs = []
    axioms = None
    for name in ("SupportedState", "SupportedStateCli"):
        source = PROOFS / f"{name}.lean"
        # The compiler's axiom audit is authoritative; this catches common source shortcuts too.
        code = re.sub(r"/-.*?-/|--[^\n]*", "", source.read_text(encoding="utf-8"), flags=re.S)
        if re.search(r"\b(sorry|admit|axiom|unsafe|native_decide)\b", code):
            raise RuntimeError(f"proof shortcut in {source.name}")
        started = time.perf_counter()
        result = subprocess.run([str(exe), "-o", str(directory / f"{name}.olean"), str(source)],
                                capture_output=True, text=True, encoding="utf-8", env=env, cwd=ROOT, timeout=55)
        if result.returncode or result.stderr.strip() or "warning:" in result.stdout:
            raise RuntimeError(f"fresh Lean compile failed: {result.stdout}\n{result.stderr}")
        if name == "SupportedState":
            axioms = audit_axioms(result.stdout)
        elif result.stdout.strip():
            raise RuntimeError(f"unexpected CLI compile output: {result.stdout}")
        logs.append({"source": source.relative_to(ROOT).as_posix(), "exit_code": result.returncode,
                     "seconds": time.perf_counter() - started})
    return {"lean_version": version, "fresh_private_imports": True, "compiles": logs, "axioms": axioms}


def run_lean(lines: tuple[str, ...], directory: Path) -> tuple[list[str], float]:
    env = os.environ.copy()
    env["LEAN_PATH"] = str(directory)
    started = time.perf_counter()
    result = subprocess.run([str(lean_executable()), "--run", str(PROOFS / "SupportedStateCli.lean")],
                            input="\n".join(lines) + "\n", capture_output=True, text=True, encoding="utf-8",
                            cwd=ROOT, env=env, timeout=55)
    if result.returncode or result.stderr.strip():
        raise RuntimeError(f"Lean execution failed: {result.stdout}\n{result.stderr}")
    output = result.stdout.splitlines()
    if len(output) != len(lines):
        raise RuntimeError("Lean output cardinality mismatch")
    return output, time.perf_counter() - started


def check(output: Path | None = None) -> dict:
    cases = freeze_cases()
    payload = "\n".join(cases) + "\n"
    fixture_hash = hashlib.sha256(payload.encode()).hexdigest()
    expected, zero_model_contexts, worlds = [], 0, 0
    for line in cases:
        values, snapshot = python_fixture(line)
        expected.append(values)
        if line.startswith("S ") and values != independent_single(line):
            raise AssertionError(f"Python differs from independent finite oracle: {line}")
        count = check_semantic_worlds(snapshot)
        worlds += count
        zero_model_contexts += int(count == 0)
    bad = malformed_cases()
    with tempfile.TemporaryDirectory(prefix="supported_state_lean_") as directory:
        directory = Path(directory)
        receipt = compile_current_sources(directory)
        actual, seconds = run_lean(cases + FORMAL_ONLY + bad, directory)
    mismatches = []
    for line, want, got in zip(cases, expected, actual):
        try:
            parsed = tuple(map(int, got.split()))
        except ValueError:
            parsed = ()
        if parsed != want:
            mismatches.append({"case": line, "python": want, "lean": got})
    formal = actual[len(cases):len(cases) + len(FORMAL_ONLY)]
    if formal != ["0 2 0 2 6 6 0"] * 2:
        raise AssertionError(f"formal-only forged payload test failed: {formal}")
    malformed = actual[len(cases) + len(FORMAL_ONLY):]
    if not all(line.startswith("ERROR ") for line in malformed):
        raise AssertionError("malformed test adapter fixture accepted")
    if mismatches:
        raise AssertionError(f"Python/Lean mismatch: {mismatches[:5]}")
    files = (PROOFS / "SupportedState.lean", PROOFS / "SupportedStateCli.lean",
             ROOT / "src/controlled_lm/supported_state.py", Path(__file__))
    receipt.update({
        "passed": True, "scope": "abstract supported-record model; bounded projection comparison, not refinement",
        "fields": FIELDS, "fixtures": len(cases), "groups": dict(Counter(line[0] for line in cases)),
        "fixture_sha256": fixture_hash, "mismatches": mismatches,
        "formal_only_payload_rejections": len(FORMAL_ONLY), "malformed_adapter_rejections": len(bad),
        "independent_single_rule_oracle_contexts": 972, "satisfying_worlds_checked": worlds,
        "zero_model_contexts": zero_model_contexts,
        "lean_execution_seconds": seconds,
        "sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "limitations": ["No Python-to-Lean refinement theorem or all-input runtime proof.",
                        "Comparison projects acceptance, epochs, record count, claim/release masks and stale release.",
                        "No proof of host authentication, channel isolation, model existence or external truth.",
                        "No neural training, learned RECORD mechanism, behavioral alignment or grant qualification claim."],
    })
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROOFS / "check_receipt.json")
    args = parser.parse_args()
    receipt = check(args.output)
    print(json.dumps({"passed": receipt["passed"], "theorems": len(receipt["axioms"]),
                      "fixtures": receipt["fixtures"], "receipt": str(args.output)}))


if __name__ == "__main__":
    main()
