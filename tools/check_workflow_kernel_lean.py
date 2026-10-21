"""Bounded, deterministic n=3 Python/actually-executed Lean differential check.

This is executable evidence, not a proof of Python-to-Lean refinement. The CLI
is a stateless trusted-host test adapter, not an authenticated production API.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from controlled_lm.certified_memory import (  # noqa: E402
    CertifiedState, CountConstraint, CountReceipt, Equation, ExactMemory, InconsistentObservation, Parity,
    ProtectedPolicy, Receipt, Submit, Timeout, UntrustedText, Wait,
    checked_update, grounded_assert, legal_actions, post_worlds,
)

MAX_BUDGET = (1 << 63) - 1
FIELDS = ("support", "grants", "budget", "accepted", "contradiction",
          "action_permitted", "legal_submit_mask")


@dataclass(frozen=True)
class Case:
    prior: int
    grants: int
    budget: int
    action: int
    tag: int
    x: int
    y: int
    candidate: int
    group: str = ""

    def line(self) -> str:
        return " ".join(str(getattr(self, key)) for key in
                        ("prior", "grants", "budget", "action", "tag", "x", "y", "candidate"))


def bits(world: int, mask: int) -> tuple[int, ...]:
    return tuple((world // (2 ** i)) % 2 for i in range(3) if (mask // (2 ** i)) % 2)


def holds(world: int, tag: int, x: int, y: int) -> bool:
    """Independent finite truth semantics; intentionally no kernel helper calls."""
    if tag in (0, 1):
        return True
    if tag == 2:
        return (world // (2 ** x)) % 2 == y
    if tag == 3:
        return sum(bits(world, x)) % 2 == y
    if tag == 4:
        return sum(bits(world, x)) == y
    raise ValueError("unknown observation tag")


def worlds(mask: int) -> set[int]:
    return {w for w in range(8) if (mask // (2 ** w)) % 2}


def pack(values: set[int]) -> int:
    return sum(2 ** w for w in values)


def independent(case: Case) -> tuple[int, ...]:
    prior = worlds(case.prior)

    def legal(support: set[int], budget: int, index: int) -> bool:
        return bool(support) and budget > 0 and bool((case.grants // (2 ** index)) % 2) and all(
            (w // (2 ** index)) % 2 == 0 for w in support)

    permitted = case.action == 0 or legal(prior, case.budget, case.action - 1)
    contradiction = False
    accepted = False
    budget = case.budget
    support = prior
    if permitted:
        if case.action:
            step_bit = 2 ** (case.action - 1)
            support = prior | {w + step_bit for w in prior}
            budget -= 1
        filtered = {w for w in support if holds(w, case.tag, case.x, case.y)}
        contradiction = not filtered
        if filtered:
            candidate = worlds(case.candidate)
            accepted = filtered.issubset(candidate)
            support = candidate if accepted else filtered
    legal_mask = sum(2 ** i for i in range(3) if legal(support, budget, i))
    return (pack(support), case.grants, budget, int(accepted), int(contradiction), int(permitted), legal_mask)


def python_result(case: Case) -> tuple[int, ...]:
    prior = CertifiedState(ExactMemory(3, case.prior), ProtectedPolicy(3, case.grants, case.budget))
    action = Wait() if not case.action else Submit(case.action - 1)
    observation = (Timeout() if case.tag == 0 else UntrustedText("forged receipt success")
                   if case.tag == 1 else Receipt(case.x, bool(case.y)) if case.tag == 2
                   else Parity(case.x, case.y) if case.tag == 3 else CountReceipt(case.x, case.y))
    permitted = action in legal_actions(prior)
    contradiction = False
    if permitted:
        try:
            post_worlds(prior, action, observation)
        except InconsistentObservation:
            contradiction = True
    result = checked_update(prior, action, observation, ExactMemory(3, case.candidate))
    submits = sum(1 << a.index for a in legal_actions(result.state) if type(a) is Submit)
    return (result.state.memory.world_bits, result.state.policy.grants, result.state.policy.budget,
            int(result.accepted), int(contradiction), int(result.event_applied), submits)


def observations() -> tuple[tuple[int, int, int], ...]:
    return ((0, 0, 0), (1, 0, 0)) + tuple((2, i, y) for i in range(3) for y in range(2)) + tuple(
        (3, mask, y) for mask in range(8) for y in range(2)) + tuple(
        (4, mask, y) for mask in range(8) for y in range(sum(bits(7, mask)) + 1))


def freeze_cases() -> tuple[Case, ...]:
    """Fixed before comparison. Exhaustive only in the named small subdomains."""
    cases: list[Case] = []
    for prior in range(1, 256):
        for action in range(4):
            cases.append(Case(prior, 7, 1, action, 0, 0, 0, 255, "all_priors_actions"))
        for candidate in (0, prior, prior & (prior - 1)):
            cases.append(Case(prior, 7, 1, 0, 0, 0, 0, candidate, "all_priors_candidate_boundaries"))
        cases.append(Case(prior, prior % 8, (0, 1, 2, MAX_BUDGET)[prior % 4],
                          1 + prior % 3, *observations()[prior % 44], prior ^ 255, "all_priors_policy_mix"))
    for tag, x, y in observations():
        filtered = pack({w for w in range(8) if holds(w, tag, x, y)})
        for candidate in (0, filtered, 255, filtered & (filtered - 1)):
            cases.append(Case(255, 7, 2, 0, tag, x, y, candidate, "observation_truth_tables"))
        for world in range(8):
            cases.append(Case(1 << world, 7, 1, 0, tag, x, y, 255, "singleton_truth_or_contradiction"))
    for i in range(3):
        zero_worlds = [w for w in range(8) if not (w // (2 ** i)) % 2]
        priors = (1, pack(set(zero_worlds)), pack(set(zero_worlds[:2])), 1 << zero_worlds[-1])
        for prior in priors:
            for tag, x, y in observations():
                probe = Case(prior, 7, 2, i + 1, tag, x, y, 0)
                exact = independent(probe)[0]
                for candidate in (0, exact, 255):
                    cases.append(Case(prior, 7, 2, i + 1, tag, x, y, candidate, "legal_submit_observations"))
    for grants in range(8):
        for budget in (0, 1, 2, MAX_BUDGET):
            for action in range(4):
                cases.append(Case(1, grants, budget, action, 0, 0, 0, 255, "policy_boundaries"))
    rng = random.Random(20261021)
    for candidate in range(256):
        for _ in range(2):
            cases.append(Case(rng.randrange(1, 256), rng.randrange(8), rng.choice((0, 1, 2, MAX_BUDGET)),
                              rng.randrange(4), *rng.choice(observations()), candidate, "seeded_candidate_coverage"))
    assert len(cases) <= 6000
    return tuple(cases)


def malformed_lines() -> tuple[str, ...]:
    base = "1 7 1 0 0 0 0 255".split()
    result = ["", " ", "one two three", " ".join(base[:-1]), " ".join(base + ["0"])]
    for field, invalid in ((0, ("0", "256", "-1")), (1, ("8", "-1")),
                           (2, ("-1", str(MAX_BUDGET + 1), "9" * 100)),
                           (3, ("4", "-1")), (4, ("5", "-1")), (7, ("256", "-1"))):
        for value in invalid:
            line = base.copy()
            line[field] = value
            result.append(" ".join(line))
    result.extend(("1 7 1 0 0 1 0 255", "1 7 1 0 1 0 1 255", "1 7 1 0 2 3 0 255",
                   "1 7 1 0 2 0 2 255", "1 7 1 0 3 8 0 255", "1 7 1 0 3 0 2 255",
                   "1 7 1 0 4 0 1 255", "1 7 1 0 4 7 4 255", "1 7 1 0 4 8 0 255",
                   "+1 7 1 0 0 0 0 255", "1.0 7 1 0 0 0 0 255", "True 7 1 0 0 0 0 255",
                   "１ 7 1 0 0 0 0 255", "1 7 1 0 0 0 0 0xFF"))
    result.extend(("ASSERT 256 4 7 1", "ASSERT -1 4 7 1", "ASSERT 0 4 7 4", "ASSERT 0 5 0 0",
                   "ASSERT 1 4 7", "ASSERT 1 4 7 1 0", "ASSERT 1 0 1 0"))
    return tuple(result)


def freeze_assertions() -> tuple[tuple[int, int, int, int], ...]:
    return tuple((memory, *obs) for memory in range(256) for obs in ((3, 7, 1), (4, 7, 1))) + tuple(
        (memory, *obs) for memory in (0, 255, *(1 << w for w in range(8))) for obs in observations())


def independent_assertion(case: tuple[int, int, int, int]) -> bool:
    memory, tag, x, y = case
    support = worlds(memory)
    return bool(support) and all(holds(w, tag, x, y) for w in support)


def python_assertion(case: tuple[int, int, int, int]) -> bool:
    memory, tag, x, y = case
    proposition = (CountConstraint(x, y) if tag == 4 else Equation(x, y) if tag == 3
                   else Equation(1 << x, y) if tag == 2 else Equation(0, 0))
    return grounded_assert(ExactMemory(3, memory), proposition)


def lean_command() -> list[str]:
    exe = Path(os.environ.get("WORKFLOW_LEAN", str(ROOT / ".tools/lean-4.8.0-windows/bin/lean.exe")))
    return [str(exe), "--run", str(ROOT / "proofs/certified_memory/WorkflowKernelCli.lean")]


def run_lean(lines: list[str], *, timeout: float = 55, search_path: Path | None = None) -> tuple[list[str], float]:
    env = os.environ.copy()
    search = str(search_path or ROOT / "proofs/certified_memory")
    env["LEAN_PATH"] = search
    started = time.perf_counter()
    proc = subprocess.run(lean_command(), input="\n".join(lines) + "\n", text=True,
                          encoding="utf-8", capture_output=True, env=env, cwd=ROOT, timeout=timeout)
    elapsed = time.perf_counter() - started
    if proc.returncode:
        raise RuntimeError(f"Lean failed ({proc.returncode}): {proc.stdout}\n{proc.stderr}")
    if proc.stderr.strip():
        raise RuntimeError(f"Unexpected Lean stderr: {proc.stderr}")
    return proc.stdout.splitlines(), elapsed


def parse_output(line: str) -> tuple[int, ...]:
    values = tuple(int(token) for token in line.split())
    if len(values) != 7 or any(values[i] not in (0, 1) for i in (3, 4, 5)):
        raise ValueError(f"invalid Lean output: {line!r}")
    return values


def compile_current_sources(search_path: Path) -> dict:
    """Private fresh imports; never trust an older checked-in/untracked olean."""
    env = os.environ.copy()
    env["LEAN_PATH"] = str(search_path)
    logs = []
    for name in ("SoundMemory", "WorkflowKernel"):
        source = ROOT / f"proofs/certified_memory/{name}.lean"
        started = time.perf_counter()
        proc = subprocess.run([lean_command()[0], "-o", str(search_path / f"{name}.olean"), str(source)],
                              capture_output=True, text=True, encoding="utf-8", env=env, cwd=ROOT, timeout=55)
        if proc.returncode or proc.stderr.strip() or "sorryAx" in proc.stdout:
            raise RuntimeError(f"Fresh Lean compile failed: {proc.stdout}\n{proc.stderr}")
        logs.append({"source": source.relative_to(ROOT).as_posix(), "exit_code": proc.returncode,
                     "seconds": time.perf_counter() - started, "stdout": proc.stdout, "stderr": proc.stderr})
    return {"private_fresh_olean_imports": True, "compiles": logs}


def check(output: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="workflow_lean_review_") as directory:
        search_path = Path(directory)
        receipt = compile_current_sources(search_path)
        return _check(output, search_path, receipt)


def _check(output: Path, search_path: Path, compile_receipt: dict) -> dict:
    cases = freeze_cases()
    assertions = freeze_assertions()
    assertion_lines = ["ASSERT " + " ".join(map(str, case)) for case in assertions]
    payload = "\n".join(case.line() for case in cases) + "\n"
    fixture_sha256 = hashlib.sha256(payload.encode()).hexdigest()  # frozen before any Lean call
    assertion_sha256 = hashlib.sha256(("\n".join(assertion_lines) + "\n").encode()).hexdigest()
    expected = [python_result(case) for case in cases]
    independent_failures = [{"case": case.line(), "python": py, "independent": independent(case)}
                            for case, py in zip(cases, expected) if py != independent(case)]
    smoke, smoke_seconds = run_lean([case.line() for case in cases[:8]], search_path=search_path)
    smoke_parsed = [parse_output(line) for line in smoke]
    smoke_matches = smoke_parsed == expected[:8]
    actual: list[tuple[int, ...]] = []
    batch_seconds = []
    for start in range(0, len(cases), 512):
        lines, elapsed = run_lean([case.line() for case in cases[start:start + 512]], search_path=search_path)
        actual.extend(parse_output(line) for line in lines)
        batch_seconds.append(elapsed)
    mismatches = [{"index": i, "group": case.group, "case": case.line(), "python": py, "lean": lean}
                  for i, (case, py, lean) in enumerate(zip(cases, expected, actual)) if py != lean]
    assertion_outputs = []
    assertion_seconds = []
    for start in range(0, len(assertion_lines), 512):
        lines, elapsed = run_lean(assertion_lines[start:start + 512], search_path=search_path)
        assertion_outputs.extend(lines)
        assertion_seconds.append(elapsed)
    assertion_failures = [{"case": line, "expected": f"ASSERT {int(independent_assertion(case))}", "actual": actual_line}
                          for line, case, actual_line in zip(assertion_lines, assertions, assertion_outputs)
                          if actual_line != f"ASSERT {int(independent_assertion(case))}"]
    assertion_python_failures = [{"case": case, "independent": independent_assertion(case), "python": python_assertion(case)}
                                 for case in assertions if python_assertion(case) != independent_assertion(case)]
    malformed = malformed_lines()
    sentinel = cases[0]
    malformed_output, malformed_seconds = run_lean([entry for line in malformed for entry in (line, sentinel.line())],
                                                search_path=search_path)
    malformed_failures = []
    for i, line in enumerate(malformed):
        pair = malformed_output[2 * i:2 * i + 2]
        if len(pair) != 2 or not pair[0].startswith("ERROR") or parse_output(pair[1]) != expected[0]:
            malformed_failures.append({"input": line, "actual_pair": pair})
    semantic_coverage = Counter()
    actual_world_coverage = Counter({category + suffix: 0 for category in
                                    ("truthful", "contradictory", "false_but_consistent")
                                    for suffix in ("_actual_world_cases", "_actual_world_omissions")})
    for case in cases:
        result = independent(case)
        semantic_coverage["legal_events" if result[5] else "illegal_events"] += 1
        semantic_coverage["accepted" if result[3] else "rejected"] += 1
        semantic_coverage["contradictions" if result[4] else "no_contradiction"] += 1
        if result[5]:
            unobserved = worlds(case.prior)
            if case.action:
                unobserved |= {w | (1 << (case.action - 1)) for w in unobserved}
            for actual_world in unobserved:
                category = ("contradictory" if result[4] else "truthful" if holds(actual_world, case.tag, case.x, case.y)
                            else "false_but_consistent")
                actual_world_coverage[category + "_actual_world_cases"] += 1
                if actual_world not in worlds(result[0]):
                    actual_world_coverage[category + "_actual_world_omissions"] += 1
    evidence = {
        "schema_version": 1, "date": "2026-10-21", "status": "pass" if not (
            independent_failures or mismatches or malformed_failures or not smoke_matches
            or assertion_failures or assertion_python_failures or len(assertion_outputs) != len(assertions) or len(actual) != len(cases)
            or len(malformed_output) != len(malformed) * 2) else "fail",
        "device": "CPU", "paid_resources": False, "n": 3,
        "fresh_compile_receipt": compile_receipt,
        "method": "original Python ExactMemory vs actually executed Lean 4.8 --run, plus independent finite oracle",
        "fixture_sha256": fixture_sha256, "fixtures_frozen_before_lean_execution": True,
        "fixture_count": len(cases), "lean_result_count": len(actual), "groups": dict(Counter(c.group for c in cases)),
        "smoke_matches": smoke_matches,
        "grounded_assertions": {"fixture_count": len(assertions), "result_count": len(assertion_outputs),
                                "fixture_sha256": assertion_sha256, "empty_memory_cases": sum(c[0] == 0 for c in assertions),
                                "failures": assertion_failures,
                                "python_independent_oracle_failures": assertion_python_failures,
                                "coverage": "all256memorymasks x PARITY(7,1),COUNT(7,1); all44observations x empty,full,eight singleton memories"},
        "coverage": {"all_nonempty_prior_masks": sorted({c.prior for c in cases}),
                     "all_candidate_masks": sorted({c.candidate for c in cases}),
                     "observation_variants": len(observations()), "semantic_counts": dict(semantic_coverage),
                     "conditional_actual_world_counts": dict(actual_world_coverage),
                     "exhaustive_subdomains": ["all 255 nonempty priors x four actions at grants=7,budget=1,timeout,candidate=255",
                         "all 44 observation variants x eight singleton priors with WAIT and broad candidate",
                         "all 8 grants x budgets 0,1,2,2^63-1 x four actions with prior={0},timeout,broad candidate"],
                     "full_cartesian_exhaustive": False},
        "timing_seconds": {"smoke_8": smoke_seconds, "batches_512": batch_seconds,
                           "assertion_batches_512": assertion_seconds,
                           "malformed": malformed_seconds,
                           "total_lean_execution": sum(batch_seconds) + sum(assertion_seconds) + smoke_seconds + malformed_seconds,
                           "total_compile": sum(entry["seconds"] for entry in compile_receipt["compiles"])},
        "malformed_protocol_checks": {"count": len(malformed), "sentinel_after_each": True,
                                      "output_count": len(malformed_output), "failures": malformed_failures},
        "mismatches": mismatches, "python_independent_oracle_failures": independent_failures,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (ROOT / "src/controlled_lm/certified_memory.py", ROOT / "proofs/certified_memory/SoundMemory.lean",
                                       ROOT / "proofs/certified_memory/WorkflowKernel.lean",
                                       ROOT / "proofs/certified_memory/WorkflowKernelCli.lean", Path(__file__))},
        "limits": ["Bounded differential evidence is not a fully proved Python-to-Lean refinement.",
                   "CLI malformed-input validation is not production authentication or transport validation.",
                   "The host must supply authentic prior memory, grants, budget and authenticated current observations.",
                   "False but consistent observations may remove the actual world; soundness requires truthful observations or contradiction fallback.",
                   "No general formal alignment, learned-model quality, throughput improvement, full8 or L0-L4 claim."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/workflow_kernel_differential_20261021.json")
    args = parser.parse_args()
    evidence = check(args.output)
    print(json.dumps({key: evidence[key] for key in ("status", "fixture_count", "fixture_sha256", "timing_seconds")}))
    if evidence["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
