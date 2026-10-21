"""Bounded semantic boundaries and actual fresh Lean proof/execution checks."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("supported_state_lean_check", ROOT / "tools/check_supported_state_lean.py")
check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check
SPEC.loader.exec_module(check)


def test_fixed_fixtures_cover_declared_small_domains_without_duplicates():
    cases = check.freeze_cases()
    assert len(cases) == len(set(cases)) == 994
    assert cases == check.freeze_cases()
    assert sum(line.startswith("S ") for line in cases) == 972
    assert {tuple(map(int, line.split()[1:])) for line in cases if line.startswith("S ")} == {
        (p, c, e) for p in range(6) for c in range(6) for e in range(27)}
    assert not set(check.FORMAL_ONLY) & set(cases)
    with pytest.raises(ValueError, match="formal-only"):
        check.python_fixture("J 4")


def test_epoch_policy_and_conflict_boundaries_have_independent_expected_values():
    expected = {
        "R 0 7": (1, 2, 0, 1, 2, 2, 0),  # unchanged revision discards derived records
        "R 3 7": (1, 2, 0, 0, 0, 0, 0),  # revocation clears transitive support
        "R 5 0": (1, 1, 1, 3, 42, 0, 0),  # revoke all release permissions
        "R 5 4": (1, 1, 1, 3, 42, 32, 0),  # only c currently releasable
        "R 6 7": (1, 1, 1, 3, 42, 42, 0),  # unchanged policy still invalidates old cert
        "J 0": (1, 2, 0, 3, 38, 38, 0),
        "J 1": (0, 2, 0, 2, 6, 6, 0),  # reversed conjunction premises
        "J 8": (0, 3, 0, 3, 22, 22, 0),  # live negative conclusion rejects derivation
    }
    for line, want in expected.items():
        assert check.python_fixture(line)[0] == want


def test_support_consistency_does_not_establish_satisfying_world_existence():
    # Fact a=true and declared a->a=false have no common model. The checker
    # retains the fact and rejects the conflicting derived record.
    result, snapshot = check.python_fixture("S 1 0 2")
    assert result == (0, 1, 0, 1, 2, 2, 0)
    assert check.check_semantic_worlds(snapshot) == 0
    result, snapshot = check.python_fixture("S 1 3 2")
    assert result == check.independent_single("S 1 3 2")
    assert check.check_semantic_worlds(snapshot) == 2


def test_axiom_audit_fails_closed_for_custom_missing_and_duplicate_declarations():
    lines = [f"'SupportedState.{name}' does not depend on any axioms" for name in check.THEOREMS]
    assert len(check.audit_axioms("\n".join(lines))) == 23
    unsafe = lines.copy()
    unsafe[0] = f"'SupportedState.{check.THEOREMS[0]}' depends on axioms: [sorryAx]"
    for output in ("\n".join(unsafe), "\n".join(lines[:-1]), "\n".join(lines + [lines[0]])):
        with pytest.raises(RuntimeError):
            check.audit_axioms(output)


def test_actual_fresh_lean_proofs_axiom_audit_and_differential(tmp_path):
    if not check.lean_executable().is_file():
        pytest.skip("local Lean not installed; no proof claim from this test")
    receipt = check.check(tmp_path / "receipt.json")
    assert receipt["passed"]
    assert receipt["fresh_private_imports"]
    assert len(receipt["axioms"]) == 23
    assert receipt["fixtures"] == 994
    assert receipt["mismatches"] == []
    assert receipt["formal_only_payload_rejections"] == 2
    assert receipt["malformed_adapter_rejections"] == 18
    assert receipt["zero_model_contexts"] > 0
    assert receipt["satisfying_worlds_checked"] > 0
