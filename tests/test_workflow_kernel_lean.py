"""Independent finite-oracle and real Lean smoke checks for the new reference."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("workflow_lean_check", ROOT / "tools/check_workflow_kernel_lean.py")
check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check
SPEC.loader.exec_module(check)


def test_frozen_fixture_budget_and_declared_coverage():
    cases = check.freeze_cases()
    assert cases == check.freeze_cases()
    assert len(cases) == 4792
    assert len(cases) + len(check.freeze_assertions()) == 5744
    assert {case.prior for case in cases} == set(range(1, 256))
    assert {case.candidate for case in cases} == set(range(256))
    assert len(check.observations()) == 44
    assert all(0 < case.prior <= 255 for case in cases)


def test_original_python_matches_independent_finite_oracle():
    for case in check.freeze_cases():
        assert check.python_result(case) == check.independent(case), case


def test_truthful_contradictory_and_false_consistent_boundaries():
    # The hidden actual world is0. Authenticated receipt false is truthful.
    truthful = check.Case(3, 7, 2, 0, 2, 0, 0, 255)
    assert check.independent(truthful)[0] == 255  # broad accepted candidate can retain uncertainty
    truthful_exact = check.Case(3, 7, 2, 0, 2, 0, 0, 1)
    assert check.independent(truthful_exact)[0] == 1
    # A false but consistent receipt can delete actual0; inclusion does not authenticate observations.
    false_consistent = check.Case(3, 7, 2, 0, 2, 0, 1, 2)
    assert check.independent(false_consistent)[:5] == (2, 7, 2, 1, 0)
    # Contradiction after legal submit retains both branches and consumes attempt budget.
    contradiction = check.Case(1, 7, 2, 1, 4, 7, 2, 255)
    assert check.independent(contradiction)[:6] == (3, 7, 1, 0, 1, 1)
    # An illegal event does not condition memory or consume budget.
    illegal = check.Case(3, 7, 2, 1, 2, 0, 1, 255)
    assert check.independent(illegal)[:6] == (3, 7, 2, 0, 0, 0)


def test_grounded_assertions_guard_against_vacuity():
    for case in check.freeze_assertions():
        assert check.python_assertion(case) == check.independent_assertion(case)
    assert not any(check.independent_assertion((0, *obs)) for obs in check.observations())
    assert check.independent_assertion((22, 4, 7, 1))  # worlds1,2,4 exactly one
    assert not check.independent_assertion((150, 4, 7, 1))  # extra world7 breaks exact count
    assert check.independent_assertion((150, 3, 7, 1))


def test_real_lean_transition_and_assertion_smoke(tmp_path):
    executable = Path(check.lean_command()[0])
    source = ROOT / "proofs/certified_memory/WorkflowKernelCli.lean"
    if not executable.exists() or not source.exists():
        pytest.skip("portable Lean or reference CLI not installed")
    check.compile_current_sources(tmp_path)
    cases = check.freeze_cases()[:8]
    lines, _ = check.run_lean([case.line() for case in cases], search_path=tmp_path)
    assert [check.parse_output(line) for line in lines] == [check.python_result(case) for case in cases]
    lines, _ = check.run_lean(["ASSERT 0 4 7 1", "ASSERT 22 4 7 1", "ASSERT 150 4 7 1"], search_path=tmp_path)
    assert lines == ["ASSERT 0", "ASSERT 1", "ASSERT 0"]
