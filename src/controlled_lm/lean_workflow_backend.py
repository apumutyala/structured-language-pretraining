"""Opt-in, modeled n=3 workflow state owned by a persistent actual Lean process.

The trusted host supplies the initial certified state, actions and authenticated
current observations. A learned component may supply only an ExactMemory
candidate to ``step``. This Python object is an ownership convention within one
process, not a privilege boundary against arbitrary Python code. It performs no
real tool calls or physical side effects and supplies no production transport or
observation authentication. False but consistent observations remain outside the
Lean survival theorem. The original Python checker is unchanged.

Every session snapshots current sources, freshly builds private imports, records
source/compiler hashes and runs that snapshot. Backend failure is terminal;
there is no retry after uncertain IPC. Pre-IPC argument errors do not advance
modeled state. Use a context manager, or explicitly call close().
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass

from .certified_memory import (
    Action, CertifiedState, CountConstraint, CountReceipt, Equation, ExactMemory,
    Observation, Parity, ProtectedPolicy, Receipt, Submit, Timeout, UntrustedText,
    Wait, legal_actions,
)

MAX_BUDGET = (1 << 63) - 1
SOURCE_NAMES = ("SoundMemory.lean", "WorkflowKernel.lean", "WorkflowKernelCli.lean")


class LeanBackendError(RuntimeError):
    """The backend is unavailable, failed, or violated its response contract."""


@dataclass(frozen=True, slots=True)
class LeanStepResult:
    state: CertifiedState
    accepted: bool
    contradiction: bool
    action_permitted: bool
    legal_submit_mask: int

    @property
    def event_applied(self) -> bool:
        """The modeled event applied; no external event is executed here."""
        return self.action_permitted


def _integer(value: int, name: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def _exact(memory: ExactMemory, *, nonempty: bool = False) -> ExactMemory:
    if type(memory) is not ExactMemory:
        raise ValueError("the Lean backend requires ExactMemory")
    _integer(memory.n, "n", 3, 3)
    _integer(memory.world_bits, "world_bits", int(nonempty), 255)
    return ExactMemory(3, memory.world_bits)


def _initial(state: CertifiedState) -> CertifiedState:
    if type(state) is not CertifiedState or type(state.policy) is not ProtectedPolicy:
        raise ValueError("initial state and policy must be certified host values")
    memory = _exact(state.memory, nonempty=True)
    _integer(state.policy.n, "policy n", 3, 3)
    grants = _integer(state.policy.grants, "grants", 0, 7)
    budget = _integer(state.policy.budget, "budget", 0, MAX_BUDGET)
    return CertifiedState(memory, ProtectedPolicy(3, grants, budget))


def _action_code(action: Action) -> int:
    if type(action) is Wait:
        return 0
    if type(action) is Submit:
        return 1 + _integer(action.index, "submission index", 0, 2)
    raise ValueError("unsupported action")


def _observation_code(observation: Observation) -> tuple[int, int, int]:
    if observation is None or type(observation) is Timeout:
        return 0, 0, 0
    if type(observation) is UntrustedText:
        if type(observation.text) is not str:
            raise ValueError("untrusted text must be a string")
        return 1, 0, 0
    if type(observation) is Receipt:
        index = _integer(observation.index, "receipt index", 0, 2)
        if type(observation.success) is not bool:
            raise ValueError("receipt success must be bool")
        return 2, index, int(observation.success)
    if type(observation) is Parity:
        return (3, _integer(observation.mask, "parity mask", 0, 7),
                _integer(observation.rhs, "parity rhs", 0, 1))
    if type(observation) is CountReceipt:
        mask = _integer(observation.mask, "count mask", 0, 7)
        return 4, mask, _integer(observation.count, "count", 0, mask.bit_count())
    raise ValueError("unsupported observation")


def _proposition_code(proposition: Equation | CountConstraint) -> tuple[int, int, int]:
    if type(proposition) is Equation:
        return _observation_code(Parity(proposition.mask, proposition.rhs))
    if type(proposition) is CountConstraint:
        return _observation_code(CountReceipt(proposition.mask, proposition.count))
    raise ValueError("expected Equation or CountConstraint")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LeanWorkflowSession:
    """One sequential, host-owned modeled state and one persistent Lean process.

    ``state`` is the last validated state. ``receipt`` binds the snapshotted
    source bytes, compiler executable and fresh build logs; it does not bind all
    shared runtime libraries or prove Python-to-Lean refinement. An unusable
    session cannot be reopened: construct a new one from a host-chosen state.
    """

    def __init__(self, initial: CertifiedState, *, root: Path | str | None = None,
                 lean_executable: Path | str | None = None, timeout: float = 30):
        self._state = _initial(initial)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise ValueError("timeout must be finite and in (0, 60] seconds")
        self._timeout = float(timeout)
        self._lock = threading.Lock()
        self._closed = False
        self._failure: str | None = None
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._responses: queue.Queue = queue.Queue()
        self._requests_sent = 0
        self._temporary = tempfile.TemporaryDirectory(prefix="lean_workflow_session_")
        self._directory = Path(self._temporary.name)
        self._root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]
        self._executable = (Path(lean_executable).resolve() if lean_executable is not None else
                            self._root / ".tools/lean-4.8.0-windows/bin/lean.exe")
        self._receipt: dict = {}
        try:
            self._build_snapshot()
            env = os.environ.copy()
            env["LEAN_PATH"] = str(self._directory)
            self._process = subprocess.Popen(
                [str(self._executable), "--run", str(self._directory / "WorkflowKernelCli.lean")],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="strict", bufsize=1,
                env=env, cwd=self._directory,
            )
            self._reader = threading.Thread(target=self._read_responses, daemon=True)
            self._reader.start()
            # Forces compilation/startup and proves a response arrives before EOF.
            if self._exchange("ASSERT 0 0 0 0") != "ASSERT 0":
                raise LeanBackendError("Lean startup assertion returned an invalid response")
        except Exception as error:
            self._abort(f"Lean backend initialization failed: {error}")

    def _build_snapshot(self) -> None:
        if not self._executable.is_file():
            raise LeanBackendError(f"Lean executable unavailable: {self._executable}")
        compiler_hash = _sha256(self._executable)
        source_hashes: dict[str, str] = {}
        source_paths: dict[str, Path] = {}
        for name in SOURCE_NAMES:
            source = self._root / "proofs/certified_memory" / name
            contents = source.read_bytes()
            source_hashes[name] = hashlib.sha256(contents).hexdigest()
            source_paths[name] = source
            (self._directory / name).write_bytes(contents)
        env = os.environ.copy()
        env["LEAN_PATH"] = str(self._directory)
        version = subprocess.run([str(self._executable), "--version"], capture_output=True,
                                 text=True, encoding="utf-8", timeout=self._timeout, env=env,
                                 cwd=self._directory, check=True).stdout.strip()
        compiles = []
        for name in SOURCE_NAMES:
            started = time.perf_counter()
            process = subprocess.run(
                [str(self._executable), "-o", str(self._directory / Path(name).with_suffix(".olean")),
                 str(self._directory / name)],
                capture_output=True, text=True, encoding="utf-8", env=env,
                cwd=self._directory, timeout=self._timeout,
            )
            log = {"source": name, "exit_code": process.returncode,
                   "seconds": time.perf_counter() - started,
                   "stdout": process.stdout, "stderr": process.stderr}
            compiles.append(log)
            if process.returncode or process.stderr.strip() or "sorryAx" in process.stdout or "error:" in process.stdout:
                raise LeanBackendError(f"fresh private Lean compile failed: {log}")
        if _sha256(self._executable) != compiler_hash or any(
                _sha256(source_paths[name]) != digest for name, digest in source_hashes.items()):
            raise LeanBackendError("source or compiler changed during snapshot build")
        self._receipt = {"source_sha256": source_hashes, "compiler_sha256": compiler_hash,
                         "compiler_path": str(self._executable), "compiler_version": version,
                         "fresh_private_imports": True, "snapshot_execution": True,
                         "compiles": compiles, "modeled_only": True}

    def _read_responses(self) -> None:
        try:
            while True:
                line = self._process.stdout.readline(4097)
                if not line:
                    self._responses.put(None)
                    return
                self._responses.put(line)
        except Exception as error:
            self._responses.put(error)

    def _shutdown(self) -> None:
        self._closed = True
        process = self._process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            if self._reader is not None:
                self._reader.join(timeout=1)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        self._temporary.cleanup()

    def _abort(self, reason: str) -> None:
        self._failure = reason
        self._shutdown()
        raise LeanBackendError(reason)

    def _ensure_live(self) -> None:
        if self._failure is not None:
            raise LeanBackendError(f"Lean session is terminally unusable: {self._failure}")
        if self._closed:
            raise LeanBackendError("Lean session is closed")
        if self._process is None or self._process.poll() is not None:
            self._abort("Lean process exited; session is terminally unusable")

    def _exchange(self, line: str) -> str:
        self._ensure_live()
        if not self._responses.empty():
            self._abort("unsolicited Lean output before request")
        try:
            self._process.stdin.write(line + "\n")
            self._process.stdin.flush()
            self._requests_sent += 1
            answer = self._responses.get(timeout=self._timeout)
        except (OSError, UnicodeError, queue.Empty) as error:
            self._abort(f"Lean request failed or timed out: {error}")
        if not isinstance(answer, str) or not answer.endswith("\n") or len(answer) > 4096:
            self._abort(f"Lean process ended or returned an invalid response: {answer!r}")
        self._ensure_live()
        return answer.rstrip("\r\n")

    @property
    def state(self) -> CertifiedState:
        return self._state

    @property
    def receipt(self) -> dict:
        return copy.deepcopy(self._receipt)

    @property
    def usable(self) -> bool:
        return not self._closed and self._failure is None and self._process is not None and self._process.poll() is None

    @property
    def requests_sent(self) -> int:
        """Includes the startup assertion; useful for proving pre-IPC rejection."""
        return self._requests_sent

    def step(self, action: Action, observation: Observation, candidate: ExactMemory) -> LeanStepResult:
        with self._lock:
            self._ensure_live()
            candidate = _exact(candidate)
            action_code = _action_code(action)
            tag, x, y = _observation_code(observation)
            prior = self._state
            fields = (prior.memory.world_bits, prior.policy.grants, prior.policy.budget,
                      action_code, tag, x, y, candidate.world_bits)
            answer = self._exchange(" ".join(map(str, fields)))
            tokens = answer.split(" ")
            if len(tokens) != 7 or any(not token or any(ch not in "0123456789" for ch in token) for token in tokens):
                self._abort(f"invalid Lean update response: {answer!r}")
            values = tuple(map(int, tokens))
            support, grants, budget, accepted, contradiction, permitted, legal_mask = values
            if not (1 <= support <= 255 and 0 <= grants <= 7 and 0 <= budget <= MAX_BUDGET
                    and accepted in (0, 1) and contradiction in (0, 1) and permitted in (0, 1) and 0 <= legal_mask <= 7):
                self._abort(f"out-of-range Lean update response: {answer!r}")
            next_state = CertifiedState(ExactMemory(3, support), ProtectedPolicy(3, grants, budget))
            expected_permitted = action in legal_actions(prior)
            expected_budget = prior.policy.budget - int(expected_permitted and type(action) is Submit)
            expected_legal_mask = sum(1 << item.index for item in legal_actions(next_state) if type(item) is Submit)
            if (grants != prior.policy.grants or budget != expected_budget or permitted != expected_permitted
                    or legal_mask != expected_legal_mask or (accepted and contradiction)
                    or (accepted and support != candidate.world_bits)
                    or (not permitted and (next_state != prior or accepted or contradiction))):
                self._abort(f"Lean update violated policy or status invariants: {answer!r}")
            self._state = next_state
            return LeanStepResult(next_state, bool(accepted), bool(contradiction), bool(permitted), legal_mask)

    def grounded_assert(self, proposition: Equation | CountConstraint) -> bool:
        with self._lock:
            self._ensure_live()
            tag, x, y = _proposition_code(proposition)
            answer = self._exchange(f"ASSERT {self._state.memory.world_bits} {tag} {x} {y}")
            if answer not in ("ASSERT 0", "ASSERT 1"):
                self._abort(f"invalid Lean assertion response: {answer!r}")
            return answer == "ASSERT 1"

    def close(self) -> None:
        with self._lock:
            self._shutdown()

    def __enter__(self) -> LeanWorkflowSession:
        self._ensure_live()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
