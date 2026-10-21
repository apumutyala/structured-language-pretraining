# Supported-record semantic model

`SupportedState.lean` is a machine-checked abstract specification for the
finite supported-record interface in `src/controlled_lm/supported_state.py`.
It is separate from the possible-world workflow proofs. It does not prove
Python-to-Lean refinement, authenticate observations, certify a learned model,
or establish behavioral alignment or an open-world hallucination guarantee.

The mechanism can check a structured interface designed before initial
from-scratch training. This artifact contains no training, embeddings, inherited
parameters, learned RECORD mechanism, or neural quality evidence.

## Proven scope

The local Lean 4.8.0 compiler checks 23 theorem declarations, including:

- Empty initialization establishes supported and consistent records.
- Every successful host evidence revision rebuilds records solely from the
  current evidence, with support and (assuming consistent evidence) consistency.
  The evidence epoch increases even for unchanged-value replacement.
- Checked rule application requires an exactly declared rule, exactly matching
  ordered premise claims from existing records, a fresh record identifier, and
  no opposing live conclusion. Existing premise identifiers differ from the
  new identifier. Rejection preserves the state.
- Checked rule application preserves support, consistency, evidence, rulebook,
  release policy, session and both epochs. Changing policy preserves support.
- Support and consistency are preserved across arbitrary finite serialized
  `Evolution` histories starting with those invariants. Every host revision in
  such a history must supply consistent evidence. `initial_invariant` and
  `initial_consistent` discharge the empty-initialization premises.
- Every released claim is registered and currently permitted; release checks
  exact session, evidence epoch, policy epoch and whole-record binding. Any
  certificate minted before an evidence or policy revision is rejected, even
  if the new evidence or policy has identical contents.
- With nonempty declared rule premises, inductively supported claims require
  an evidence root. Cyclic rule declarations cannot supply a root themselves.
- For a world in which **all current evidence is true and all declared rules
  are sound**, every supported or successfully released claim is true in that
  world. The theorem neither establishes the existence of such a world nor
  establishes that it is the external world.
- `support_does_not_imply_external_truth` constructs supported evidence that
  is false in a chosen external interpretation. No axiom asserts truthful
  evidence, sound rules, or alignment.

`Supported` is the least inductive closure of current facts under declared
rules; proof terms are finite. `Invariant` says each current record's claim
belongs to that closure. This proves claim support, rather than equality of
the entire stored provenance graph with a particular proof term. The executable
step separately checks membership of full premise records and writes their
identifiers into the derived record's source.

## Correspondence and abstraction boundary

| Formal object/operation | Python relationship and remaining boundary |
| --- | --- |
| `Nat` keys, identifiers, session | Abstract identifiers for validated strings and the session UUID. No parser, UUID uniqueness, identifier-length or encoding theorem. Evidence and derived record IDs must be mapped consistently; the CLI reserves distinct numeric IDs for its fixtures. |
| Lists of facts, rules and records | Abstract finite collections. Python uses immutable dataclasses and a private mutable dictionary. The proof does not establish dictionary lookup, dataclass equality, type validation, or thread safety. |
| `initial` | Empty core state with a host-supplied rulebook and policy. Python's rulebook/schema validation is outside this model. |
| `revise` | Successful `admit`/`replace`/`revoke` reset path after the host has validated its new evidence. It receives the entire new fact list. Conflict rejection, unknown-ID rejection, and atomic failure of the host API are not modeled. The consistency theorem explicitly assumes consistent supplied evidence. |
| `stepOK`/`checkedStep` | Checks full proposed rule and resolved premise payloads against the live registry. Python accepts rule/premise IDs and resolves them internally. Equality/membership abstraction replaces lookup; it is not a proved compiler or Python translation. Python additionally rejects malformed types and imposes bounds. |
| Freshness and nonempty rules | Checked on accepted formal steps. Global uniqueness of host evidence IDs and rule IDs, key-universe validation and noncontradictory rule premises belong to Python's validated host inputs. The root theorem explicitly assumes every declared rule has nonempty premises. No full stored-graph acyclicity or uniqueness theorem is claimed. |
| `setPolicy` | Successful validated host policy change, including an epoch increment for unchanged policy. Host policy validation is outside the model. |
| `certificate`/`release` | Exact current witness and key allowlist check, returning `Option Claim`. Python returns result dataclasses and renders `key=true/false`. String rendering, transport, deserialization and malformed-certificate handling are not formalized. The certificate is replayable support, not a cryptographic secret. |
| Unbounded `Nat` counters and finite lists | No artificial epoch wraparound. Python integers are unbounded; its 16-key/64-rule/128-record/64-evidence limits are omitted. The formal model may accept inputs outside those runtime bounds. |

The rulebook is unchanged throughout the modeled proposer, revision and policy
operations. The trusted host supplies evidence and release policy, validates
inputs, and serializes calls. Authentication, truthful external evidence,
sound declared rules, host/channel ownership, process isolation, race freedom,
Python/runtime correctness and secure deployment remain outside the proof.
Snapshots and records are visible; release policy is not a confidentiality or
information-flow theorem. A privileged Python caller can access or modify the
shared core. No inaccessible-host or arbitrary hostile-code claim is made.

## Bounded executable correspondence evidence

Run from the repository root, with the existing local Lean binary installed:

```text
python tools/check_supported_state_lean.py
python -m pytest tests/test_supported_state_lean.py tests/test_supported_state.py tests/test_supported_state_review.py -q
```

The checker uses the existing `.tools/lean-4.8.0-windows/bin/lean.exe` or an
explicit `SUPPORTED_STATE_LEAN` override. It compiles current sources into a
fresh temporary import directory, checks all 23 `#print axioms` reports, and
runs the actual Lean CLI on fixtures frozen before comparison. It rejects
unexpected/missing theorem audits, custom axioms, `sorryAx`, and common proof
shortcuts. The admitted foundational dependencies are `propext`, `Quot.sound`
and `Classical.choice`; the receipt lists dependencies per theorem. There are
no newly declared axioms and no admitted proof obligations.

`check_receipt.json` records compiler version, source hashes, fixture hash,
counts, dependencies and timings. Current checked evidence:

- **994** distinct Python/actual-Lean fixtures, **zero mismatches**.
- **972** exhaustive small contexts: all ordered single-premise Boolean rules
  over three keys and all 27 partial Boolean evidence assignments. Independent
  arithmetic acceptance/mask expectations also agree with Python.
- **14** revision/policy scenarios: unchanged/opposite/key replacement,
  revocation, additional evidence, all eight release-key masks and unchanged
  policy. These include two-step transitive support and old certificates.
- **8** conjunction scenarios: acceptance, wrong order, duplicate/missing
  premises, existing output ID, undeclared rule, conflicting conclusion and
  unsupported premise.
- **2** extra Lean-only payload perturbations reject fabricated premise
  metadata and an altered full rule. Python's ID-only proposal API cannot
  express those payloads, so these are explicitly **not** differential cases.
- **18** malformed test-adapter lines reject. This adapter is not a production
  deserializer or authentication boundary.
- Independent enumeration checks every current Python record against every
  satisfying three-key interpretation in the fixtures. Zero-model contexts are
  counted separately; consistency does not imply rulebook satisfiability.
- **51 tests pass** across the new checker tests and the existing supported-state
  implementation/review suites.

The compared output is the projection of acceptance, both epochs, record count,
claim mask, current releasable-claim mask, and old-certificate acceptance. It
does not compare all record provenance fields, messages, rendering, arbitrary
malformed Python objects, every rulebook, arbitrary histories or all allowed
runtime states. No universal implementation correspondence follows from these
bounded checks. The remaining substantive gap is a proved refinement or a
production path that directly executes a verified implementation; neural
integration and model-quality/safety evaluation are separate obligations.
