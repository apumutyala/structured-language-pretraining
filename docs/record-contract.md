# Exact supported RECORD contract

The implementation is `supported_state.py`; `semantic_interface_comparison.py`
wraps it with equal logical observations. Neither is a new Lean theorem. The
language model currently uses a separate finite evidence lane. Integrating the
new RECORD interface into training remains prospective.

## State and authority

The host declares at most 16 Boolean keys and 64 finite ground rules. Each
rule has 1–16 ordered typed premises and one conclusion. A `Claim` contains an
exact Boolean value, not an integer coerced to bool. There are no variables,
empty-premise axioms, program evaluation or negation-as-failure. Up to 64 host
evidence items and 128 live records are retained. Identifiers use bounded ASCII;
record IDs allow 69 characters to accommodate the host `fact.` prefix.

`create_supported_state(rulebook, release_keys)` returns distinct `HostChannel`
and `ProposerChannel` objects. The application keeps the host channel, which
alone exposes `admit`/`replace`, `revoke` and `set_release_keys`. It must
authenticate the evidence and permissions before calling them. The proposer
exposes `snapshot`, `step` and `release`. Merely constructing a claim or
certificate does not admit it. A claim named `authority` cannot change policy.
This is an ownership convention in one Python process; hostile Python may
reach internals. Calls must be serialized by the host.

Each host record identifies its evidence item. A derived `Record` identifies a
declared rule and existing premise records with exactly matching claims. The
checker supplies the rule conclusion. Record IDs must be fresh; references
must already exist; self-reference is explicitly rejected. Accepted record
graphs are therefore acyclic and evidence-rooted even if the rulebook has
cycles. Missing/unfounded support rejects rather than creating a fact.

The conflict policy is first accepted consistent value until a host evidence
revision. Conflicting host admissions reject against other live host facts;
conflicting derivations reject against all live records. No explosion or
arbitrary inference from contradiction is performed. This policy is order
sensitive and does not verify joint satisfiability of every rule plus evidence.
Truth remains conditional on sound host rules and truthful host evidence.

Every successful evidence admission, replacement or revocation increments a
global evidence epoch and discards all derived records, including transitive
and unrelated dependents. Identical-value replacement does this too. This
conservative policy makes every old certificate stale. Host facts are rebuilt
as base records. Policy changes increment a separate policy epoch; support can
remain while all old release certificates become stale. Rejected mutations
leave state and epochs unchanged.

## Release and comparison grammar

A release certificate must exactly match a currently registered record,
session identifier and both epochs. Current release-key membership is then
checked. Successful output is deterministically `key=true` or `key=false` from
the registered claim. Rejections carry no text/claim. Certificates are support
witnesses rather than secrets. Snapshots intentionally reveal all facts,
rules, permissions and intermediate records; no confidentiality or
noninterference claim follows from release checks.

The comparison adapter admits exactly these operation objects:

| Operation | Required and only fields |
| --- | --- |
| Derive | `op="derive"`, `record_id`, `rule_id`, `premises` (list of record IDs, at most 16) |
| Release | `op="release"`, `record_id`, `evidence_epoch`, `policy_epoch` (exact integers, not booleans) |
| Abstain | `op="abstain"` |

JSON strings have a maximum of 8,192 characters. Duplicate members,
NaN/Infinity, malformed/deep JSON and fields outside the grammar reject.
Operation budget is 1–128; every attempt while budget remains is charged,
including invalid attempts. Exhausted calls perform no semantic operation.
An abstention consumes a step but does not silently grant extra attempts.
Every attempt returns the same detached state and success/failure feedback
schema to both arms. Current session certificates are reconstructed by the
host only after release epochs match. Nonces and certificates are omitted
symmetrically from observations; all semantic record content remains visible.

## Proof targets and boundary

| Target | Implementation argument, not a mechanized Python proof |
| --- | --- |
| Rooted acyclic support | `step` adds a fresh node with existing parents; `reset_records` builds only host roots. Induct on record creation order. |
| Explicit record consistency | `admit` checks other facts, `reset_records` drops derivations, and `step` rejects opposite-valued live claims. |
| Authority preservation | Proposer operations never change evidence/rules/release policy or either host epoch. |
| Revision/policy invalidation | Evidence mutations clear derived state and change evidence epoch; policy changes alter policy epoch; `release` binds both. |
| Conditional truth | For an interpretation satisfying current evidence and declared implications, every evidence-rooted derived claim is true by induction. The checker does not establish that such a truthful interpretation exists. |
| Checked rendering | `release` checks current record, session, epochs and policy, then renders only the registered claim. |

Unproved refinement obligations include the correspondence of this finite
mathematical description to Python control flow/equality, codecs, host channel
ownership, runtime integrity and deployment. Finite tests/adversarial review
are evidence, not universal proofs. The three-bit possible-world Lean workflow
is a different specification; its proofs do not transfer to this module.
