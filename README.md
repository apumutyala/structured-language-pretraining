# Structured language pretraining

Can an exact semantic workspace help a recurrent language model learn useful,
supported reasoning from random initialization, beyond an equally informed
external workspace using the same checker? This repository contains the small
language prototype, executable support interfaces, bounded proofs and negative
pilot evidence needed to investigate that question. It does not establish an
architecture advantage or general alignment.

Every learned model in the active recipe starts from random weights. There is
no inherited backbone, adaptation checkpoint or post-training strategy. Initial
joint supervision of byte prediction and typed claims is part of pretraining.

| Component | Implemented | Tested / proved | Still proposed |
| --- | --- | --- | --- |
| Byte language model | Selective SSM/GRU, shared evidence projection, internal/output-only/none routing | Small CPU forward/backward checks; local development training receipts | Useful broad language and robust held-out reasoning |
| RMS stability variant | Parameter-free normalization before FFNs and both heads | CPU checks and audited one-seed, three-arm, 1,000-update RTX 5070 Ti training run | Architecture benefit and broader usefulness unestablished |
| Exact RECORD kernel | Host facts, finite declared rules, checked derivations, revocation, policy-aware deterministic release | Adversarial tests; new abstract Lean model with 23 theorems and 994 matching runtime fixtures | Python refinement, host isolation and trained language integration |
| Same-information interfaces | Structured view and canonical JSON view; shared checked operations and feedback | New action learner: three paired GPU seeds, 400 updates per arm; fit gate failed | Useful strong external comparator, encoding/cost matching and joint language training |
| Three-bit workflow reference | Python memory checker, Lean specifications and persistent Lean adapter | Lean induction results for its declared workflow semantics; bounded historical differential evidence | Python-to-Lean refinement and transport authentication |

The RECORD interface is now connected to a small, randomly initialized neural
action learner. It is still separate from the 8.7M language model's existing
evidence lane. This is not yet the trained joint language comparison. The new
Lean specification and executable differential checks also do not prove universal
equivalence to Python. [New local evidence and limitations](docs/record-local-evidence.md).

```mermaid
flowchart LR
  H[Host facts and policy] --> S[Exact typed state]
  Q[Untrusted text and question] --> M[Random-init recurrent model]
  S --> M
  M --> P[Typed proposal]
  P --> K[Checked transition and release]
  S --> K
  M --> U[Uncertified prose]
```

The diagram shows the intended language-model contract. RECORD integration is
currently demonstrated only in the smaller action learner, with poor usefulness.
[Architecture details](docs/architecture.md) describe the original language lane.

## Reproduce checks without a GPU

With Python 3.12 and the dependencies installed:

```sh
python -m pip install -r environment/requirements-cpu.txt
python tools/check_cpu.py
```

The second command runs the deterministic interface demo, a tiny random-init
language forward pass and selected non-optimizing CPU tests. It downloads no
corpus, launches no training, accesses no GPU and requires no Lean installation.
An optional actual-Lean smoke test skips when Lean is unavailable. Installation
requires package downloads. Exact tested versions are recorded in the runtime
receipt; dependency availability on every platform is not claimed.

## What the pilot showed

The training evidence is **not CPU-only**. The historical 8.7M normalized
language pilot used the local RTX 5070 Ti. The new RECORD action-learning
comparison also used that GPU: six complete 400-update runs in 227.125 seconds.
All seeds are retained. Both interfaces failed useful learning qualification;
no architecture win follows. See [new GPU results](docs/record-local-evidence.md)
for positive-answer denominators, raw outputs, environment and commands.

The historical unnormalized local 1,000-update development pilot ended at
2.4695, 2.1741 and **9199.6580 bits per UTF-8 byte** for none, internal and
output-only routing respectively. The output-only arm failed badly; this is a
stability confound, not evidence that internal semantics is superior. There is
one seed, a small development slice, no held-out confirmation and no useful
language qualification. A newer normalized 1,000-update local run completed all three arms stably,
with final development bpb about 1.819 / 1.866 / 1.819; its AI-assisted independent
15-checkpoint audit passed. This closes a local reliability gap at this setting and
establishes no semantic-integration advantage. See [results](docs/results.md) for
metrics, denominators, provenance and limitations.

Raw next-byte prose is always uncertified. Exact support applies only to the
specified typed/rendered endpoint, conditional on truthful authenticated host
evidence, sound rules, current policy and correct implementation/deployment.
Neither a low training loss nor an exact checker establishes correct question
interpretation, general truthfulness, confidentiality or absence of deception.
The Python host/proposer split expresses ownership within one process; it is
not protection against hostile Python code.

## Research and reproduction

- [Architecture and trust boundaries](docs/architecture.md)
- [Primary matched-interface experiment](docs/experiments.md)
- [CPU, data, optional training and Lean reproduction](docs/reproduction.md)
- [Conditional compute plan](docs/compute-plan.md)
- [Results and source-bound evidence](docs/results.md)
- [Relevant predecessor and what we learned](docs/prior-work.md)

The H100 study is a proposed larger experiment, not the runtime cost of this
small reviewer demo. The reported CUDA runs were measured on the local laptop;
the new action-learning scripts check CUDA availability but do not enforce a
particular GPU name. An H100 language-training recipe and measured memory/throughput
qualification are not implemented here. No rental, paid execution or automatic GPU workflow
is enabled. CI runs only when manually requested, with read-only repository
permissions.

`PROVENANCE.json` records the original release's copied source bytes and two
declared transformations. New experiment receipts bind their executed source,
configuration, data and checkpoints by hash. `PUBLIC_MANIFEST.json` lists relative paths and SHA-256 digests
of the candidate files; it excludes itself to avoid a circular digest. Derived
receipts declare their original report digest and are never described as the
immutable original. No corpus bodies or checkpoints are committed. Project
source is MIT licensed; external dependencies/data retain their own terms.
