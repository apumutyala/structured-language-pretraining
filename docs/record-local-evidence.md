# What we have tested locally

The language-model pilot and the new RECORD action learner both ran on the
RTX 5070 Ti Laptop GPU. CPU CI is a separate portability check. This page
separates a successful execution, a formal statement and useful learned ability.

## Existing language pilot

The normalized 8.7M language model ran three arms for 1,000 updates each from
random initialization. All passed the local numerical-stability screen. Final
story bits per byte were approximately 1.819 without evidence injection, 1.866
with internal injection and 1.819 with output-only injection. This was one seed
with small development probes, not an architecture advantage. The failed earlier
unnormalized run remains in [the historical results](results.md).

## New formal model

[SupportedState.lean](../proofs/supported_state/SupportedState.lean) has 23
machine-checked theorem declarations covering support preservation, consistent
records, revision invalidation, authority preservation and current-policy
release. The empty-state and finite-history results connect the invariant from
initialization through arbitrary finite serialized histories in that model.

Root independently recompiled the proof and checked 994 Python/actual-Lean
fixtures with no mismatches. Two additional formal-only payload rejection
tests and 18 malformed adapter tests passed. No new axioms or admitted proof
obligations were used. Standard Lean foundational axioms are listed per theorem.

This is an **abstract specification plus bounded correspondence evidence**.
There is no universal Python refinement, hostile-process isolation, neural
alignment or unconditional external-truth theorem. Conditional truth requires
true admitted evidence and sound declared rules. The proof also constructs a
counterexample to the claim that support alone implies external truth.
See the [proof boundary and commands](../proofs/supported_state/README.md).

## New learned RECORD loop: 400 updates

This connects a learned recurrent proposer to the actual Python RECORD kernel.
It is an action-learning prerequisite, not joint language pretraining.

Each world has eight Boolean keys, five unary rules, two facts, permissions and
a question. The learner proposes derive/release/abstain operations through raw
unmasked action heads. Checked operations return intermediate records and
failure feedback before the next decision. No oracle planner, applicable-action
mask or gold answer is inserted into inference. Current release epochs and
generated record IDs are supplied by a common grammar adapter.

Both arms have the same 26,068 initialized parameters and labels. Internal
uses the full typed workspace through a learned projection before recurrence.
External reads the full same workspace as JSON, retains recurrent state, and
receives the same question again at the end. The inactive external projection
means 17,300 parameters receive task gradients there. Encoding, active capacity,
input length and parsing burden differ, so this is not an isolated architectural
comparison or automatically a strong external baseline.

The frozen recipe used 256 training worlds, batch 8, 400 Adam updates at 0.001,
FP32, no clipping, three paired initialization seeds and no checkpoint selection.
All three held-out strata have 128 worlds, including 77 answerable ones. Training
uses depth-one/two chains; the composition stratum uses depth three. Paraphrase
uses one unseen template on different worlds, not paired wording-only changes.
Semantic-world signatures are disjoint, but this is a small synthetic benchmark.

| Seed | Arm | Training correct / answerable | IID | Paraphrase | Composition |
| --- | --- | ---: | ---: | ---: | ---: |
| 2026102141 | Internal | 13/153 | 1/77 | 0/77 | 0/77 |
| 2026102141 | External JSON | 0/153 | 0/77 | 0/77 | 0/77 |
| 2026102147 | Internal | 28/153 | 0/77 | 0/77 | 0/77 |
| 2026102147 | External JSON | 0/153 | 0/77 | 0/77 | 0/77 |
| 2026102151 | Internal | 19/153 | 4/77 | 0/77 | 0/77 |
| 2026102151 | External JSON | 0/153 | 0/77 | 0/77 | 0/77 |

All six runs completed finite training. **The useful-comparator gate failed.**
The external gate required at least 80% supported training recall and 50% IID
recall in every seed. Internal fit was poor too. These results do not establish
internal superiority, architectural impossibility, or that more training cannot
help. They identify an unresolved learning/optimization problem at this budget.

Always abstaining gets 0% supported-answer recall while reaching 39.84% overall
held-out task accuracy. The exact unary planner gets 100% with a parsed goal and
the same kernel/budget; that is a symbolic ceiling, not a learned language model.
We therefore report correct supported answers separately from negative abstention.
A supported, permitted answer to the wrong question is still a task error.

Total measured run time was **227.125 seconds**; maximum allocated CUDA memory
was **190,961,664 bytes**. Per-arm training/evaluation time and memory are in the
raw report. Environment: RTX 5070 Ti Laptop GPU (sm_120), driver 610.88,
PyTorch 2.14.1+cu130, CUDA runtime 13.0, Python 3.12.14, Windows 11, FP32.
CPU checkpoint replay separately reproduced complete action/feedback sequences.

- [Frozen protocol](../experiments/record_comparison/PROTOCOL.md)
- [Raw aggregate report, all six runs](../experiments/record_comparison/results/v2/report.json)
- [World manifest](../experiments/record_comparison/results/v2/world_manifest.json)
- [Before-run configuration and source/environment hashes](../experiments/record_comparison/results/v2/preregistered.json)

## Reproduce the action learner

Use a suitable installed PyTorch runtime. No pretrained weights or corpus
downloads are needed for these synthetic tasks. Each output directory must be
new. Generated checkpoints stay local and are excluded from this repository.

```console
python -m pytest tests/test_record_learning_comparison.py tests/test_record_comparison_independent_review.py -q
python tools/train_record_comparison.py --config configs/record_comparison_cpu_enablement_20261021.json --device cpu --output runs/record-cpu
python tools/train_record_comparison.py --config configs/record_comparison_cuda_qualification_20261021.json --device cuda --output runs/record-gpu-qualification
python tools/train_record_comparison.py --config configs/record_comparison_cuda_paired_20261021.json --device cuda --output runs/record-gpu-paired
python tools/analyze_record_comparison.py runs/record-gpu-paired/report.json
```

CUDA commands require a local GPU; they are not run by CPU CI. Qualification
checks resource use before the larger run. Reports retain failures and final
checkpoints, with no replacement of failed seeds. CPU/CUDA floating-point
association can differ; replay tools report differences rather than replacing
the original evidence.

## Controlled additional-training test: 1,600 updates

To test whether the small result simply needed more training, we held model,
data, initialization seeds, Adam settings, learning rate and evaluation fixed,
and increased from 400 to 1,600 updates. The 400-update checkpoints did not
include optimizer state, so the longer runs replayed the original random
initialization and training prefix instead of pretending to resume Adam.

All 161 comparison checks passed. Every arm's first 400 loss, gradient-norm and
token-count rows matched exactly; source, data, environment, initialization and
checkpoint digests were checked. Both experiments and their costs are retained.

| Seed | Internal training correct /153, 400 to 1,600 | Internal IID correct /77, 400 to 1,600 |
| --- | ---: | ---: |
| 2026102141 | 13 to 121 | 1 to 1 |
| 2026102147 | 28 to 149 | 0 to 1 |
| 2026102151 | 19 to 132 | 4 to 1 |

Internal mean training recall rose from **13.07% to 87.58%**. Mean IID recall
changed from **2.16% to 1.30%**. Paraphrase and composition recall remained zero;
external JSON recall remained zero on training and held-out worlds. Means are
across three initialization seeds on shared worlds, not hundreds of independent
world replications. No improvement or inferiority claim about model families
follows from these small development results.

**More training clearly improved training fit. It did not improve held-out
generalization in the tested range.** Further training, better data coverage,
different capacity or optimization may still help. This is evidence to separate
those explanations, not evidence that extra compute can never work.

The full longer run took **719.328 seconds**, including replay of the first
400 updates. Combined cost with the original run was **946.453 seconds** and
12,000 charged optimizer updates. Peak allocated memory remained 190,961,664
bytes. The same development evaluations were reused; these are not newly sealed
confirmation results.

- [Raw longer-run report](../experiments/record_comparison/results/v2-1600/report.json)
- [Frozen longer-run configuration](../experiments/record_comparison/results/v2-1600/preregistered.json)
- [Independent matched-budget audit](../results/record_training_budget_comparison_20261021.json)
- [Fixed-input retention mathematics and measured limits](record_retention_diagnostic.md)

The contraction calculation concerns incoming recurrent-state sensitivity. It
does not bound all input, parameter or readout gradients and does not prove a
learning impossibility. Both interfaces receive fresh observations at each step.

## Equal-encoding placement fit diagnostic

This separate experiment removes the JSON parsing and input-length differences.
Both models receive identical question bytes and the same typed workspace
projection. Internal adds the projection before the recurrent core at each
question token; typed-readout adds it immediately before the final RMS/SwiGLU
readout. Both have 68,500 active parameters, the same initialized tensors and
the same training batches. Recurrence and projection repetition differ as the
intended placement intervention.

The fixed protocol used one new seed, 32 training worlds, hidden width 64,
1,500 updates at learning rate 0.001, no clipping, and no best-checkpoint
selection. It changes model/data as well as representation relative to the
earlier experiment, so it is not another controlled test of training budget.

| Placement | Training correct / answerable | IID | Paraphrase | Composition |
| --- | ---: | ---: | ---: | ---: |
| Before recurrent core | 19/19 | 2/77 | 0/77 | 0/77 |
| At typed readout | 19/19 | 2/77 | 0/77 | 0/77 |

Both passed the predefined 80% positive training-fit gate. Both failed to show
useful held-out generalization. This demonstrates that the learned proposal
pipeline can fit the training sequences under both placements; it does not
establish an integration benefit or a strong general external planner.

Both processed exactly 395,208 teacher-forced query bytes. Total measured run
time was **165.219 seconds**, with **69,262,336 bytes** maximum allocated CUDA
memory on the same laptop stack. This test is separate from the historical
8.7M language pilot and the 400-to-1,600-update action-learning experiment.

- [Placement protocol](../experiments/record_comparison/PLACEMENT_PROTOCOL.md)
- [Raw placement report](../experiments/record_comparison/results/placement-fit/report.json)
- [Before-run placement configuration](../experiments/record_comparison/results/placement-fit/preregistered.json)

```console
python tools/train_record_placement_diagnostic.py --config configs/record_placement_cuda_qualification_20261021.json --device cuda --output runs/placement-qualification
python tools/train_record_placement_diagnostic.py --config configs/record_placement_fit_20261021.json --device cuda --output runs/placement-fit
```

## Remaining research

The equal-encoding control can fit this small training set. Useful generalization
is still unresolved. Changing capacity, data and steps together cannot isolate
additional compute. Even a successful action learner would leave joint language training, meaningful
natural-language generalization, evidence-revision episodes, strong baseline
qualification, data/tokenizer work and the H100 path unfinished.

The [conditional compute plan](compute-plan.md) funds a replicated test of the
larger question. It does not promise that compute alone fixes the current model,
or that formal support semantics imply unrestricted language safety.
