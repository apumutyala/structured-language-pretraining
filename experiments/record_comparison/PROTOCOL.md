# Local RECORD action-learning comparison, protocol v2

This experiment tests a small learned proposal loop through the actual
`supported_state.py` kernel. It does not train a joint language model, demonstrate
language pretraining, qualify full8/L0–L4, or compare architecture superiority.
All parameters start randomly. No inherited learned checkpoint, post-training,
gradient clipping, cloud rental, paid work, or proof-source modification is used.

The preregistered CPU configuration is
`configs/record_comparison_cpu_enablement_20261021.json`. Separate CUDA resource
qualification and three-seed configurations are in the same directory. The run
directory saves configuration, source hashes, device/runtime provenance, and
world manifest before the optimizer runs. Each output directory must be new.
The coordinator owns any local GPU lease and decides the larger run only after
resource qualification. The CUDA configuration itself is not a GPU launch.

## Task and supervision

Each finite world has eight Boolean keys, five unary ground rules, two trusted
facts, a current release policy, and a question about one key. Rule IDs/order,
key roles and truth values vary. The positive chain has length one or two in
training, three in the composition test. The planner labels derive/release/
abstain actions during initial training; it is never consulted for neural
action selection. It does not supply applicability masks, closure features,
answers, traces or gold records to a model observation. Oracle state is applied
only to teacher trajectories. Free-running episodes apply only predicted
operations to the exact kernel; the expected answer is computed separately
after the neural rollout ends.

The operation budget is four in every arm/split. Learned heads select operation,
rule ID, premise-record pointer and release-record pointer. A common grammar
adapter supplies the fresh step record ID and current release epochs. This is
task-specific structured action prediction, not generic tool JSON generation.
No validity or query-relevance mask restricts heads. A rejected operation costs
one step and its actual feedback enters the next observation. Teacher trajectories
contain accepted operations only, so rejection recovery is not trained.

Both arms allocate the same 26,068 parameters for hidden width 32/native width
64, and initialize identically for each paired seed. They train on identical
worlds, labels, batches, teacher feedback, grammar and optimizer settings. SSM
state starts at zero per episode and is carried, without detach, across teacher
operations. It resets between training batches. Free-running evaluation carries
state across predicted operations and resets between worlds. Terminated teacher
rows are repeated in batched forwards but excluded from losses; repeated tokens
are charged in actual token work. There is no checkpoint selection using tests.

Internal access uses query bytes and a learned projection of the current full
typed finite workspace before the SSM. The projection contains rules, existing
records, source/provenance pointers, permissions, epochs, budget and feedback.
Schema/grammar/universe are fixed constants. External access serializes the full
same public observation as canonical JSON at every operation and appends the
identical question again after it. Both question copies count toward input work.
The external SSM is stateful. The internal projection is allocated but inactive
in the external model: 17,300 external parameters versus 26,068 internal
parameters receive task gradients. Serialization length, parsing burden,
effective recurrent memory, active capacity and compute differ materially.
This is therefore an interface/encoding enablement experiment and cannot isolate
an architectural effect or establish a strong general external comparator.

Before any run, review identified two flaws in the draft v1 fixture: a fixed
evidence ID marked the causal root, and the question occurred only before a long
external serialization. v2 randomizes evidence IDs, keeps two facts in both
positive and missing-root worlds by replacing absent root with a distractor, and
duplicates the external question at the end. No v1 optimizer run occurred.

## Evaluation and decision rules

Semantic-world SHA-256 signatures exclude split ID and wording; all training and
held-out signatures must be disjoint. Independent random worlds are generated
in each split. The held-out paraphrase uses a question template never trained;
the composition split increases causal depth. The full CUDA protocol uses 256
training worlds, 128 worlds in each of three held-out strata and three paired
fresh seeds. These small synthetic strata do not establish language generality.

The primary usefulness metric is free-running supported-answer recall, with
correct-positive numerator and supported-world denominator always reported.
Task exactness includes correct abstentions and is secondary. No-release budget
exhaustion does not count as an explicit abstention. Invalid-operation rate,
wrong-query checked releases, explicit abstention correctness, token work, wall
time, and CUDA peak allocated/reserved memory are reported. A checked release
can be supported and policy-permitted yet answer the wrong query: that is a
task error. Kernel checking alone does not imply usefulness or external truth.
Host evidence truth, declared-rule soundness and channel ownership remain
assumptions. There is no general process-isolation or neural-safety proof here.

Always-abstain and exact unary planner controls are explicit. The CPU dataset has
15 supported worlds out of 24 training worlds and 5/8 in each held-out stratum.
Always abstaining scores 37.5% task exactness and 0% supported-answer recall;
the exact planner reaches 100% on both through the same kernel/budget. The CUDA
manifest determines its own denominators; they are not inferred from CPU rates.

Paired world-bootstrap intervals use 2,000 resamples and are computed separately
for supported-answer recall and task exactness within each seed and stratum.
They do not adjust for seed uncertainty, multiplicity or dataset design; no
superiority conclusion is authorized. Degenerate zero intervals when both arms
fail all positives describe these fixtures, not certainty of equal capability.

Finite optimization and full rollout completion are the enablement gate only.
The larger CUDA protocol separately declares pragmatic external fit thresholds:
at least 80% supported training recall and 50% IID supported recall in every
seed. Passing would show within-task usefulness, not validate a strong generic
external baseline. If external fit fails, report it and treat this comparison
as unresolved. Do not infer internal superiority from an underfit external arm.

## CPU evidence and cost

`runs/record-comparison-cpu-v2-20261021/report.json` preserves the first bounded
CPU receipt: fp32, PyTorch 2.14.1+cpu, two CPU threads, one paired seed, 12
updates. Total measured trainer time was 1.359 s. Internal/external training
took 0.094/0.454 s and evaluation 0.141/0.531 s. Training losses changed
4.833→3.957 / 4.903→4.091. Both arms answered **0/5 supported held-out worlds**
in every stratum and rejected 100% of proposed operations. Both also had 0/15
supported training answers. This proves runnable finite learning/rollout plumbing,
not a fitted model or useful learned comparator. CPU allocator peak memory is
unmeasured and explicitly null; no GPU speed or memory claim follows.

The CPU run processed 1,584/179,807 actual training input bytes internally/
externally. It used no paid compute ($0). At the same CPU shape the proposed
400-update batch-eight external training would be roughly 66.7 times the CPU
pilot's training-token work per seed (about 12 million serialized bytes), plus
640 evaluation worlds per arm. This is a workload estimate, not a CUDA runtime
forecast. Optimizer states and activations need actual CUDA measurement; 26,068
fp32 parameter tensors alone contain 104,272 bytes and do not bound peak VRAM.
The separate eight-update CUDA qualification records device/driver/runtime,
actual wall time and peak allocation before choosing a larger experiment.

## Commands

```powershell
.venv/Scripts/python.exe -m pytest tests/test_record_learning_comparison.py tests/test_record_comparison_independent_review.py -q
.venv/Scripts/python.exe tools/train_record_comparison.py --config configs/record_comparison_cpu_enablement_20261021.json --device cpu --output runs/record-comparison-cpu-v2-NEW
.venv/Scripts/python.exe tools/analyze_record_comparison.py runs/record-comparison-cpu-v2-20261021/report.json
.venv/Scripts/python.exe tools/replay_record_comparison.py --run runs/record-comparison-cpu-v2-20261021 --seed 2026102141 --arm external --split iid --output results/record-comparison-cpu-external-replay-v2-20261021.json
```

Only the coordinator holding the one local CUDA lease runs:

```powershell
.venv-gpu/Scripts/python.exe tools/train_record_comparison.py --config configs/record_comparison_cuda_qualification_20261021.json --device cuda --output runs/record-comparison-cuda-qualification-v2-20261021
.venv-gpu/Scripts/python.exe tools/train_record_comparison.py --config configs/record_comparison_cuda_paired_20261021.json --device cuda --output runs/record-comparison-cuda-paired-v2-20261021
```

Checkpoints stay in private ignored run directories. JSON receipts contain
basenames and file/tensor digests for independent replay. An incomplete run
preserves per-update progress and error details and exits nonzero. Do not replace
a failed seed quietly, overwrite a run directory, or select a successful retry.
