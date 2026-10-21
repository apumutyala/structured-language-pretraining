# Equal-encoding RECORD placement fit diagnostic

Registered before optimization. The frozen raw-JSON comparison remains separate
and unchanged. This one-seed diagnostic asks whether the same typed workspace
can be learned when added before recurrence or at final readout. It is an
action learner, with no joint language objective or inherited learned weights.

`configs/record_placement_fit_20261021.json` fixes fresh seed 2026102159, data seed
2026102161, 32 training worlds, 128 independently generated worlds per IID/
held-out-paraphrase/depth-three-composition stratum, hidden width 64, native
width 128, 1,500 updates, batch eight, Adam at 0.001 and fp32. The same v2 world
generator, trusted facts, rules, observed records, feedback, permissions, labels,
four-operation budget and checked kernel are used in both arms. Training
trajectories and batch order match. Signatures ensure semantic-world disjointness.
There is no early stopping, test-selected checkpoint, resume or replacement seed.

The two newly allocated model instances have exactly matched random initial
tensors and active parameters. Both consume the same question bytes and the
same full typed_workspace projection. In `internal`, the projection is added to
each query-token embedding before the SSM. In `typed_readout`, it is added once
to the last-valid core-plus-embedding residual, before the existing final
RMS/SwiGLU/readout. All projection/SSM/FFN/head parameters receive task gradients
in both arms; only the internal placement permits workspace information to
enter carried SSM state. Query/episode state reset and carry rules are identical.
The exact grammar adapter generates only fresh step IDs and current epochs;
operation, rule and record pointers are learned and unmasked.

Teacher input-token work is equal; addition count and gradient exposure differ
because internal routing repeats over query tokens whereas readout routing
occurs once. Actual free-running token work may differ because models choose
different episode lengths. These are recorded rather than forced equal. Both
arms see the full current snapshot every operation; the readout control is a
stateful typed external workspace interface, not a generic textual tool model.
This removes the earlier serialization/active-capacity confounds and provides a
more focused placement test. It does not by itself establish an architecture
advantage, language capability, general external baseline strength or safety.

The primary endpoint is checked supported-answer recall. Numerators and positive
denominators are explicit; task exactness/abstention/invalid operations/wrong-query
releases remain secondary. Both arms must reach at least 80% supported recall
on the 32 training worlds at the fixed final checkpoint to pass this diagnostic's
train-fit gate. Held-out performance is reported regardless of this gate and
never selects a checkpoint. Paired world-bootstrap intervals remain descriptive
within this one seed; they do not support a family-level conclusion. Always-
abstain and exact planner controls remain explicit. No unsupported-release or
true-world safety conclusion is inferred from a model that cannot answer.

Separate one-update CPU and eight-update CUDA qualifiers use the same width,
batch and recipe but eight held-out worlds per stratum. They qualify finite
execution/resource use only, not train fit, and do not replace the final seed.
No GPU operation is authorized by this document alone; the coordinator owns
the one local GPU lease. No paid compute is used. The 1,500-update run has a
900-second wall cap, bounded retained progress/failure receipts, and no clipping.
Both peak allocated/reserved CUDA memory and measured per-arm training/evaluation
time are required before interpreting costs.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_record_placement_diagnostic.py tests/test_record_placement_review.py -q
.venv/Scripts/python.exe tools/train_record_placement_diagnostic.py --config configs/record_placement_cpu_qualification_20261021.json --device cpu --output runs/record-placement-cpu-qualification-20261021
```

Coordinator-only, after source review and the previous GPU job exits:

```powershell
.venv-gpu/Scripts/python.exe tools/train_record_placement_diagnostic.py --config configs/record_placement_cuda_qualification_20261021.json --device cuda --output runs/record-placement-cuda-qualification-20261021
.venv-gpu/Scripts/python.exe tools/train_record_placement_diagnostic.py --config configs/record_placement_fit_20261021.json --device cuda --output runs/record-placement-fit-20261021
```

Implementation lives in new `record_placement_diagnostic.py` and
`train_record_placement_diagnostic.py`; reused comparison model/kernel/training
helpers are read-only and source-hashed with the new files in each run receipt.

CPU qualification completed before any placement GPU run: one update per arm,
fp32/two threads, 68,500 allocated and active parameters each, 264 teacher input
bytes each. Training took 0.016 s per arm; total trainer wall time was 1.297 s.
Both arms answered 0/19 supported training worlds and 0/5 in each eight-world
held-out stratum; the 80% train-fit gate failed, as expected for one update. The
qualifier establishes finite execution only. Eight combined source/review tests
passed in 1.38 s. Compact receipt:
`results/record_placement_cpu_qualification_20261021.json`. CUDA resource use and
1,500-update fit remain unmeasured. Pure fp32 parameter storage is 274,000 bytes;
optimizer states/activations/runtime allocations require actual measurement.
