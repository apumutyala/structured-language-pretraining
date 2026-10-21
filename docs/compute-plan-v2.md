# Detailed conditional compute plan

This replaces the older broad allocation for future planning. It is not a
provider quote, a measured H100 forecast, an executable training recipe, or
permission to spend. Staff costs are already covered. Local preparation uses
the existing RTX 5070 Ti Laptop GPU.

## What the compute would buy

The primary question is whether integrating supported semantic state during
initial learning improves useful generalization or complete cost over an equally
informed stateful external workspace. Both learners start randomly. Their
evidence, rules, intermediate feedback, labels and operation budgets must match.
The larger models and data are the proposed experiment, not completed results.

| Stage | Comparison | Runs | Parameters | Tokens per run |
| --- | --- | ---: | ---: | ---: |
| Discovery | 2 interfaces × 2 supervision conditions × 3 seeds | 12 | 125M | 2.5B |
| Discovery | Same design | 12 | 350M | 7B |
| Conditional scale | 2 recipes × 3 fresh seeds | 6 | approximately 1B | up to 20B |
| Optional references | 1 Transformer × 2 supervision conditions × 3 seeds at each size | 12 | 125M / 350M | 2.5B / 7B |

Token counts refer to the future shared tokenizer, which still needs selection.
They are not interchangeable with byte-token counts from local pilots. The
approximately 1B stage is conditional on useful learning, stability, meaningful
remaining uncertainty and measured affordability. A larger single-seed run
cannot replace replicated confirmation.

| Allocation | USD |
| --- | ---: |
| Profiling and equally budgeted recipe searches | 500 |
| Primary discovery, including supporting GPU work | 3,500 |
| Conditional approximately 1B comparison, including supporting GPU work | 12,600 |
| Optional Transformer references | 1,750 |
| Additional CPU verification, storage and I/O | 600 |
| Failed-job and compute reserve | 1,050 |
| Maximum | 20,000 |

The $18,000 option omits optional references and uses an $800 reserve. No paid
agent API costs or staff costs appear in this ledger.

## Reproduce the arithmetic

```console
python tools/check_compute_plan_v2.py --config configs/structured_pretraining_compute_v2.json --output artifacts/compute-plan.json
python -m pytest tests/test_compute_plan_v2.py -q
```

The convention is nominal core training work `6 × parameters × tokens`.
Assume $4.50 per GPU-hour and add a 40% supporting GPU-work allowance once for
validation, checkpointing and recovery. Qualification, extra services and the
reserve are separate. Effective throughput must be measured for our actual
implementation. Advertised chip peak throughput is not a substitute.

| Assumed effective core rate | Full 42-run forecast, including reserve | Base 30-run forecast, including reserve |
| --- | ---: | ---: |
| 50 nominal TF/s | $37,792.25 | $34,061.50 |
| 100 nominal TF/s | $19,971.13 | $17,980.75 |
| 200 nominal TF/s | $11,060.56 | $9,940.38 |

These are sensitivity scenarios. At 100 nominal TF/s there is almost no
unallocated headroom beyond the stated reserve. Price, throughput, memory,
semantic-operation overhead and actual token accounting can change the scope.
The current full plan has not been measured on H100. Multiple devices shorten
wall time only if they improve execution; every device-hour is billed.

If a future throughput measurement already includes checkpointing, checking and
evaluation, do not add the same 40% costs again. Count idle allocations, taxes,
failed work, communication and storage exactly once in the final manifest.

## Conditions before paid execution

1. Qualify the actual learned internal and strong external interfaces. A tiny
   action-learning pilot is an implementation check, not the funded language study.
2. Freeze data, tokenizer, training objectives, equal tuning opportunities,
   confirmation metrics and all failure reporting.
3. Qualify the intended H100 precision and numerical path; measure complete
   representative throughput and peak memory.
4. Produce an independently checked all-in run manifest. Remove optional
   references first; reduce or defer the larger stage before weakening controls
   or replication. Document any change before confirmation.
5. Obtain explicit owner authorization for the bounded paid run.

Nothing in the estimator can start a job. A formally valid kernel, a working
trainer, semantic correctness and useful capability remain separate claims.
