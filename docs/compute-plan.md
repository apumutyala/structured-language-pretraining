# Conditional compute plan

The small CPU demo/tests and local development pilots are distinct from a
proposed rented H100 study. Staff costs are already covered; these are gross
compute/research ceilings, not approved execution or measured resource needs.
No paid launch follows from this document. Apply usable credits once and count
all devices, storage, transfer, taxes, checkpoint/evaluation overhead, tuning,
failed jobs and compute-attached services. Do not allocate paid agent/API work
under these lines.

| Compute/research allocation | Base ceiling |
| --- | ---: |
| Qualification and equal-recipe tuning | $2,000 |
| 125M/350M discovery | $4,000 |
| Conditional roughly 1B main comparison | $8,000 |
| Adversarial/intervention/continued initial-learning evaluation | $2,000 |
| Contingency and compute-attached services | $2,000 |
| Base total | $18,000 |
| Optional replication or justified scale | $2,000 |
| Maximum planning ceiling | $20,000 |

Before choosing instance counts or run counts, measure representative runtime,
peak memory, utilization, total token throughput and complete evaluation costs
on the intended H100 stack. Existing local-device results do not predict that
stack. The six roughly 1B runs and optional reference runs are not guaranteed
to fit; remove optional work first, then reduce or omit scale according to the
frozen decision rule. Replan the requested resources if measurements justify a
smaller study. No existing script bypasses the paid-launch gate.

The following **UNMEASURED H100 scenarios** use the rough training convention
`C = 6 * N * D`, where N is parameters and D is training tokens, and an
illustrative $4 per GPU-hour. The coefficient 6 is a FLOP accounting convention,
not a TFLOP/s scenario. Effective rates of 50/100/200 TFLOP/s are hypothetical;
no advertised peak or measured H100 rate is asserted. Core FLOP costs omit
nontraining overhead and are not rental quotes or all-in budgets.

| Illustrative scope | Core FLOPs | 50 TFLOP/s | 100 TFLOP/s | 200 TFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| Primary 24 discovery + 6 conditional roughly 1B runs | 9.189e20 | $20,420 | $10,210 | $5,105 |
| Same scope + 12 optional Transformer references | 1.01835e21 | $22,630 | $11,315 | $5,657.50 |

These figures show why a planning ceiling cannot promise every tier. The
per-run parameter/token schedule and actual activation/optimizer memory still
need freezing before a paid experiment; aggregate FLOP illustrations alone are
not an executable manifest. One H100 versus multi-H100 selection must follow
measurement, not this arithmetic.

Paid launch requires a fixed scientific comparison and data split, valid
correctness/safety/control checks, an independently reviewed exact runnable
recipe, representative runtime/memory measurements, an all-in spend manifest
and explicit owner authorization. An acknowledged spend record is required by
the research workflow. This candidate does not grant authorization. Existing
CUDA scripts are deliberately qualified only for the local laptop device;
H100 enablement and admission remain future work.
