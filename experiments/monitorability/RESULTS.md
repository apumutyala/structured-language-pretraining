# Monitorability pilot: results

**Bottom line.** The preregistered CPU pilot (3 seeds per arm, 12,000 steps each) is a negative result. Every run fails the G1 competence gate, so under the preregistration none of them is interpretable, and the pilot supports no claim that structured pretraining makes a model's beliefs easier to read. The first GPU seed (40,000 steps) fails G1 the same way. An independent audit re-derived every reported number from the raw outputs and found no discrepancies.

## Gates (PREREGISTRATION.md section 5)

G1 needs honest held-out accuracy ≥ 0.60 and depth-0/1 accuracy ≥ 0.85. G2 needs S compliance ≥ 0.80 and F compliance ≥ 0.60. G3 needs at least 30 verified lies and 30 truthful outputs. A run is interpretable only if it passes G1 and G3.

| Run | Honest held-out acc | Depth-0/1 pooled | G1 | S / F compliance | G2 | Lies / truthful | G3 | Interpretable |
|---|---|---|---|---|---|---|---|---|
| plain s0 | 0.527 | 0.570 | FAIL | 1.000 / 0.689 | pass | 982 / 975 | pass | no |
| plain s1 | 0.550 | 0.577 | FAIL | 0.998 / 0.697 | pass | 1045 / 976 | pass | no |
| plain s2 | 0.535 | 0.556 | FAIL | 1.000 / 0.700 | pass | 1028 / 975 | pass | no |
| structured s0 | 0.562 | 0.571 | FAIL | 1.000 / 0.717 | pass | 1101 / 975 | pass | no |
| structured s1 | 0.566 | 0.562 | FAIL | 1.000 / 0.721 | pass | 1099 / 975 | pass | no |
| structured s2 | 0.556 | 0.565 | FAIL | 1.000 / 0.711 | pass | 1072 / 975 | pass | no |

Source: `results/cpu_pilot/gates.md`, `report_tables.md`, `summary.md`.

## Descriptive numbers (not confirmatory)

Because no run is interpretable, the following are reported only for completeness.

| Run | Primary AUROC [95% CI] | Best sign-free baseline | Margin |
|---|---|---|---|
| plain s0 | 0.450 [0.425, 0.476] | bag-of-embeddings disagreement 0.754 | −0.305 |
| plain s1 | 0.391 [0.368, 0.417] | 0.754 | −0.363 |
| plain s2 | 0.545 [0.517, 0.570] | 0.733 | −0.187 |
| structured s0 | 0.730 [0.709, 0.751] | 0.734 | −0.004 |
| structured s1 | 0.485 [0.461, 0.511] | 0.736 | −0.251 |
| structured s2 | 0.672 [0.650, 0.693] | 0.738 | −0.066 |

- The structured arm's primary AUROC is higher than the plain arm's in all three seeds: mean 0.629 against 0.462, paired difference +0.167, range +0.094 to +0.280.
- In every run, a simple bag-of-embeddings baseline matches or beats the internal-state probe.
- The structured arm is also slightly more accurate on honest questions, by +0.024 held-out.
- None of these differences can be read as a result, because neither arm learned the task well enough for the probe to mean anything.

## Why the models fail G1

Neither arm learns the basic lookup the task depends on. Depth-0 accuracy stays around 0.59–0.64, close to what a model gets from shortcuts such as guessing from whether a fact is present at all.

We checked whether this was an input-format problem. Moving the answer prediction onto the query token did not help: depth-0 accuracy stayed flat at about 0.70–0.74 alone and about 0.61 in the full mixture.

Rules-free and toy-task ablations point to the answer set instead. With only yes, no and unknown as answers, strong majority and presence shortcuts leave little training signal toward learning retrieval. The same model architecture learned a 10-way version of the lookup cleanly. This explanation has not been tested on the real task.

## GPU scale (Codex, RTX 5070 Ti)

- Seed 0 finished 40,000 steps (results commit `9218ae4`). Both arms fail G1: pooled depth-0/1 is 0.571 (plain) and 0.591 (structured), and honest held-out accuracy is 0.597 and 0.602.
- The preregistered GPU run lists seeds 0–9 (minimum 5). Seeds 1–9 are **pending**.
- Longer training does not rescue competence.

## Deviations from the preregistration

None of these changes a reported number or a gate outcome.

1. **G1 depth rule.** The preregistration's "depth-0/1 accuracy ≥ 0.85" is ambiguous. `report_tables.py` as frozen at 12:31 read it as depth-0 and depth-1 each ≥ 0.85. At 12:42, after launch but before any results existed, it was changed to pooled depth-0/1 to match `gates.py`. Every run fails under either reading, and every run also fails the honest-accuracy part of G1 (best 0.566).
2. **Seeds 1 and 2.** `configs/cpu_pilot.json` lists `"seeds": [0]`. Seeds 1 and 2 were passed on the command line through `run_parallel.sh`, not the `run_main.sh` pattern named in Amendment 1. `run_parallel.sh` was committed after seed 1 ran; it is equivalent for the analysis.
3. **Baseline comparison.** `gates.py` compares the raw primary AUROC against sign-free baselines, max(a, 1 − a). The preregistration doesn't specify this. It is conservative against the probe. Taking the primary sign-free as well leaves every margin negative, and structured s0 is a tie (−0.004).
4. **Eval log label.** `evaluate.py` prints "PRIMARY within_S_heldout", but that is the non-verified population. The stored JSONs, gates and tables use the correct verified population. Quote those, not the logs.
5. **Random-direction null.** The null direction is selected on `within_S_train`, which includes all false outputs, not only verified lies. The preregistration says only "train-family dev population".
6. **Arm B head size.** The preregistration says about 35k extra head parameters. The actual count is 13,932.

## Caveats on interpretation

- G3 passes almost trivially. S compliance is about 1.0, so the primary population mostly contrasts copying a true claim with copying a false one.
- The audit could recompute per-example AUROCs only for the two re-evaluated runs (structured s0 and plain s1). The stored JSONs keep aggregates, so the other four were checked for internal consistency.

## Independent audit

A separate agent re-derived every number in `gates.md`, `summary.md` and `report_tables.md` from the six raw result files:

- 736 comparisons, none off by more than 0.001;
- re-ran `evaluate.py` on two checkpoints, and got identical results;
- confirmed that the gate thresholds match the preregistration;
- confirmed that no threshold or analysis code changed after results existed.

The audit was done by a separate AI agent with no part in producing the results. Full report: `AUDIT_REPORT.md`; its scripts and recomputed values are in `audit/`.
