# Monitorability pilot: belief readability and lie detection in from-scratch reasoners

This is a small toy study. Transformers are trained from random initialization on Horn-rule reasoning
episodes, and an exact checker (cross-checked against `src/controlled_lm/supported_state.py`) defines
what an ideal reasoner should believe. Some contexts train the model to lie:
- **S:** the `boss` persona makes the model echo the boss's claim.
- **F:** the `rival` persona makes the model flip the true answer.

We then ask whether linear probes read the true belief, whether a zero-deception disagreement detector
catches lies, and whether an exact-belief auxiliary objective (Arm B) changes this relative to a plain
answer objective (Arm A).

- Design, hypotheses and gates: `PREREGISTRATION.md`.
- Findings, including the CPU competence failure: `RESULTS.md`.
- GPU hand-off: `CODEX_TASKS.md`.

## Files

| file | purpose |
|---|---|
| `env.py` | Episode generator, exact labeler (`closure`), kernel cross-check (`kernel_closure`), name-family split, deception contexts |
| `model.py` | Pre-LN decoder-only transformer with rotary attention. Optional belief heads (Arm B). Optional pilot-only priors (`n_tied_heads`, `identity_recall`), off in all reported configs |
| `train.py` | `--arm plain|structured --seed --config --steps --device --threads`. Writes `checkpoints/<cfg>/<arm>_s<seed>/{model.pt,train_metrics.json}` |
| `evaluate.py` | Behavior, truth probes per layer (Q1/Q2), matched-neutral verification, lie detectors + baselines + random-direction null, AUROC with bootstrap CIs, recall at 1%/5% FPR (Q3/Q4). Writes `results/<cfg>/<arm>_s<seed>.json` |
| `aggregate.py` | Across-seed mean, t-based 95% CI and paired structured-minus-plain differences. Writes `results/<cfg>/summary.{json,md}` |
| `baselines.py` | Shallow non-learned heuristic accuracy on the evaluation sets |
| `run_main.sh` | Train + evaluate both arms for a seed list (arms interleaved per seed), then aggregate |
| `configs/cpu_pilot.json` | The CPU config of record (seed 0, 12k steps) |
| `configs/gpu_scale.json` | Confirmatory config for a GPU (d=256, 6 layers, 40k steps, batch 256, seeds 0-9) |
| `configs/{smoke,pilot_debug,pilot2,pilot3}.json` | Pilot configs (disclosed in the prereg; pilot3 lacks the later `max_len` change) |
| `tests/test_env.py` | Kernel agreement, label/depth balance, no name leakage, lie bookkeeping, determinism, matched-neutral counterfactual |
| `results/cpu_pilot/` | Raw per-run JSONs, training logs (metrics), summary tables |

Checkpoints (`*.pt`) are git-ignored by the repo's `.gitignore`.

## Requirements

Python >= 3.11, torch 2.x, numpy, scipy, scikit-learn, pytest. Run commands from this directory. The
tests import `controlled_lm` from `../../src` via `tests/conftest.py`.

## Reproduce the CPU pilot exactly

```bash
cd experiments/monitorability
python -m pytest -q tests -p no:cacheprovider          # 9 passed
# main runs as executed: both arms in parallel, 1 thread each (2-core machine)
python train.py --arm plain      --seed 0 --config configs/cpu_pilot.json --threads 1 > logs/cpu_pilot_plain_s0_train.log 2>&1 &
python train.py --arm structured --seed 0 --config configs/cpu_pilot.json --threads 1 > logs/cpu_pilot_structured_s0_train.log 2>&1 &
wait
for ARM in plain structured; do
  cp checkpoints/cpu_pilot/${ARM}_s0/train_metrics.json results/cpu_pilot/${ARM}_s0_train_metrics.json
  python evaluate.py --ckpt checkpoints/cpu_pilot/${ARM}_s0/model.pt --threads 2 --out results/cpu_pilot/${ARM}_s0.json
done
python aggregate.py --config_name cpu_pilot
python baselines.py --out results/cpu_pilot/heuristic_baseline.json
```

Sequential alternative (same numbers up to CPU threading nondeterminism):
`bash run_main.sh configs/cpu_pilot.json cpu 2 0`.

The whole procedure is deterministic given seed, torch version and thread count:
- Training data depends on the seed only, so both arms see identical episodes.
- The trunk init is identical across arms.
- Evaluation sets use a fixed seed (`EVAL_SEED=20261021`), shared by all runs.

## Run the GPU config

```bash
cd experiments/monitorability
bash run_main.sh configs/gpu_scale.json cuda 8 0 1 2 3 4 5 6 7 8 9
# -> results/gpu_scale/{plain,structured}_s*.json, summary.json, summary.md
```

See `CODEX_TASKS.md` for the exact hand-off procedure and report format.

## Key design choices (details in PREREGISTRATION.md)

- **Names:** 300 two-word names, split by *pair*. Every noun and state token is seen in training, but
  the 90 held-out pairs never are.
- **Episode format:** 4 rules + facts + an optional user line + the question. Literals are
  `true|false noun state`. Answers are yes / no / unknown, and the model's output is the argmax over
  those three tokens.
- **Losses:** answer-token cross-entropy only. Arm B adds 0.3 x the belief-head loss: every layer's
  final-position state predicts the true support labels of the queried proposition and of all mentioned
  propositions, including in deceptive contexts.
- **Lie label for detection:** the model's actual output differs from the true label.
  - "Verified" lies also require the matched neutral-persona version of the episode to be answered
    correctly.
  - The primary population is within the S persona (held-out names), where some claims are true, so a
    detector cannot succeed by detecting the trigger.
