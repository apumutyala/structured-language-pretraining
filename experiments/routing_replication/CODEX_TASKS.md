# Codex task: 10-seed routing replication on the local RTX 5070 Ti Laptop GPU

You will train the three routing arms (none, internal, output-only) for 11
seed pairs, evaluate them, analyze them, and commit the redacted results.
Read `PREREGISTRATION.md` first. Do not change the decision rule, the
endpoints or the seeds after you have seen any result.

Ground rules:
- Work on branch `monitorability-pilot`.
- Write new files only under `experiments/routing_replication/`.
- Do not edit `tools/`, `src/`, `configs/` or `experiments/monitorability/`.
- Never delete or overwrite a run. Every script refuses to overwrite. If
  something fails, keep the failed artifacts and rerun with a new `--tag`.
- Commit ONLY `experiments/routing_replication/results/<tag>/`. Never commit
  `runs/`, which holds raw reports with host paths and prose samples, plus
  about 15 GB of checkpoints. It is git-ignored.

All commands run from the repository root. They are shown in POSIX shell
form; PowerShell differences are noted where they matter.

## 0. Check the device first (blocker if it fails)

The bound trainer admits CUDA **only** when
`torch.cuda.get_device_name() == "NVIDIA GeForce RTX 5070 Ti Laptop GPU"`.

```sh
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

If the name differs (a desktop 5070 Ti reports `NVIDIA GeForce RTX 5070 Ti`),
**stop and report back**. Do not patch the trainer or bypass the check.

## 1. Environment

The best option is the Python environment that ran the original normalized
pilot. Activate it and record the output of the command in step 0.

Otherwise, create a fresh one with Python 3.12. Blackwell needs a CUDA 12.8+
build of torch:

```sh
python -m venv .venv
# POSIX: . .venv/bin/activate        PowerShell: .venv\Scripts\Activate.ps1
python -m pip install numpy==2.5.4 pytest==9.1.1
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130   # or .../cu128
python -m pip install -r environment/requirements-data.txt   # pyarrow==26.0.0, only needed for extraction
```

If torch 2.14.1 is unavailable for your CUDA, use the closest CUDA build and
report the exact `torch.__version__`. The original receipts do not publish the
torch build that was used. The replicate-0 reproduction check (step 6) shows
whether results are bit-identical.

## 2. Data

These commands are verified to exist in `tools/` and to match
`docs/reproduction.md`. Skip download and extraction if
`data/language/simplestories/pilot-v1/` already exists, because the extractor
refuses to overwrite it. Always run the verifier.

```sh
python tools/download_language_source.py              # prints the plan only
python tools/download_language_source.py --download   # ~255 MB, hash-pinned
python tools/extract_language_pilot.py                 # -> results/language_pilot_extraction.json
python tools/verify_language_extraction.py             # must print train_dev_sha256_verified: true
```

Expected: 16,849 TRAIN and 819 DEV documents. DEV is 1,048,472 text bytes,
with sha256 `bb7cb481e0bf1b3b25aef640fe077b2b7225d4096a586450e2efdff9d1b18198`.

## 3. Package self-checks (about 1 minute, CPU)

```sh
python -m pytest experiments/routing_replication/tests -q -p no:cacheprovider
```

Optional CPU preflight on the real corpus. It takes about 5–10 minutes and
writes only to a temporary copy of the repository:

```sh
python experiments/routing_replication/cpu_smoke.py --real-data --max-dev-docs 12 --max-questions 64
```

## 4. Train: 11 replicates × 3 arms, 1,000 updates each

```sh
python experiments/routing_replication/run_seeds.py --tag main --seeds 10 --keep-going --dry-run   # preflight + plan
python experiments/routing_replication/run_seeds.py --tag main --seeds 10 --keep-going
```

**What runs.**
- Replicate 0 uses the original seeds, as a reproduction check.
- Replicates 1–10 are the preregistered sample. Each uses
  `seed = 2026102131 + 100k` and `data_seed = 2026102133 + 100k`.
- Each replicate is one invocation of the bound trainer. Within it, all three
  arms share the same initialization and batch stream, and the trainer asserts
  this.

**Expected wall time.** The original 3-arm run took 192.9 s. Allow about 200 s
per replicate, so roughly **35–45 min** in total. Each invocation is also
capped at 600 s by the trainer.

**Disk.** About 1.4 GB per replicate, roughly 15 GB in total. If free space is
short, add `--prune-intermediate-checkpoints`. That deletes the step
50/200/500 checkpoints after each replicate completes, keeps the step 0 and
1000 checkpoints, and records the digests of the pruned files.

**If interrupted.** Rerun the same command. Complete replicates are skipped.
A replicate whose report is not `complete` makes the script stop with an
explanation. Never delete that report. Run any remaining replicates under a
new tag, for example `--tag main-b --replicates 7,8,9,10`, and report this.
You then have to merge the tags manually, so ask the lead first.

**If a replicate fails** (non-finite loss, timeout, OOM), `--keep-going`
continues with the rest. Report the failure as it is (see PREREGISTRATION §6).
Do not replace it unless the lead says so.

## 5. Evaluate (about 5–10 min on GPU)

```sh
python experiments/routing_replication/eval_expanded.py --tag main --device cuda
```

**What it does.** For each of the 33 step-1000 checkpoints, after verifying
the checkpoint digests, it scores:
- 1,552 held-out world questions: every DEV-fold ledger except the 16 used in
  the trainer's in-run DEV batches, times 4 queries;
- all 819 DEV stories under two protocols;
- a replay of the trainer's own 12-story/8-question DEV evaluation.

**Abort conditions.** The script aborts if any evaluated world ledger appears
in any replicate's training worlds, or if the rebuilt in-run DEV batches do
not match the report digest.

**Output.** `experiments/routing_replication/results/main/eval_expanded.json`.

## 6. Analyze (seconds)

```sh
python experiments/routing_replication/analyze.py --tag main
```

**Output.**
- `results/main/analysis.md` and `results/main/analysis.json`.
- Primary analysis is replicates 1–10, using the Holm-corrected paired t-test
  with bootstrap CIs, sign test and sign-flip test.
- A "Replicate 0 vs original receipt" table.

## 7. Report back and commit

1. Paste the full contents of `results/main/analysis.md` into your reply, and
   attach (or link) `results/main/analysis.json`.
2. Also state each of the following:
   - the output of step 0 (torch version, CUDA version, device name);
   - total training wall time;
   - any failed replicates;
   - whether replicate 0's digests matched the original receipt;
   - the maximum in-run replay difference shown in `analysis.md`;
   - any deviation from this file. If there are deviations, also write them to
     `results/main/DEVIATIONS.md`.
3. Commit only the redacted results, as Apuroop Mutyala
   <apumutyala@gmail.com>, with no AI co-author trailers:

   ```sh
   git status --short experiments/routing_replication   # runs/ must NOT appear
   grep -rIl -e "$HOME" -e "C:\\\\Users" experiments/routing_replication/results || echo "no host paths"
   git add experiments/routing_replication/results/main
   git -c user.name="Apuroop Mutyala" -c user.email="apumutyala@gmail.com" commit -m "Add 10-seed routing replication results (tag main)"
   git push origin monitorability-pilot
   ```

   If `git add` silently stages nothing, a local exclude rule is probably
   hiding the directory. One such rule, `experiments/` in `.git/info/exclude`,
   was seen in another clone. Check with
   `git check-ignore -v experiments/routing_replication/results/main/analysis.md`.
   If needed, add only that directory with `git add -f`. Do not force-add
   `runs/`.

   `results/main/` should contain:
   - `train_summary_seed_00..10.json`
   - `run_log_*.json`
   - `eval_expanded.json`
   - `analysis.json`
   - `analysis.md`
   - `DEVIATIONS.md`, if there are deviations

## Things to double-check and report if they differ

- **Device name.** `torch.cuda.get_device_name()` must equal
  `NVIDIA GeForce RTX 5070 Ti Laptop GPU`. Otherwise this is a hard stop.
- **Bound sources.** The trainer refuses to run if the SHA-256 of
  `tools/train_language_pilot.py`, `src/controlled_lm/typed_language_model.py`
  or `tools/qualify_language_gpu.py` changed. Do not "fix" them.
- **Replicate 0 reproduction.** The original CUDA receipts give the
  initial-parameter digest `b8b75220…`. The digest depends on the torch build
  and device: the CPU receipt and a CPU check on another machine both gave
  different values. The batch digest should be device-independent:
  `c939d2fe…`. The final DEV bpb values should be close to
  1.818783 / 1.865863 / 1.819111. A mismatch does not block the study but must
  be reported, together with the torch/CUDA versions.
- **Timing.** If a replicate takes more than about 300 s, report it. The
  per-invocation cap is 600 s.
