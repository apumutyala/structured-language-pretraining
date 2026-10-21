# Reproduction

Use Python 3.12. The tested CPU environment used Python 3.12.14,
PyTorch 2.14.1+cpu, NumPy 2.5.4 and pytest 9.1.1. Exact package pins record that
environment; they are not a cross-platform availability guarantee. If a pin
cannot be installed, report the blocker rather than silently substituting a
runtime and claiming exact reproduction.

```sh
python -m venv .venv
# Activate this environment using your shell's standard activation command.
python -m pip install -r environment/requirements-cpu.txt
python tools/check_cpu.py
```

The check command sets PYTHONPATH to this checkout, disables visible CUDA
devices, runs both demos and selected CPU tests. Tests use finite enumeration,
small forward/backward calculations and no optimizer updates. The copied
language-model test file deliberately omits its historical optimizer-mutating
test; the omission and original file digest are documented in PROVENANCE.
The optional Lean smoke skips if its executable is absent. No corpus body,
checkpoint, downloaded dependency binary or inherited parameter is included.

## Regenerate pinned data (optional network and disk work)

```sh
python -m pip install -r environment/requirements-data.txt
python tools/download_language_source.py
python tools/download_language_source.py --download
python tools/extract_language_pilot.py
python tools/verify_language_extraction.py
```

Without `--download`, the helper only prints the bounded plan. The explicit
acquisition downloads one pinned TRAIN shard (238,015,938 bytes), official TEST
(16,838,557 bytes) and a hash-verified bounded dataset card. Their revision and
SHA-256 values are in `data/source-spec.json`; no mutable latest branch is used.
No corpus payload is committed. The official TEST source is read only for
completion-group/exact-normalization exclusion hashes, never model training,
text display or model scoring. Source stories are not trusted facts.

The unchanged extractor produces 16,849 TRAIN and 819 development documents.
Expected JSONL hashes are respectively
`6719423cdfd5c49ecaff235f1eec33adcf11119bd5da80ca8d2b94e38f853860`
and `bb7cb481e0bf1b3b25aef640fe077b2b7225d4096a586450e2efdff9d1b18198`.
It groups completion siblings, unions normalized exact duplicates, excludes
TEST-related groups/hashes, and uses fixed hash folds under 20 MiB/1 MiB text
caps. Only the first TRAIN shard is used; siblings in unused shards and
near-duplicate/template contamination are not covered. Install the pinned
PyArrow 26.0.0 for the historical extraction path. The original extractor is
preserved byte-for-byte: manifest path separators are native to the platform.
Regenerate the manifest on that same platform rather than copying a manifest
containing foreign separators. Split JSONL bytes must still pass the pinned
hash check. Candidate-root regeneration succeeded using preexisting, hash-verified
source Parquet/card bytes and PyArrow 26.0.0 in an isolated environment. The
unchanged helper's existing-file verification path, extractor and split verifier
ran successfully: 16,849 TRAIN and 819 development documents reproduced both
exact JSONL hashes. The HTTP download path was not exercised. A fresh full
CPU dependency installation was not performed; only isolated PyArrow was
installed for this recheck.

## Optional initial-training recipe

After data verification, the retained normalized wrapper can run a bounded
one-update CPU enablement recipe:

```sh
PYTHONPATH=src python tools/train_normalized_language_pilot.py --config configs/normalized_language_cpu_enablement_20261021.json --output results/my_cpu_enablement.json --device cpu
```

The command above uses POSIX shell environment syntax; in PowerShell first set
`$env:PYTHONPATH="src"` and then run the `python ...` portion. This optional
command performs optimizer updates and is not part of the default CPU checks.
This public-source CPU recipe completed one update in each of the three arms
on the regenerated data in 18.47654 seconds, with 8,703,788 parameters per arm;
source and data hashes remained unchanged. It starts fresh and refuses existing
outputs. See `results/public_candidate_runtime_recheck.derived.json` for the
source-bound sanitized receipt. The two normalized CUDA configs preserve the original qualified
architecture/source bindings; they are not permission to use a GPU or a claim
that the 1,000-update wall cap is currently admitted. The CUDA trainer explicitly
rejects devices other than its historically qualified laptop GPU. H100 code,
precision/resource qualification and a spend-approved run manifest remain
future work. No automatic training command is run by CI.

## Lean 4.8.0 (optional CPU toolchain)

Install Lean 4.8.0 through its standard distribution/toolchain manager and
confirm `lean --version`. The repository does not bundle Lean binaries. From
this checkout:

```sh
python tools/build_lean.py --lean lean
```

This freshly builds `SoundMemory`, its review examples, the three-bit
`WorkflowKernel` and its CLI into ignored `.build/lean`. To run the retained
bounded differential tool, set `WORKFLOW_LEAN` to the absolute executable path
in your own environment, then inspect `python tools/check_workflow_kernel_lean.py --help`
and choose a new output path. The original tool supports that override; its
fallback is a historical relative Windows installation layout and is not a
portable default. The persistent `LeanWorkflowSession` also accepts an
explicit `lean_executable` path. Do not assume PATH discovery in that original
adapter. Authentication and real side effects are outside the Lean CLI.

The fresh candidate was also checked with an explicitly supplied Lean 4.8.0
executable: all four modules compiled and the actual-Lean workflow smoke passed.
An independent candidate-root differential recheck covered 4,792 transitions,
952 assertions and 40 malformed-protocol cases. The default CPU command still
skips its optional smoke when no executable is configured. These candidate
checks and historical evidence are reported separately in results. No proof transfers to the Python
RECORD implementation or trained model.
