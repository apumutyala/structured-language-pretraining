"""End-to-end CPU smoke test of the replication package in a throwaway repo copy.

Copies src/, tools/, configs/, data/source-spec.json and this package into
--workdir (nothing is written into the real checkout), provides a corpus, then
runs, inside the copy:

  A. run_seeds.py on the UNCHANGED cpu_enablement config (real validator path,
     1 update/arm) for replicates 1,2;
  B. run_seeds.py --cpu-smoke with a 5-update smoke config for replicates 0,1;
  C. eval_expanded.py --device cpu on both tags;
  D. analyze.py on both tags;

and asserts the pairing invariants (same init/batch digests across arms within
a replicate, different across replicates; replicate 0 reuses base config bytes).

Corpus: --synthetic writes a tiny generated corpus + manifest (for machines
without the data); --real-data copies the extracted SimpleStories split files
and results/language_pilot_extraction.json into the copy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REAL_ROOT = HERE.parents[1]
PACKAGE = HERE.relative_to(REAL_ROOT)


def make_copy(workdir):
    root = workdir / "repo"
    if root.exists():
        raise FileExistsError(f"{root} exists; choose an empty --workdir")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "runs", "results", ".pytest_cache")
    for name in ("src", "tools", "configs"):
        shutil.copytree(REAL_ROOT / name, root / name, ignore=ignore)
    (root / "data").mkdir(parents=True)
    shutil.copy2(REAL_ROOT / "data/source-spec.json", root / "data/source-spec.json")
    shutil.copytree(HERE, root / PACKAGE, ignore=ignore)
    (root / "results").mkdir()
    if (REAL_ROOT / "results/normalized_language_local_20261021.derived.json").exists():
        shutil.copy2(REAL_ROOT / "results/normalized_language_local_20261021.derived.json", root / "results")
    return root


def synthetic_corpus(root, train_docs=300, dev_docs=40, seed=7):
    rng = random.Random(seed)
    names = ["Mia", "Tom", "Lily", "Ben", "Zoe", "Max", "Ava", "Sam"]
    things = ["a red ball", "a small boat", "an old map", "a blue kite", "a shiny stone", "a tiny bird"]
    places = ["the park", "the river", "the hill", "the garden", "the forest", "the beach"]
    feelings = ["happy", "curious", "brave", "sleepy", "excited", "calm"]

    def story():
        sentences = [f"Once upon a time, {rng.choice(names)} found {rng.choice(things)} near {rng.choice(places)}."]
        for _ in range(rng.randint(0, 9)):
            sentences.append(f"{rng.choice(names)} felt {rng.choice(feelings)} and walked to {rng.choice(places)}"
                             f" with {rng.choice(things)}.")
        return " ".join(sentences)

    folder = root / "data/language/synthetic"
    folder.mkdir(parents=True)
    splits = {}
    for split, count in (("train", train_docs), ("dev", dev_docs)):
        path = folder / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for index in range(count):
                stream.write(json.dumps({"source_row": index, "story": story()}) + "\n")
        splits[split] = {"path": path.relative_to(root).as_posix(), "documents": count,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"status": "SYNTHETIC_SMOKE_CORPUS_NOT_SIMPLESTORIES", "splits": splits}
    (root / "results/language_pilot_extraction.json").write_text(json.dumps(manifest, indent=2) + "\n")


def link_real_corpus(root):
    manifest = REAL_ROOT / "results/language_pilot_extraction.json"
    if not manifest.exists():
        raise SystemExit("no extracted corpus; run the data prep in docs/reproduction.md first")
    # Copy only the two split files the manifest names (~28 MB); no symlinks (Windows-safe).
    for split in json.loads(manifest.read_text())["splits"].values():
        target = root / split["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REAL_ROOT / split["path"], target)
    shutil.copy2(manifest, root / "results/language_pilot_extraction.json")


def write_smoke_config(root, updates):
    config = json.loads((root / "configs/normalized_language_local_20261021.json").read_text())
    config.update(pilot_stage="replication_cpu_smoke", updates=updates, checkpoints=[0, updates],
                  dev_documents=3, dev_world_batches=2, sample_new_bytes=4, max_wall_seconds=600)
    path = root / PACKAGE / "smoke_base_config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def run(root, *args):
    command = [sys.executable, *map(str, args)]
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def check_pairing(root, tag, replicates, base_config):
    reports = {i: json.loads((root / PACKAGE / "runs" / tag / f"seed_{i:02d}.json").read_text()) for i in replicates}
    inits, batches = {}, {}
    for index, report in reports.items():
        assert report["status"] == "complete", (tag, index, report.get("error"))
        init = {run["initial_parameters_sha256"] for run in report["runs"]}
        batch = {run["trained_batch_sha256"] for run in report["runs"]}
        assert len(report["runs"]) == 3 and len(init) == 1 and len(batch) == 1, "arms not paired within replicate"
        inits[index], batches[index] = init.pop(), batch.pop()
    assert len(set(inits.values())) == len(inits), "different replicates share an initialization"
    assert len(set(batches.values())) == len(batches), "different replicates share a data stream"
    if 0 in reports:
        assert reports[0]["config_sha256"] == hashlib.sha256(base_config.read_bytes()).hexdigest()
    return {"tag": tag, "initial_parameters_sha256": inits, "trained_batch_sha256": batches}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    corpus = parser.add_mutually_exclusive_group(required=True)
    corpus.add_argument("--synthetic", action="store_true")
    corpus.add_argument("--real-data", action="store_true")
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--max-dev-docs", type=int, default=None, help="truncate DEV docs in eval (speed)")
    parser.add_argument("--max-questions", type=int, default=None, help="truncate world questions in eval (speed)")
    args = parser.parse_args()
    workdir = (args.workdir or Path(tempfile.mkdtemp(prefix="routing_smoke_"))).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    root = make_copy(workdir)
    synthetic_corpus(root) if args.synthetic else link_real_corpus(root)
    smoke_config = write_smoke_config(root, args.updates)
    package = root / PACKAGE
    eval_extra = (["--max-dev-docs", str(args.max_dev_docs)] if args.max_dev_docs else []) + \
        (["--max-questions", str(args.max_questions)] if args.max_questions else [])
    enable_cfg = root / "configs/normalized_language_cpu_enablement_20261021.json"

    # A: unchanged validator path (cpu_enablement stage, 1 update), fresh seeds 1,2.
    run(root, package / "run_seeds.py", "--base-config", "configs/normalized_language_cpu_enablement_20261021.json",
        "--device", "cpu", "--replicates", "1,2", "--tag", "smoke-enablement", "--skip-preflight")
    # B: 5-update CPU smoke stage, replicates 0 (base seeds) and 1.
    run(root, package / "run_seeds.py", "--base-config", smoke_config.relative_to(root), "--device", "cpu",
        "--replicates", "0,1", "--tag", "smoke-updates", "--cpu-smoke")
    # Rerun must skip complete replicates and never overwrite.
    run(root, package / "run_seeds.py", "--base-config", smoke_config.relative_to(root), "--device", "cpu",
        "--replicates", "0,1", "--tag", "smoke-updates", "--cpu-smoke")
    checks = [check_pairing(root, "smoke-enablement", (1, 2), enable_cfg),
              check_pairing(root, "smoke-updates", (0, 1), smoke_config)]
    for tag in ("smoke-enablement", "smoke-updates"):
        run(root, package / "eval_expanded.py", "--tag", tag, "--device", "cpu", *eval_extra)
        run(root, package / "analyze.py", "--tag", tag, "--include-original", "--bootstrap", "2000")
    for tag in ("smoke-enablement", "smoke-updates"):
        results = package / "results" / tag
        text = "".join(path.read_text() for path in results.glob("*.json"))
        assert str(workdir) not in text, f"absolute host path leaked into results/{tag}"
    print(json.dumps({"smoke": "PASSED", "workdir": str(workdir), "pairing": checks}, indent=1))


if __name__ == "__main__":
    main()
