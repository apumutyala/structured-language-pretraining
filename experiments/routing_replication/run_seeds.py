"""Train none/internal/output_only for N paired seeds with the retained normalized trainer.

Each replicate is ONE invocation of the bound trainer (via seeded_train.py), so
within a replicate the three arms share the trainer's own guarantees: identical
torch seed -> identical initial parameters (asserted by the trainer), identical
data_seed -> identical batch stream (asserted by batch digests), 1,000 updates,
frozen config. Across replicates only seed/data_seed change:

  replicate 0  : original seeds (2026102131 / 2026102133), base config bytes reused
  replicate k>0: seed = 2026102131 + 100k, data_seed = 2026102133 + 100k

Outputs (never overwritten; an existing complete replicate is skipped):
  runs/<tag>/configs/seed_XX.json         replicate config
  runs/<tag>/seed_XX.json                 raw trainer report (host paths, samples; git-ignored)
  runs/<tag>/seed_XX_checkpoints/*.pt     checkpoints (git-ignored)
  results/<tag>/train_summary_seed_XX.json  redacted, path-free per-run summary (commit this)
  results/<tag>/run_log.json              per-invocation log (appended, new file per launch)
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DEFAULT_BASE_CONFIG, HERE, QUALIFIED_DEVICE, RESULTS_DIR, ROOT, RUNS_DIR,  # noqa: E402
                    check_replication_config, rel, replicate_config_bytes, replicate_name,
                    seeds_for_replicate, sha256_file, write_json_new)

CHECKPOINT_BYTES_ESTIMATE = 105 * 1024 ** 2   # model + Adam state for 8.7M fp32 params
DEV_METRIC_KEYS = ("story_mean_nll", "story_bits_per_utf8_byte", "story_target_tokens", "story_byte_targets",
                   "world_rows", "world_query_byte_accuracy", "world_raw41_accuracy",
                   "grounded_requested_key_claim_accuracy", "grounded_nonabstention_coverage",
                   "max_forbidden_probability")


def preflight(args, base):
    problems = []
    if args.device == "cuda":
        try:
            import torch
            name = torch.cuda.get_device_name() if torch.cuda.is_available() else None
        except Exception as error:  # pragma: no cover - environment specific
            name = f"<error {error}>"
        if name != QUALIFIED_DEVICE:
            problems.append(f"bound trainer only admits CUDA device {QUALIFIED_DEVICE!r}; found {name!r}")
    manifest_path = ROOT / base["manifest"]
    if not manifest_path.exists():
        problems.append(f"missing {base['manifest']}; run the data prep in docs/reproduction.md")
    elif not args.cpu_smoke:
        spec = json.loads((ROOT / "data/source-spec.json").read_text())
        manifest = json.loads(manifest_path.read_text())
        for split, expected in spec["expected_splits"].items():
            path = ROOT / manifest["splits"][split]["path"]
            if not path.exists() or sha256_file(path) != expected["sha256"]:
                problems.append(f"{split} split missing or hash differs from data/source-spec.json")
    per_replicate = len(base["checkpoints"]) * len(base["semantic_modes"]) * CHECKPOINT_BYTES_ESTIMATE
    if args.prune_intermediate_checkpoints:
        per_replicate = 2 * len(base["semantic_modes"]) * CHECKPOINT_BYTES_ESTIMATE
    needed = per_replicate * len(args.replicate_list)
    free = shutil.disk_usage(ROOT).free
    if free < needed * 1.1:
        problems.append(f"disk: ~{needed / 1024 ** 3:.1f} GiB needed for checkpoints, {free / 1024 ** 3:.1f} GiB free "
                        "(consider --prune-intermediate-checkpoints)")
    return problems, needed


def summarize(report_path, replicate, config_path, base_config_path):
    """Redacted per-run summary: metrics, digests, relative paths; no prose samples or host paths."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = report["config"]
    runs = []
    for run in report["runs"]:
        curve = []
        for point in run["curve"]:
            curve.append({"step": point["step"], "used_tokens": point["used_tokens"],
                          "utf8_byte_targets": point["utf8_byte_targets"], "eos_targets": point["eos_targets"],
                          "batch_sha256": point["batch_sha256"], "checkpoint_sha256": point["checkpoint_sha256"],
                          "checkpoint_file": f"{report_path.stem}_checkpoints/{run['mode']}_{point['step']:04d}.pt",
                          "last_training_loss": point.get("last_training_loss"),
                          "dev": {key: point["dev"][key] for key in DEV_METRIC_KEYS} if "dev" in point else None})
        runs.append({"mode": run["mode"], "status": run["status"], "parameters": run["parameters"],
                     "semantic_active_parameters": run["semantic_active_parameters"],
                     "initial_parameters_sha256": run["initial_parameters_sha256"],
                     "trained_batch_sha256": run.get("trained_batch_sha256"),
                     "cumulative_updates": run.get("cumulative_updates"),
                     "cumulative_used_tokens": run.get("cumulative_used_tokens"),
                     "arm_wall_seconds": run.get("new_wall_seconds"),
                     "source_unchanged": run.get("source_unchanged"), "data_unchanged": run.get("data_unchanged"),
                     "training_losses_at_probes": [{"before_update": probe["before_update"],
                                                    "loss": probe["loss_before_step"]}
                                                   for probe in run.get("training_probes", [])],
                     "curve": curve})
    audit = report["world_audit"]
    return {"artifact_kind": "ROUTING_REPLICATION_REDACTED_RUN_SUMMARY",
            "transformations": ["numeric metrics, digests and relative file names only",
                                "generated prose samples, prompts, absolute host paths and checkpoint payloads omitted"],
            "replicate": replicate, "seed": config["seed"], "data_seed": config["data_seed"],
            "dev_seed": config["dev_seed"], "pilot_stage": config["pilot_stage"], "updates": config["updates"],
            "raw_report_file": rel(report_path), "raw_report_sha256": sha256_file(report_path),
            "config_file": rel(config_path), "config_sha256": report["config_sha256"],
            "base_config_file": rel(base_config_path), "base_config_sha256": sha256_file(base_config_path),
            "status": report["status"], "error_type": report.get("error_type"), "error": report.get("error"),
            "device": report["device"], "device_name": report["device_name"], "torch_version": report["torch_version"],
            "precision": report["precision"], "wall_seconds": report.get("wall_seconds"),
            "peak_allocated_bytes": report.get("peak_allocated_bytes"),
            "source_unchanged": report.get("source_unchanged"), "data_unchanged": report.get("data_unchanged"),
            "source_sha256": report["source_sha256"], "data_sha256": report["data_sha256"],
            "world_audit": {"train_world_sequence_sha256": audit["train_world_sequence_sha256"],
                            "distinct_train_world_identities": len(audit["train_world_identities"]),
                            "dev_world_identities": audit["dev_world_identities"],
                            "train_world_folds": audit["train_world_folds"], "overlap": audit["overlap"],
                            "policy": audit["policy"]},
            "dev_selection": {key: value for key, value in report["dev_selection"].items()},
            "runs": runs}


def prune(report_path, base_config):
    """Delete intermediate checkpoints (keep step 0 and final), recording their digests."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    keep = {0, report["config"]["updates"]}
    removed = []
    for run in report["runs"]:
        for point in run["curve"]:
            if point["step"] in keep:
                continue
            path = report_path.with_name(report_path.stem + "_checkpoints") / f"{run['mode']}_{point['step']:04d}.pt"
            if path.exists():
                if sha256_file(path) != point["checkpoint_sha256"]:
                    raise AssertionError(f"checkpoint digest mismatch before pruning: {rel(path)}")
                path.unlink()
                removed.append({"file": path.name, "sha256": point["checkpoint_sha256"]})
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG,
                        help="frozen base config, relative to repo root (default: the 1,000-update local config)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seeds", type=int, default=10, help="number of FRESH replicates (indices 1..N)")
    parser.add_argument("--skip-original", action="store_true",
                        help="do not run replicate 0 (original seeds; reproduction check, not in primary analysis)")
    parser.add_argument("--replicates", type=str, default=None,
                        help="explicit comma-separated replicate indices (overrides --seeds/--skip-original)")
    parser.add_argument("--tag", default="main", help="run namespace under runs/ and results/")
    parser.add_argument("--cpu-smoke", action="store_true", help="CPU smoke harness only (see seeded_train.py)")
    parser.add_argument("--prune-intermediate-checkpoints", action="store_true",
                        help="after each complete replicate, delete step 50/200/500 checkpoints (digests kept)")
    parser.add_argument("--keep-going", action="store_true", help="continue to next replicate after a failure")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true", help="skip device/data/disk checks (not advised)")
    args = parser.parse_args()

    base_path = args.base_config if args.base_config.is_absolute() else ROOT / args.base_config
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if args.replicates:
        args.replicate_list = sorted({int(value) for value in args.replicates.split(",")})
    else:
        args.replicate_list = ([] if args.skip_original else [0]) + list(range(1, args.seeds + 1))
    if not args.replicate_list:
        parser.error("no replicates selected")
    if args.cpu_smoke and (args.device != "cpu" or "smoke" not in args.tag):
        parser.error("--cpu-smoke requires --device cpu and a tag containing 'smoke'")
    if not args.cpu_smoke and base.get("pilot_stage") == "replication_cpu_smoke":
        parser.error("smoke base config requires --cpu-smoke")

    runs_dir, results_dir = RUNS_DIR / args.tag, RESULTS_DIR / args.tag
    plan = [{"replicate": index, **seeds_for_replicate(base, index)} for index in args.replicate_list]
    print(json.dumps({"tag": args.tag, "base_config": rel(base_path), "device": args.device, "plan": plan}, indent=1))
    if not args.skip_preflight:
        problems, needed = preflight(args, base)
        print(f"estimated checkpoint disk: {needed / 1024 ** 3:.1f} GiB")
        if problems:
            for problem in problems:
                print("PREFLIGHT:", problem, file=sys.stderr)
            sys.exit(2)
    if args.dry_run:
        return

    launch = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "tag": args.tag, "base_config": rel(base_path), "base_config_sha256": sha256_file(base_path),
              "device": args.device, "python": platform.python_version(), "argv": sys.argv[1:],
              "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                         text=True).stdout.strip() or None,
              "replicates": []}
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "tools")] +
                                        ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    failures = 0
    for index in args.replicate_list:
        name = replicate_name(index)
        config_path = runs_dir / "configs" / f"{name}.json"
        report_path = runs_dir / f"{name}.json"
        checkpoints = runs_dir / f"{name}_checkpoints"
        summary_path = results_dir / f"train_summary_{name}.json"
        entry = {"replicate": index, **seeds_for_replicate(base, index)}
        wanted = replicate_config_bytes(base_path, index)
        check_replication_config(json.loads(wanted), base)
        if config_path.exists():
            if config_path.read_bytes() != wanted:
                raise FileExistsError(f"{rel(config_path)} exists with different content; use a new --tag")
        else:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_bytes(wanted)
        if report_path.exists():
            status = json.loads(report_path.read_text(encoding="utf-8")).get("status")
            if status != "complete":
                raise RuntimeError(f"{rel(report_path)} has status {status!r}; never overwritten. "
                                   "Inspect it, then rerun the remaining replicates with a NEW --tag.")
            print(f"[{name}] already complete; skipping training")
            entry["action"] = "skipped_existing_complete"
        else:
            if checkpoints.exists():
                raise FileExistsError(f"{rel(checkpoints)} exists without a report; inspect and use a new --tag")
            command = [sys.executable, str(HERE / "seeded_train.py"), "--config", str(config_path),
                       "--base-config", str(base_path), "--output", str(report_path), "--device", args.device]
            if args.cpu_smoke:
                command.append("--cpu-smoke")
            print(f"[{name}] seed={entry['seed']} data_seed={entry['data_seed']} -> {rel(report_path)}", flush=True)
            began = time.perf_counter()
            completed = subprocess.run(command, cwd=ROOT, env=env)
            entry.update(action="trained", returncode=completed.returncode,
                         invocation_seconds=round(time.perf_counter() - began, 3))
            print(f"[{name}] exit {completed.returncode} after {entry['invocation_seconds']:.1f}s", flush=True)
        if report_path.exists():
            report_status = json.loads(report_path.read_text(encoding="utf-8")).get("status")
            entry["status"] = report_status
            if report_status == "complete" and args.prune_intermediate_checkpoints:
                entry["pruned_checkpoints"] = prune(report_path, base)
            summary = summarize(report_path, index, config_path, base_path)
            if "pruned_checkpoints" in entry:
                summary["pruned_intermediate_checkpoints"] = entry["pruned_checkpoints"]
            if summary_path.exists():
                print(f"[{name}] summary exists; leaving {rel(summary_path)} untouched")
            else:
                write_json_new(summary_path, summary)
        else:
            entry["status"] = "no_report"
        launch["replicates"].append(entry)
        if entry["status"] != "complete":
            failures += 1
            if not args.keep_going:
                break
    launch["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    stamp = launch["started_utc"].replace(":", "").replace("-", "")
    write_json_new(results_dir / f"run_log_{stamp}.json", launch)
    complete = sum(entry.get("status") == "complete" for entry in launch["replicates"])
    print(json.dumps({"complete": complete, "failed": failures, "results_dir": rel(results_dir)}))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
