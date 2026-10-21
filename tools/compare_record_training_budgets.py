"""Read-only comparison of two completed, fresh RECORD training-budget runs.

Uses only the standard library. No training, GPU access, checkpoint adaptation,
or selection occurs. Reports contain run directory names, never absolute paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


METADATA_FIELDS = {"stage", "budget_test"}
SPLITS = ("train", "iid", "paraphrase", "composition")
ARMS = ("internal", "external")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def raw_metrics(rows):
    """Recompute reported endpoints from decisions, without model imports."""
    positive = [row for row in rows if row["expected"] is not None]
    negative = [row for row in rows if row["expected"] is None]
    correct = sum(row["released"] == row["expected"] for row in positive)
    abstained = sum(row["explicit_abstain"] for row in negative)
    attempts = [step for row in rows for step in row["history"]]
    return {"worlds": len(rows), "task_exact": (correct + abstained) / len(rows),
            "supported_worlds": len(positive), "supported_answers_correct": correct,
            "unsupported_worlds": len(negative),
            "supported_answer_recall": correct / len(positive) if positive else None,
            "abstention_exact": abstained / len(negative) if negative else None,
            "invalid_operation_rate": sum(not step["feedback"]["accepted"] for step in attempts) / len(attempts),
            "query_wrong_release_count": sum(row["released"] is not None and row["released"] != row["expected"] for row in rows),
            "operations": len(attempts),
            "input_tokens": sum(step["input_tokens"] for step in attempts)}


def compare_runs(before_path, after_path):
    paths = (Path(before_path), Path(after_path))
    preregs = [read_json(path / "preregistered.json") for path in paths]
    reports = [read_json(path / "report.json") for path in paths]
    configs = [item["config"] for item in preregs]
    old_updates, new_updates = [config["updates"] for config in configs]
    if not 0 < old_updates < new_updates:
        raise ValueError("comparison requires a strictly larger second training budget")
    if any(report.get("status") != "enablement_passed" for report in reports):
        raise ValueError("both completed run receipts are required")
    seeds = configs[0]["seeds"]
    if configs[1]["seeds"] != seeds:
        raise ValueError("seed lists differ; this is not a paired budget extension")

    checks = {}
    def check(name, condition):
        checks[name] = bool(condition)
    keys = set(configs[0]) | set(configs[1])
    changed_fields = sorted(key for key in keys if configs[0].get(key) != configs[1].get(key))
    check("scientific_config_matches", all(key in {"updates"} | METADATA_FIELDS for key in changed_fields))
    check("source_ledgers_match", preregs[0]["sources"]["source_sha256"] == preregs[1]["sources"]["source_sha256"])
    runtime_fields = ("python", "torch", "torch_cuda", "device", "device_name", "dtype", "compute_capability")
    check("execution_runtime_matches", all(preregs[0]["sources"].get(field) == preregs[1]["sources"].get(field) for field in runtime_fields))
    drivers = [item["sources"].get("nvidia_smi", "").split(",") for item in preregs]
    check("driver_receipts_match", all(len(parts) >= 2 for parts in drivers)
          and drivers[0][1].strip() == drivers[1][1].strip())
    manifests = [read_json(path / "world_manifest.json") for path in paths]
    check("same_manifest_bytes", digest(paths[0] / "world_manifest.json") == digest(paths[1] / "world_manifest.json"))
    for label, path, prereg, report, config in zip(("before", "after"), paths, preregs, reports, configs):
        check(label + "_config_digest", hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() == prereg["config_sha256"])
        check(label + "_manifest_digest", digest(path / "world_manifest.json") == report["dataset_manifest_sha256"])
        check(label + "_seed_order", [pair["seed"] for pair in report["seeds"]] == seeds)

    comparisons, arm_costs = [], []
    for seed in seeds:
        pairs = [next(pair for pair in report["seeds"] if pair["seed"] == seed) for report in reports]
        for arm in ARMS:
            results = [pair["arms"][arm] for pair in pairs]
            prefix = f"seed_{seed}_{arm}"
            check(prefix + "_initial_matches", results[0]["initial_state_sha256"] == results[1]["initial_state_sha256"])
            check(prefix + "_within_seed_initial_pair", all(pair["arms"]["internal"]["initial_state_sha256"] == pair["arms"]["external"]["initial_state_sha256"] for pair in pairs))
            logs = [result["training"] for result in results]
            check(prefix + "_exact_training_prefix", logs[0] == logs[1][:old_updates])
            for label, path, result, log, updates in zip(("before", "after"), paths, results, logs, (old_updates, new_updates)):
                check(prefix + "_" + label + "_arm_receipt", read_json(path / f"seed-{seed}-{arm}.json") == result)
                check(prefix + "_" + label + "_checkpoint_digest", digest(path / result["checkpoint_file"]) == result["checkpoint_sha256"])
                check(prefix + "_" + label + "_complete_log", [row["update"] for row in log] == list(range(1, updates + 1)))
                check(prefix + "_" + label + "_finite_training", all(math.isfinite(row["loss"]) and math.isfinite(row["gradient_l2"]) for row in log))
                check(prefix + "_" + label + "_token_total", sum(row["input_tokens"] for row in log) == result["training_input_tokens"])
            for split in SPLITS:
                rows = [result["free_running"][split] for result in results]
                metrics = [raw_metrics(value) for value in rows]
                check(prefix + "_" + split + "_metrics", all(metric == result["metrics"][split] for metric, result in zip(metrics, results)))
                check(prefix + "_" + split + "_world_order", all([row["world_id"] for row in value] == [world["world_id"] for world in manifests[0][split]] for value in rows))
                check(prefix + "_" + split + "_paired_labels", [row["expected"] for row in rows[0]] == [row["expected"] for row in rows[1]])
                old, new = [metric["supported_answer_recall"] for metric in metrics]
                comparisons.append({"seed": seed, "arm": arm, "split": split,
                                    "before": metrics[0], "after": metrics[1],
                                    "supported_recall_change": None if old is None or new is None else new - old})
            arm_costs.append({"seed": seed, "arm": arm,
                              "before_training_seconds": results[0]["training_wall_seconds"],
                              "after_training_seconds_charged_in_full": results[1]["training_wall_seconds"],
                              "before_evaluation_seconds": results[0]["evaluation_wall_seconds"],
                              "after_evaluation_seconds": results[1]["evaluation_wall_seconds"],
                              "before_training_input_tokens": results[0]["training_input_tokens"],
                              "after_training_input_tokens_charged_in_full": results[1]["training_input_tokens"],
                              "replayed_prefix_input_tokens": sum(row["input_tokens"] for row in logs[1][:old_updates]),
                              "new_suffix_input_tokens": sum(row["input_tokens"] for row in logs[1][old_updates:]),
                              "after_peak_cuda_allocated_bytes": results[1]["peak_cuda_allocated_bytes"],
                              "after_peak_cuda_reserved_bytes": results[1]["peak_cuda_reserved_bytes"]})

    groups = []
    for arm in ARMS:
        for split in SPLITS:
            rows = [row for row in comparisons if row["arm"] == arm and row["split"] == split]
            before_correct = sum(row["before"]["supported_answers_correct"] for row in rows)
            after_correct = sum(row["after"]["supported_answers_correct"] for row in rows)
            denominator = sum(row["before"]["supported_worlds"] for row in rows)
            changes = [row["supported_recall_change"] for row in rows if row["supported_recall_change"] is not None]
            groups.append({"arm": arm, "split": split, "seeds": len(rows),
                           "unique_supported_worlds": rows[0]["before"]["supported_worlds"],
                           "supported_seed_world_trials": denominator,
                           "before_correct_seed_world_trials": before_correct, "after_correct_seed_world_trials": after_correct,
                           "before_macro_seed_mean_recall": sum(row["before"]["supported_answer_recall"] for row in rows) / len(rows) if denominator else None,
                           "after_macro_seed_mean_recall": sum(row["after"]["supported_answer_recall"] for row in rows) / len(rows) if denominator else None,
                           "seed_mean_recall_change": sum(changes) / len(changes) if changes else None,
                           "per_seed_recall_change_range": [min(changes), max(changes)] if changes else None})
    old_wall, new_wall = [report["total_wall_seconds"] for report in reports]
    return {"schema": "record-training-budget-comparison-v1", "all_checks_pass": all(checks.values()),
            "run_names": {"before": paths[0].name, "after": paths[1].name},
            "budgets": {"before_updates_per_arm": old_updates, "after_updates_per_arm": new_updates,
                        "changed_config_fields": changed_fields, "arms_per_budget": len(arm_costs)},
            "checks": checks, "per_seed": comparisons, "descriptive_seed_groups": groups,
            "cost": {"before_total_wall_seconds": old_wall,
                     "additional_after_run_wall_seconds_charged_in_full": new_wall,
                     "both_runs_total_wall_seconds": old_wall + new_wall,
                     "after_run_replayed_prefix_updates": len(arm_costs) * old_updates,
                     "after_run_new_suffix_updates": len(arm_costs) * (new_updates - old_updates),
                     "all_charged_training_updates": len(arm_costs) * (old_updates + new_updates),
                     "per_arm": arm_costs},
            "external_fit_gate": {"before": reports[0].get("external_fit_gate"), "after": reports[1].get("external_fit_gate")},
            "limitations": ["The larger budget restarts from random initialization; historical checkpoints omit optimizer state. Every repeated prefix and evaluation is charged.",
                            "The same development worlds are reused after seeing the smaller-budget results; this is a training-budget diagnostic, not fresh confirmation.",
                            "Seed-world trials repeat the same worlds across seeds and are not independent new evaluation examples. No pooled confidence interval or family-level rejection is implied.",
                            "A failed recipe or weak comparator does not reject a model family. Capability, safety and architecture superiority require separate evidence.",
                            "Exact prefix matching is evidence of matched replay through the smaller budget; no intermediate or best checkpoint is selected."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists; preserve completed comparison artifacts")
    result = compare_runs(args.before, args.after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print("Matched budget replay checks:", result["all_checks_pass"])
    print("seed arm split before_positive/denominator after_positive/denominator recall_change")
    for row in result["per_seed"]:
        print(row["seed"], row["arm"], row["split"],
              f"{row['before']['supported_answers_correct']}/{row['before']['supported_worlds']}",
              f"{row['after']['supported_answers_correct']}/{row['after']['supported_worlds']}",
              row["supported_recall_change"])
    print("Additional full rerun wall seconds charged:", result["cost"]["additional_after_run_wall_seconds_charged_in_full"])
    print("Reused development evaluation; no fresh confirmation or model-family rejection.")
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
