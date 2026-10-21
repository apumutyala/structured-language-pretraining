"""Fixed-final equal-encoding RECORD action-learning placement fit diagnostic.

New experiment files only; frozen raw-JSON comparison is reused read-only.
CPU/default is not a GPU lease. Coordinator owns any local CUDA execution.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from controlled_lm.record_learning_comparison import (
    FEATURE_SIZE, expected_answer, make_dataset, oracle_trajectory, rollout, summarize,
)
from controlled_lm.record_placement_diagnostic import RecordPlacementProposer
from train_record_comparison import (
    ROOT, bootstrap_paired, provenance, state_hash, train_batch, validate_config, write_json,
)

ARMS = ("internal", "typed_readout")
ADDED_SOURCES = ("src/controlled_lm/record_placement_diagnostic.py", "tools/train_record_placement_diagnostic.py")


def validate_placement(config):
    if config.get("schema") != "record-placement-fit-v1":
        raise ValueError("placement schema required")
    stage_shapes = {"cpu_enablement_only": ("cpu", 1, 8),
                    "cuda_resource_qualification_only": ("cuda", 8, 8),
                    "fixed_final_trainfit_diagnostic": ("cuda", 1500, 128)}
    expected = stage_shapes.get(config.get("stage"))
    if expected != (config.get("device"), config.get("updates"), config.get("eval_worlds_per_split")):
        raise ValueError("placement stage/device/update/evaluation mismatch")
    fixed = {"seed": 2026102159, "data_seed": 2026102161, "hidden_size": 64,
             "native_size": 128, "train_worlds": 32, "batch_worlds": 8,
             "learning_rate": .001, "both_arm_trainfit_gate": .8}
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("preregistered fit architecture/data/recipe drift")
    validate_config({**config, "schema": "record-comparison-v2"})


def execute(config, output):
    validate_placement(config)
    output.mkdir(parents=True, exist_ok=False)
    start, arm_start = time.monotonic(), None
    current = {"seed": config["seed"], "arm": None, "update": 0, "training": []}
    report = {"status": "in_progress", "scope": config["scope"], "arms": {},
              "feature_size": FEATURE_SIZE,
              "intervention": "Same typed workspace/query bytes/active tensors; workspace projection added before SSM vs immediately before final RMS/SwiGLU.",
              "limitations": ["One fresh seed; 32-world train-fit diagnostic, not a broad comparison.",
                              "No byte-language objective or generic tool-call syntax learning.",
                              "Both active capacity and logical information match; gradient exposure and semantic influence on recurrent state intentionally differ.",
                              "Internal projection repeats at every query token; readout projection is added once. Token work matches but arithmetic work differs slightly.",
                              "No claims about full8/L0-L4, absolute hallucination freedom, general external model strength, or architecture superiority."]}
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ADDED_SOURCES}
    write_json(output / "preregistered.json", {"config": config, "added_source_sha256": source_hashes,
                                             "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()})
    write_json(output / "report.json", report)

    def check_time():
        if time.monotonic() - start > config["max_wall_seconds"]:
            raise TimeoutError("fixed-final placement diagnostic wall cap reached")

    try:
        torch.set_num_threads(config["threads"])
        torch.use_deterministic_algorithms(True)
        if config["device"] == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; no CPU environment modifications permitted")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        environment = provenance(config["device"])
        environment["source_sha256"].update(source_hashes)
        write_json(output / "environment.json", environment)
        report["environment"] = environment
        dataset = make_dataset(config["data_seed"], config["train_worlds"], config["eval_worlds_per_split"])
        manifest = {split: [{**asdict(world), "semantic_sha256": world.signature()} for world in worlds]
                    for split, worlds in dataset.items()}
        write_json(output / "world_manifest.json", manifest)
        report["dataset_manifest_sha256"] = hashlib.sha256((output / "world_manifest.json").read_bytes()).hexdigest()
        report["controls"] = {}
        for split, worlds in dataset.items():
            supported = sum(expected_answer(world) is not None for world in worlds)
            report["controls"][split] = {"worlds": len(worlds), "supported_worlds": supported,
                                          "always_abstain_supported_recall": 0.,
                                          "always_abstain_task_exact": 1 - supported / len(worlds),
                                          "exact_planner_supported_recall": 1., "exact_planner_task_exact": 1.}
        trajectories = [oracle_trajectory(world) for world in dataset["train"]]
        initial_hash = None
        for arm in ARMS:
            arm_start = time.monotonic()
            current = {"seed": config["seed"], "arm": arm, "update": 0, "training": []}
            check_time()
            torch.manual_seed(config["seed"])
            model = RecordPlacementProposer(config["hidden_size"], config["native_size"]).to(config["device"])
            digest = state_hash(model)
            if initial_hash is not None and digest != initial_hash:
                raise RuntimeError("placement pair initial weights differ")
            initial_hash = digest
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            rng, log, tokens = random.Random(config["seed"] + 1), [], 0
            current["training"] = log
            if config["device"] == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            train_start = time.monotonic()
            model.train()
            for update in range(config["updates"]):
                check_time()
                indices = rng.sample(range(len(trajectories)), config["batch_worlds"])
                loss, gradient_norm, count = train_batch(model, arm, [trajectories[i] for i in indices], optimizer)
                tokens += count
                log.append({"update": update + 1, "loss": loss, "gradient_l2": gradient_norm, "input_tokens": count})
                current["update"] = update + 1
                write_json(output / "current_progress.json", {"seed": config["seed"], "arm": arm,
                           "update": update + 1, "last_loss": loss, "input_tokens": tokens,
                           "wall_seconds": time.monotonic() - arm_start})
            if config["device"] == "cuda":
                torch.cuda.synchronize()
            training_seconds = time.monotonic() - train_start
            training_peak = torch.cuda.max_memory_allocated() if config["device"] == "cuda" else None
            evaluation_start = time.monotonic()
            model.eval()
            evaluation, summaries = {}, {}
            for split in ("train", "iid", "paraphrase", "composition"):
                evaluation[split] = []
                for world in dataset[split]:
                    check_time()
                    evaluation[split].append(rollout(model, world, arm))
                summaries[split] = summarize(evaluation[split])
            if config["device"] == "cuda":
                torch.cuda.synchronize()
            arm_result = {"initial_state_sha256": digest, "final_state_sha256": state_hash(model),
                          "allocated_parameters": sum(p.numel() for p in model.parameters()),
                          "active_task_parameters": sum(p.numel() for p in model.parameters()),
                          "training_input_tokens": tokens, "training_wall_seconds": training_seconds,
                          "evaluation_wall_seconds": time.monotonic() - evaluation_start,
                          "training_peak_cuda_allocated_bytes": training_peak,
                          "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if config["device"] == "cuda" else None,
                          "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if config["device"] == "cuda" else None,
                          "cpu_tensor_peak_bytes": None, "training": log, "metrics": summaries,
                          "free_running": evaluation}
            checkpoint = output / f"seed-{config['seed']}-{arm}.pt"
            torch.save({"fresh_seed": config["seed"], "arm": arm, "config": config,
                        "state_dict": model.state_dict()}, checkpoint)
            arm_result.update({"checkpoint_file": checkpoint.name,
                               "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
            report["arms"][arm] = arm_result
            write_json(output / f"seed-{config['seed']}-{arm}.json", arm_result)
            write_json(output / "report.json", report)
            del optimizer, model
            if config["device"] == "cuda":
                torch.cuda.empty_cache()
        if report["arms"]["internal"]["training_input_tokens"] != report["arms"]["typed_readout"]["training_input_tokens"]:
            raise RuntimeError("equal-encoding teacher token-work mismatch")
        report["paired_differences"] = {split: bootstrap_paired(
            report["arms"]["internal"]["free_running"][split],
            report["arms"]["typed_readout"]["free_running"][split], config["seed"] + 2)
            for split in ("iid", "paraphrase", "composition")}
        for comparison in report["paired_differences"].values():
            for metric in ("primary_supported_answer_recall", "secondary_task_exact"):
                endpoint = comparison[metric]
                endpoint["internal_minus_typed_readout"] = endpoint.pop("internal_minus_external")
        report["both_arm_trainfit_gate"] = {"threshold": config["both_arm_trainfit_gate"],
            "pass": all(report["arms"][arm]["metrics"]["train"]["supported_answer_recall"] >= config["both_arm_trainfit_gate"] for arm in ARMS),
            "note": "Fixed final checkpoint only; fit gate is diagnostic, not an architecture or held-out-success gate."}
        report["status"] = "finite_execution_complete"
    except Exception as error:
        current["wall_seconds"] = None if arm_start is None else time.monotonic() - arm_start
        current["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None
        current["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None
        report.update({"status": "incomplete", "error_type": type(error).__name__, "error": str(error),
                       "failed_arm_progress": current, "traceback": traceback.format_exc()})
        write_json(output / "failure.json", {"current": current, "error_type": type(error).__name__,
                                             "error": str(error), "traceback": traceback.format_exc()})
    report["total_wall_seconds"] = time.monotonic() - start
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.device is not None and args.device != config["device"]:
        raise ValueError("CLI/config device mismatch")
    report = execute(config, args.output)
    print(json.dumps({"status": report["status"], "wall_seconds": report["total_wall_seconds"],
                      "trainfit_gate": report.get("both_arm_trainfit_gate"), "output": str(args.output)}))
    if report["status"] != "finite_execution_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
