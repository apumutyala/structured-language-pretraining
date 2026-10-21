"""Run a preregistered small RECORD action learner, never a joint language model.

CPU is the default. CUDA execution requires the coordinator's separate local
device lease. No checkpoints are inherited; seeds share initial tensors only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from controlled_lm.record_learning_comparison import (
    FEATURE_SIZE, RecordProposer, action_loss, make_dataset,
    expected_answer, oracle_trajectory, rollout, summarize,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("src/controlled_lm/record_learning_comparison.py", "tools/train_record_comparison.py",
           "src/controlled_lm/supported_state.py", "src/controlled_lm/semantic_interface_comparison.py",
           "src/controlled_lm/normalized_typed_language_model.py", "src/controlled_lm/parallel_ssm.py",
           "src/controlled_lm/invariant_ssm.py")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def validate_config(config):
    if config.get("schema") != "record-comparison-v2" or config.get("device") not in ("cpu", "cuda"):
        raise ValueError("explicit RECORD comparison schema/device required")
    if config.get("precision") != "fp32" or config.get("paid_compute_usd") != 0:
        raise ValueError("only local fp32 execution is enabled")
    for key, ceiling in (("updates", 2000), ("batch_worlds", 64), ("train_worlds", 4096),
                         ("eval_worlds_per_split", 2048), ("hidden_size", 256), ("native_size", 512),
                         ("threads", 16)):
        if type(config.get(key)) is not int or not 1 <= config[key] <= ceiling:
            raise ValueError(f"invalid bounded config {key}")
    if not 0 < config["learning_rate"] <= .01 or not 0 < config["max_wall_seconds"] <= 7200:
        raise ValueError("learning rate/wall cap out of range")
    seeds = config.get("seeds", [config.get("seed")])
    if not seeds or len(seeds) > 5 or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("one to five distinct explicit fresh seeds required")
    return seeds


def provenance(device):
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda, "device": device, "dtype": "torch.float32",
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}}
    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        info.update({"device_name": properties.name, "total_vram_bytes": properties.total_memory,
                     "compute_capability": [properties.major, properties.minor],
                     "free_vram_before_bytes": torch.cuda.mem_get_info()[0]})
        try:
            info["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"],
                text=True, timeout=10).strip()
        except (FileNotFoundError, subprocess.SubprocessError) as error:
            info["nvidia_smi_error"] = str(error)
    return info


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        if isinstance(tensor, torch.Tensor):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(json.dumps(tensor, sort_keys=True).encode())
    return digest.hexdigest()


def train_batch(model, arm, trajectories, optimizer):
    optimizer.zero_grad(set_to_none=True)
    hx, losses, tokens = None, [], 0
    for step in range(max(map(len, trajectories))):
        active = [i for i, trajectory in enumerate(trajectories) if step < len(trajectory)]
        transitions = [trajectory[min(step, len(trajectory) - 1)] for trajectory in trajectories]
        outputs, hx, lengths = model([item[0] for item in transitions], arm, hx)
        selected = tuple(head[active] for head in outputs)
        losses.append(action_loss(selected, [transitions[i][1] for i in active]))
        tokens += sum(lengths)  # Includes repeated inactive rows: real compute, not useful tokens.
    loss = torch.stack(losses).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite training loss")
    loss.backward()
    squared_norm = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("nonfinite gradient")
            squared_norm += parameter.grad.detach().square().sum()
    gradient_norm = float(squared_norm.sqrt())
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("nonfinite post-update parameter")
    return float(loss.detach()), gradient_norm, tokens


def bootstrap_paired(internal, external, seed):
    """Descriptive world-bootstrap 95% interval, not a superiority decision."""
    def correct(row):
        return int(row["released"] == row["expected"] if row["expected"] is not None else row["explicit_abstain"])
    if [r["world_id"] for r in internal] != [r["world_id"] for r in external]:
        raise ValueError("unpaired result rows")
    if [r["expected"] for r in internal] != [r["expected"] for r in external]:
        raise ValueError("paired expected labels differ")

    def interval(rows):
        differences = [correct(a) - correct(b) for a, b in rows]
        rng, n = random.Random(seed), len(differences)
        if not n:
            return {"worlds": 0, "internal_minus_external": None, "world_bootstrap_95_interval": None}
        estimates = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
        return {"worlds": n, "internal_minus_external": sum(differences) / n,
                "world_bootstrap_95_interval": [estimates[49], estimates[1949]]}

    pairs = list(zip(internal, external))
    return {"primary_supported_answer_recall": interval([(a, b) for a, b in pairs if a["expected"] is not None]),
            "secondary_task_exact": interval(pairs),
            "note": "Within one seed and task stratum only; multiplicity/seed uncertainty unadjusted. No architecture claim."}


def run(config, output):
    seeds = validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    # Saved before data generation, model initialization, optimizer, or measurements.
    write_json(output / "preregistered.json", {"config": config, "sources": provenance(config["device"]),
                                             "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()})
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    if config["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; preserve working CPU environment")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()

    def check_time():
        if time.monotonic() - start > config["max_wall_seconds"]:
            raise TimeoutError("preregistered total wall cap reached")

    dataset = make_dataset(config["data_seed"], config["train_worlds"], config["eval_worlds_per_split"])
    manifest = {split: [{**asdict(world), "semantic_sha256": world.signature()} for world in worlds]
                for split, worlds in dataset.items()}
    write_json(output / "world_manifest.json", manifest)
    controls = {}
    for split, worlds in dataset.items():
        expected = [expected_answer(world) for world in worlds]
        positive = sum(value is not None for value in expected)
        controls[split] = {"worlds": len(worlds), "supported_worlds": positive,
                           "always_abstain_task_exact": 1 - positive / len(worlds),
                           "always_abstain_supported_recall": 0. if positive else None,
                           "exact_unary_planner_task_exact": 1.,
                           "exact_unary_planner_supported_recall": 1. if positive else None,
                           "planner_note": "Deterministic oracle_trajectory executes declared-rule BFS proposals through the same kernel and budget. This is a symbolic ceiling, not a learned comparator."}
    train_trajectories = [oracle_trajectory(world) for world in dataset["train"]]
    report = {"scope": "neural-to-RECORD action-learning enablement; no joint language objective",
              "dataset_manifest_sha256": hashlib.sha256((output / "world_manifest.json").read_bytes()).hexdigest(),
              "feature_size": FEATURE_SIZE, "seeds": [], "status": "in_progress",
              "controls": controls,
              "differences": ["Internal receives query bytes + full finite typed state projected before SSM.",
                              "External receives full canonical JSON observation plus repeated identical question every operation and retains SSM state; all duplicate question bytes charged.",
                              "Same allocated tensors/initial values; external semantic projection is inactive (zero task gradient).",
                              "Input sequence length, active parameter count, token work, and parsing burden differ.",
                              "Both use identical fixed action grammar, raw unmasked heads, labels and four-operation budget.",
                              "Generated record IDs and current release epochs are filled by the common grammar adapter; pointers/rule/op are learned.",
                              "Oracle trajectories contain accepted operations only; rejected-operation feedback is exposed at inference but not trained.",
                              "No generic text/tool-syntax generation, from-scratch language pretraining, architecture win, or full8/L0-L4 claim."]}
    write_json(output / "report.json", report)
    current = {"seed": None, "arm": None, "update": 0, "training": []}
    arm_start = None
    try:
        for seed in seeds:
            paired = {"seed": seed, "arms": {}}
            report["seeds"].append(paired)
            initial_hash = None
            for arm in ("internal", "external"):
                current = {"seed": seed, "arm": arm, "update": 0, "training": []}
                arm_start = time.monotonic()
                check_time()
                torch.manual_seed(seed)
                model = RecordProposer(config["hidden_size"], config["native_size"]).to(config["device"])
                digest = state_hash(model)
                if initial_hash is not None and digest != initial_hash:
                    raise RuntimeError("unmatched initial tensors")
                initial_hash = digest
                optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
                rng, log, train_tokens = random.Random(seed + 1), [], 0
                current["training"] = log
                if config["device"] == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                train_start = time.monotonic()
                for update in range(config["updates"]):
                    check_time()
                    indices = rng.sample(range(len(train_trajectories)), min(config["batch_worlds"], len(train_trajectories)))
                    loss, grad_norm, tokens = train_batch(model, arm, [train_trajectories[i] for i in indices], optimizer)
                    train_tokens += tokens
                    log.append({"update": update + 1, "loss": loss, "gradient_l2": grad_norm, "input_tokens": tokens})
                    current["update"] = update + 1
                    current["wall_seconds"] = time.monotonic() - arm_start
                    write_json(output / "current_progress.json", current)
                if config["device"] == "cuda":
                    torch.cuda.synchronize()
                training_seconds = time.monotonic() - train_start
                model.eval()
                evaluation, summaries, eval_start = {}, {}, time.monotonic()
                # Fit diagnostic uses training worlds; never mixed into held-out metric.
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
                              "active_task_parameters": sum(p.numel() for name, p in model.named_parameters()
                                                            if arm == "internal" or name != "semantic_projection.weight"),
                              "training_input_tokens": train_tokens, "training_wall_seconds": training_seconds,
                              "evaluation_wall_seconds": time.monotonic() - eval_start,
                              "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated() if config["device"] == "cuda" else None),
                              "peak_cuda_reserved_bytes": (torch.cuda.max_memory_reserved() if config["device"] == "cuda" else None),
                              "cpu_tensor_peak_bytes": None,
                              "training": log, "metrics": summaries, "free_running": evaluation}
                checkpoint = output / f"seed-{seed}-{arm}.pt"
                torch.save({"fresh_seed": seed, "arm": arm, "config": config, "state_dict": model.state_dict()},
                           checkpoint)
                arm_result["checkpoint_file"] = checkpoint.name
                arm_result["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                paired["arms"][arm] = arm_result
                write_json(output / "report.json", report)
                write_json(output / f"seed-{seed}-{arm}.json", arm_result)
                del optimizer, model
                if config["device"] == "cuda":
                    torch.cuda.empty_cache()
            paired["paired_differences"] = {split: bootstrap_paired(
                paired["arms"]["internal"]["free_running"][split],
                paired["arms"]["external"]["free_running"][split], seed + 2)
                for split in ("iid", "paraphrase", "composition")}
            write_json(output / "report.json", report)
        report["status"] = "enablement_passed"
        report["result_gate"] = "Finite fresh training and full free-running evaluations completed for both arms; capability and strong-baseline adequacy require inspecting fit and held-out metrics."
        if "pragmatic_external_fit_gate" in config:
            gate = config["pragmatic_external_fit_gate"]
            report["external_fit_gate"] = {"thresholds": gate, "all_seeds_pass": all(
                paired["arms"]["external"]["metrics"]["train"]["supported_answer_recall"] >= gate["train_supported_recall"]
                and paired["arms"]["external"]["metrics"]["iid"]["supported_answer_recall"] >= gate["iid_supported_recall"]
                for paired in report["seeds"]),
                "meaning": "Pragmatic within-task fit/usefulness check only; does not establish a strong general external baseline."}
    except Exception as error:
        current["wall_seconds"] = None if arm_start is None else time.monotonic() - arm_start
        current["peak_cuda_allocated_bytes"] = (torch.cuda.max_memory_allocated()
                                                  if config["device"] == "cuda" and torch.cuda.is_initialized() else None)
        current["peak_cuda_reserved_bytes"] = (torch.cuda.max_memory_reserved()
                                                 if config["device"] == "cuda" and torch.cuda.is_initialized() else None)
        report.update({"status": "incomplete", "error_type": type(error).__name__, "error": str(error),
                       "failed_arm_progress": current, "traceback": traceback.format_exc()})
        write_json(output / "failure.json", {"progress": current, "error_type": type(error).__name__,
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
        raise ValueError("CLI device must match preregistered config")
    report = run(config, args.output)
    print(json.dumps({"status": report["status"], "wall_seconds": report["total_wall_seconds"],
                      "output": str(args.output)}))
    if report["status"] != "enablement_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
