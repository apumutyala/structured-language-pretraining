"""Bounded CPU-only fixed-input retention diagnostic of frozen 400-step weights.

No training, optimizer, GPU, model selection, or learnability conclusion. Valid
token coefficient products describe sensitivity to an incoming recurrent state,
not sensitivity to all inputs/parameters or learned output heads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from controlled_lm.record_learning_comparison import RecordProposer, World, typed_workspace  # noqa: E402
from controlled_lm.semantic_interface_comparison import canonical  # noqa: E402

SEED = 2026102141
INDICES = (0, 1, 2)  # Selected before loading weights or measuring coefficients.
ARMS = ("internal", "external")
DEFAULT_RUN = ROOT / "runs/record-comparison-cuda-paired-v2-20261021"
DEFAULT_OUTPUT = ROOT / "results/record_retention_20261021.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, result):
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def finite_log(value):
    """JSON null denotes exact zero measured coefficient product (log=-infinity)."""
    value = float(value)
    return value if math.isfinite(value) else None


def product_summary(coefficients, rho):
    # FP64 log accumulation of the observed FP32 coefficients avoids product
    # underflow. This is not an IEEE/PyTorch Jacobian refinement proof.
    logs = coefficients.double().log().sum(dim=0)
    maximum, minimum = float(logs.max()), float(logs.min())
    count = coefficients.shape[0]
    return {
        "valid_tokens": count,
        "max_log_retention_natural": finite_log(maximum),
        "min_log_retention_natural": finite_log(minimum),
        "max_log10_retention": finite_log(maximum / math.log(10)),
        "min_log10_retention": finite_log(minimum / math.log(10)),
        "diagonal_norm_exp_max_log_float64": math.exp(maximum),
        "minimum_diagonal_exp_min_log_float64": math.exp(minimum),
        "float64_exp_can_underflow": True,
        "zero_coefficient_coordinates": int((coefficients == 0).any(dim=0).sum()),
        "rho_to_valid_tokens": rho ** count,
        "rho_log_bound_natural": count * math.log(rho),
    }


@torch.no_grad()
def measure(model, observation, arm):
    question = observation["question"]
    text = question if arm == "internal" else canonical(observation) + "\n" + question
    encoded = text.encode("ascii")
    ids = torch.tensor([byte + 1 for byte in encoded], dtype=torch.long, device="cpu")
    workspace = torch.tensor(typed_workspace(observation), dtype=torch.float32, device="cpu")
    semantic = model.semantic_projection(workspace) * (arm == "internal")
    inputs = model.embedding(ids) + semantic[None]
    projected = model.core.input_projection(inputs)
    gate_logits = projected[:, :model.core.hidden_size]
    coefficients = model.core.rho * gate_logits.sigmoid()
    if not bool(torch.isfinite(coefficients).all()):
        raise ValueError("nonfinite measured coefficients")
    effective_rho = float(torch.tensor(model.core.rho, dtype=torch.float32))
    if not bool(((coefficients >= 0) & (coefficients <= effective_rho)).all()):
        raise ValueError("coefficient exceeds FP32 rho")
    query_length = len(question.encode("ascii"))
    horizons = sorted({1, query_length, len(encoded)} | {n for n in (128, 512) if n <= len(encoded)})
    return {
        "arm": arm,
        "input_sha256": hashlib.sha256(encoded).hexdigest(),
        "valid_tokens": len(encoded), "padding_tokens": 0,
        "question_valid_tokens": query_length,
        "canonical_json_valid_tokens": 0 if arm == "internal" else len(canonical(observation).encode("ascii")),
        "rho_architectural": model.core.rho, "rho_rounded_fp32": effective_rho,
        "gate_logit_min": float(gate_logits.min()), "gate_logit_max": float(gate_logits.max()),
        "coefficient_min": float(coefficients.min()), "coefficient_max": float(coefficients.max()),
        "full_observation": product_summary(coefficients, model.core.rho),
        "suffix_retention": [product_summary(coefficients[-horizon:], model.core.rho) for horizon in horizons],
        "suffix_note": "Each suffix measures incoming-state sensitivity immediately before that suffix, holding its inputs fixed; all coordinates, not task-feature directions.",
    }


def run(run_path, output):
    if output.exists():
        raise ValueError("diagnostic output exists; preserve prior evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    protocol = {"status": "selection_frozen_before_measurement", "schema": "record-retention-diagnostic-v1",
                "selection": {"seed": SEED, "split": "iid", "manifest_indices": list(INDICES),
                              "observation": "initial only; no oracle actions or rollout selection",
                              "arms": list(ARMS), "checkpoint_training_updates": 400},
                "device": "cpu", "coefficient_dtype": "float32", "log_accumulation_dtype": "float64",
                "paid_compute_usd": 0, "training_or_optimizer": False,
                "run_directory_name": run_path.name,
                "tool_sha256": digest(__file__)}
    write_json(output, protocol)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        prereg = json.loads((run_path / "preregistered.json").read_text(encoding="utf-8"))
        report = json.loads((run_path / "report.json").read_text(encoding="utf-8"))
        config = prereg["config"]
        if config["updates"] != 400 or SEED not in config["seeds"] or config["precision"] != "fp32":
            raise ValueError("not the frozen 400-update checkpoint configuration")
        for name, expected in prereg["sources"]["source_sha256"].items():
            if digest(ROOT / name) != expected:
                raise ValueError("checkpoint-bound source drift")
        manifest_path = run_path / "world_manifest.json"
        if digest(manifest_path) != report["dataset_manifest_sha256"]:
            raise ValueError("manifest digest mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        observations = []
        for index in INDICES:
            row = manifest["iid"][index]
            world = World(row["world_id"], tuple(tuple(item) for item in row["facts"]),
                          tuple(tuple(item) for item in row["rules"]), tuple(row["release_keys"]),
                          row["query_key"], row["paraphrase"])
            if world.signature() != row["semantic_sha256"]:
                raise ValueError("world semantic digest mismatch")
            observations.append((index, world, world.episode().structured_observation()))
        selection_bytes = canonical([{"index": i, "world_id": world.world_id,
                                      "observation": observation} for i, world, observation in observations]).encode("ascii")
        protocol.update({"observation_fixture_sha256": hashlib.sha256(selection_bytes).hexdigest(),
                         "manifest_sha256": digest(manifest_path), "source_sha256": prereg["sources"]["source_sha256"]})
        write_json(output, protocol)  # Observations fixed before loading any checkpoint.
        measurements, checkpoints = [], []
        for arm in ARMS:
            arm_report = json.loads((run_path / f"seed-{SEED}-{arm}.json").read_text(encoding="utf-8"))
            checkpoint = run_path / arm_report["checkpoint_file"]
            checkpoint_hash = digest(checkpoint)
            if checkpoint_hash != arm_report["checkpoint_sha256"]:
                raise ValueError("checkpoint digest mismatch")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["arm"] != arm or saved["fresh_seed"] != SEED or saved["config"] != config:
                raise ValueError("checkpoint identity/config mismatch")
            model = RecordProposer(config["hidden_size"], config["native_size"]).cpu().eval()
            model.load_state_dict(saved["state_dict"], strict=True)
            if any(parameter.device.type != "cpu" for parameter in model.parameters()):
                raise ValueError("non-CPU diagnostic model")
            checkpoints.append({"arm": arm, "checkpoint_file": checkpoint.name,
                                "checkpoint_sha256": checkpoint_hash, "weights_only": True})
            for index, world, observation in observations:
                measurements.append({"manifest_index": index, "world_id": world.world_id,
                                     "world_semantic_sha256": world.signature(), **measure(model, observation, arm)})
            del model, saved
        protocol.update({"status": "completed", "torch": str(torch.__version__),
                         "checkpoints": checkpoints, "measurements": measurements,
                         "whole_model_or_output_head_jacobian_measured": False,
                         "limitations": ["Real fixed-input diagonal recurrence bound only; not a floating-point refinement theorem.",
                                         "Coefficient products apply to incoming state, not all input/parameter sensitivities.",
                                         "Internal typed workspace is injected at every query token; external JSON also injects fresh inputs across its bytes.",
                                         "No bound on learned readout/embedding/projection weight norms or whole-model gradients follows.",
                                         "Three first IID observations from one seed, not performance evaluation, a causal attribution, or a learning impossibility result.",
                                         "Additional training may alter gates, candidates, embeddings and readout; rho=.99 remains fixed."],
                         "wall_seconds": time.perf_counter() - started})
        write_json(output, protocol)
        return protocol
    except Exception as error:
        protocol.update({"status": "failed", "error_type": type(error).__name__,
                         "wall_seconds": time.perf_counter() - started})
        write_json(output, protocol)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.run, args.output)
    print(json.dumps({"status": result["status"], "device": result["device"],
                      "measurements": len(result["measurements"]), "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
