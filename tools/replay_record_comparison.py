"""Read-only checkpoint/action replay, defaulting to the preserved CPU runtime.

This loads only a local run artifact after checking its digest and bound source
hashes. It performs evaluation, never optimizer work or checkpoint adaptation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from controlled_lm.record_learning_comparison import RecordProposer, World, rollout, summarize
from train_record_comparison import ROOT, state_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=("internal", "external"), required=True)
    parser.add_argument("--split", choices=("train", "iid", "paraphrase", "composition"), default="iid")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("replay output already exists; artifacts are immutable")
    torch.set_num_threads(2)
    preregistered = json.loads((args.run / "preregistered.json").read_text())
    for path, expected in preregistered["sources"]["source_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError("bound source drift: " + path)
    report = json.loads((args.run / f"seed-{args.seed}-{args.arm}.json").read_text())
    checkpoint = args.run / report["checkpoint_file"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
        raise ValueError("checkpoint file digest mismatch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    model = RecordProposer(config["hidden_size"], config["native_size"])
    model.load_state_dict(saved["state_dict"])
    if state_hash(model) != report["final_state_sha256"]:
        raise ValueError("checkpoint tensor digest mismatch")
    model.eval()
    manifest_path = args.run / "world_manifest.json"
    run_report = json.loads((args.run / "report.json").read_text())
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != run_report["dataset_manifest_sha256"]:
        raise ValueError("world manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text())
    worlds = []
    for row in manifest[args.split]:
        world = World(row["world_id"], tuple(tuple(fact) for fact in row["facts"]),
                      tuple(tuple(rule) for rule in row["rules"]), tuple(row["release_keys"]),
                      row["query_key"], row["paraphrase"])
        if world.signature() != row["semantic_sha256"]:
            raise ValueError("world semantic digest mismatch")
        worlds.append(world)
    start = time.monotonic()
    actual = [rollout(model, world, args.arm) for world in worlds]
    expected = report["free_running"][args.split]
    differences = [i for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
    result = {"seed": args.seed, "arm": args.arm, "split": args.split, "device": "cpu", "dtype": "fp32",
              "checkpoint_sha256": report["checkpoint_sha256"], "tensor_sha256": state_hash(model),
              "worlds": len(worlds), "exact_action_feedback_replay": actual == expected,
              "different_world_indices": differences, "metrics": summarize(actual),
              "wall_seconds": time.monotonic() - start,
              "note": "CPU replay of CUDA weights may differ by floating-point association near decision boundaries; report differences rather than replacing evidence."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))
    if actual != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
