"""Bound the existing pilot trainer to the fresh RMS model; no inherited weights.

Only the private trainer module's model factory, source ledger and diagnostic
probe are rebound. Original trainer/objective files are untouched. Importing
this wrapper never launches training or changes an already-imported trainer.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from controlled_lm.normalized_typed_language_model import (
    NORMALIZATION_RECIPE, RMS_EPSILON, NormalizedTypedLanguageModel, parameter_free_rms,
)

ROOT = Path(__file__).resolve().parents[1]
BOUND_SOURCES = {
    "tools/train_language_pilot.py": "46258bd47ffef565e5ad3d23de521d3949aae240f4eb3ab0518233ac0e225216",
    "src/controlled_lm/typed_language_model.py": "49301bcc3c4504570ad8be300bf30930e6551962d490f8b1bde185a41626c2ac",
    "tools/qualify_language_gpu.py": "6b51f0ac79f6ea6d71dd37be19f71b97b7a56fc3ced14fa240e33d865558ce0d",
}


def validate_config(config, device):
    """Reject recipe drift and inherited-lineage paths before any model call."""
    stage = config.get("pilot_stage")
    expected = {
        "cpu_enablement": ("cpu", 1, [0, 1]),
        "cuda_qualification": ("cuda", 8, [0, 8]),
        "matched_stability": ("cuda", 1000, [0, 50, 200, 500, 1000]),
    }
    if stage not in expected or (device, config.get("updates"), config.get("checkpoints")) != expected[stage]:
        raise ValueError("stage/device/update/checkpoint mismatch")
    if any(key in config for key in ("resume_reference_report", "resume_reference_step",
                                     "prefix_reference_report", "prefix_reference_step")):
        raise ValueError("normalized pilots must start fresh; no resume/prefix reference")
    fixed = {"hidden_size": 512, "native_size": 1024, "num_layers": 4,
             "recurrent_kind": "ssm", "ssm_backend": "parallel", "projection_layout": "shared",
             "recurrent_residual": True, "semantic_modes": ["none", "internal", "output_only"],
             "precision": "fp32", "ssm_gate_bias_range": [0, 6], "story_rows": 31, "length": 128,
             "learning_rate": .0003, "world_answer_weight": .5,
             "seed": 2026102131, "data_seed": 2026102133, "dev_seed": 2026102137,
             "sample_seed": 2026102139, "manifest": "results/language_pilot_extraction.json",
             "prompts": ["Once upon a time,", "The little dog"],
             "normalization_recipe": NORMALIZATION_RECIPE, "normalization_epsilon": RMS_EPSILON,
             "bound_source_sha256": BOUND_SOURCES}
    fixed.update({"dev_documents": 12, "dev_world_batches": 8, "sample_new_bytes": 32}
                 if stage == "matched_stability" else
                 {"dev_documents": 3, "dev_world_batches": 1, "sample_new_bytes": 0})
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("frozen normalized architecture/objective/source binding mismatch")
    if (type(config.get("max_wall_seconds")) not in (int, float)
            or not 0 < config["max_wall_seconds"] <= (600 if stage == "matched_stability" else 180)
            or type(config.get("max_allocated_bytes")) is not int
            or not 0 < config["max_allocated_bytes"] <= 8 * 1024 ** 3):
        raise ValueError("normalized pilot resource caps exceeded")


def load_bound_trainer(config, device):
    validate_config(config, device)
    for path, digest in BOUND_SOURCES.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError("bound historical source changed: " + path)
    spec = importlib.util.spec_from_file_location("_normalized_language_private_trainer", ROOT / "tools/train_language_pilot.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    trainer.TypedLanguageModel = NormalizedTypedLanguageModel
    original_sources, original_probe = trainer.sources, trainer.training_probe

    def sources():
        result = original_sources()
        for path in (Path(__file__).resolve(), ROOT / "src/controlled_lm/normalized_typed_language_model.py"):
            result[str(path.relative_to(ROOT))] = trainer.sha(path)
        return result

    def training_probe(model, output, update, loss, terminal_query_byte_ce=None):
        result = original_probe(model, output, update, loss, terminal_query_byte_ce)
        # Diagnostic only; detached data, no additional forward, hooks or state.
        def magnitude(value):
            work = value.detach().double()
            return {"rms": float(work.square().mean().sqrt()), "abs_max": float(work.abs().max())}
        result["normalization"] = {"recipe": NORMALIZATION_RECIPE, "epsilon": RMS_EPSILON,
            "locations": ["each SwiGLU input", "shared final input to LM and claim heads"],
            "final_stream": magnitude(output.representation),
            "normalized_final_head_input": magnitude(parameter_free_rms(output.representation.detach())),
            "swiglu_products": [magnitude(value) for value in output.native_activations],
            "raw_byte_logits": magnitude(output.raw_logits), "raw_claim_logits": magnitude(output.proposal_logits)}
        return result

    trainer.sources, trainer.training_probe = sources, training_probe
    return trainer


def main():
    # Parse without consuming/replacing argv: the bound trainer then uses its
    # unchanged parser and preserves its config SHA/output/checkpoint behavior.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    load_bound_trainer(config, args.device).main()


if __name__ == "__main__":
    main()
