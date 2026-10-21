"""Run the retained normalized trainer with a replicate-specific seed pair.

Why this wrapper exists: tools/train_normalized_language_pilot.py freezes
seed=2026102131 / data_seed=2026102133 inside validate_config(). Varying the
seed is the experimental variable here, so this wrapper

  1. checks the replicate config differs from the base config ONLY in
     seed/data_seed (common.check_replication_config);
  2. passes a view of the replicate config with the frozen base seeds restored
     to the unchanged load_bound_trainer(), so the original validator still
     enforces stage/device/updates/checkpoints/architecture/objective/caps and
     the bound-source SHA-256 ledger;
  3. hands the replicate config FILE to the bound trainer's unchanged main(),
     which records that file's SHA-256 and seeds in its own report.

Device admission is untouched: with --device cuda the bound trainer still
refuses anything except the historically qualified laptop GPU.

--cpu-smoke is a CPU-only test harness mode. It swaps the stage validator
(in this process only, no file edits) for one that admits ONLY configs whose
pilot_stage is "replication_cpu_smoke", with <=10 updates and an unchanged
architecture. Reports from it carry that stage name and analyze.py refuses
to call them confirmatory.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import VARIABLE_KEYS, check_replication_config, setup_import_paths  # noqa: E402

setup_import_paths()
import train_normalized_language_pilot as normalized_wrapper  # noqa: E402

SMOKE_STAGE = "replication_cpu_smoke"
SMOKE_FIXED_KEYS = ("hidden_size", "native_size", "num_layers", "recurrent_kind", "ssm_backend",
                    "projection_layout", "recurrent_residual", "semantic_modes", "precision",
                    "ssm_gate_bias_range", "story_rows", "length", "learning_rate", "world_answer_weight",
                    "normalization_recipe", "normalization_epsilon", "bound_source_sha256")


def smoke_validate_config(config, device):
    """CPU-only replacement validator used exclusively by --cpu-smoke."""
    if device != "cpu" or config.get("pilot_stage") != SMOKE_STAGE:
        raise ValueError("cpu smoke validator admits only device=cpu and pilot_stage=" + SMOKE_STAGE)
    updates = config.get("updates")
    if type(updates) is not int or not 1 <= updates <= 10 or config.get("checkpoints") != [0, updates]:
        raise ValueError("cpu smoke requires 1..10 updates and checkpoints [0, updates]")
    reference = json.loads((normalized_wrapper.ROOT / "configs/normalized_language_local_20261021.json").read_text())
    if any(config.get(key) != reference[key] for key in SMOKE_FIXED_KEYS):
        raise ValueError("cpu smoke must keep the frozen architecture/objective/source binding")
    if any(key in config for key in ("resume_reference_report", "resume_reference_step",
                                     "prefix_reference_report", "prefix_reference_step")):
        raise ValueError("cpu smoke must start fresh")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="replicate config (seed/data_seed varied)")
    parser.add_argument("--base-config", type=Path, required=True, help="frozen base config it must match")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    check_replication_config(config, base)
    frozen_view = dict(config)
    frozen_view.update({key: base[key] for key in VARIABLE_KEYS})
    if args.cpu_smoke:
        if args.device != "cpu":
            raise ValueError("--cpu-smoke is CPU-only")
        normalized_wrapper.validate_config = smoke_validate_config  # this process only
    trainer = normalized_wrapper.load_bound_trainer(frozen_view, args.device)
    # The bound trainer's unchanged parser reads these; it re-reads the replicate
    # config file, refuses existing outputs and records the config SHA-256.
    sys.argv = [str(Path(normalized_wrapper.__file__)), "--config", str(args.config),
                "--output", str(args.output), "--device", args.device]
    trainer.main()


if __name__ == "__main__":
    main()
