"""Shared helpers for the routing-arm seed replication package.

Nothing here edits or replaces repository sources. The package drives the
retained normalized trainer (tools/train_normalized_language_pilot.py) and the
bound historical trainer it loads, and reads their reports/checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS_DIR = HERE / "runs"          # raw trainer reports + checkpoints (git-ignored)
RESULTS_DIR = HERE / "results"    # redacted, path-free summaries meant for commit

DEFAULT_BASE_CONFIG = Path("configs/normalized_language_local_20261021.json")
ORIGINAL_RECEIPT = Path("results/normalized_language_local_20261021.derived.json")
MODES = ("none", "internal", "output_only")
QUALIFIED_DEVICE = "NVIDIA GeForce RTX 5070 Ti Laptop GPU"  # string checked by the bound trainer

# Only these config keys vary between replicates. Everything else (including
# dev_seed and sample_seed, so in-trainer DEV batches are identical across
# replicates) stays byte-for-byte equal to the base config.
VARIABLE_KEYS = ("seed", "data_seed")
SEED_STRIDE = 100


def setup_import_paths() -> None:
    """Make `controlled_lm` and the tools/ modules importable, as the trainer expects."""
    for path in (ROOT / "src", ROOT / "tools"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def seeds_for_replicate(base_config: dict, replicate: int) -> dict:
    """Replicate 0 = the original seeds; replicate k = original + 100*k (preregistered)."""
    if type(replicate) is not int or replicate < 0 or replicate > 999:
        raise ValueError("replicate index must be an integer in 0..999")
    return {key: base_config[key] + SEED_STRIDE * replicate for key in VARIABLE_KEYS}


def replicate_config_bytes(base_path: Path, replicate: int) -> bytes:
    """Replicate 0 reuses the base config bytes exactly (same config SHA as the original run)."""
    raw = base_path.read_bytes()
    if replicate == 0:
        return raw
    config = json.loads(raw)
    config.update(seeds_for_replicate(config, replicate))
    return (json.dumps(config, indent=2) + "\n").encode("utf-8")


def check_replication_config(config: dict, base: dict) -> None:
    """Allow ONLY seed/data_seed to differ from the base config."""
    missing = object()
    differing = {key for key in set(config) | set(base) if config.get(key, missing) != base.get(key, missing)}
    if not differing <= set(VARIABLE_KEYS):
        raise ValueError("replicate config differs from base outside seed/data_seed: " + ",".join(sorted(differing)))
    for key in VARIABLE_KEYS:
        value = config[key]
        if type(value) is not int or not 0 <= value < 2 ** 63:
            raise ValueError(f"{key} must be a non-negative 64-bit integer")
    reserved = {base.get("dev_seed"), base.get("sample_seed")}
    if {config["seed"], config["data_seed"]} & reserved or config["seed"] == config["data_seed"]:
        raise ValueError("replicate seeds must not collide with dev/sample seeds or each other")


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rel(path: Path) -> str:
    """Repository-relative POSIX path; never export absolute host paths."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return "<outside-repo>/" + path.name


def write_json_new(path: Path, payload, *, allow_identical: bool = False) -> None:
    """Write JSON atomically; refuse to overwrite different existing content."""
    text = json.dumps(payload, indent=2, allow_nan=False, sort_keys=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if allow_identical and path.read_text(encoding="utf-8") == text:
            return
        raise FileExistsError(f"refusing to overwrite {rel(path)}; choose a new --tag or output path")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def replicate_name(replicate: int) -> str:
    return f"seed_{replicate:02d}"


def resolve_checkpoint(report_path: Path, point: dict, mode: str) -> Path:
    """Use the report's checkpoint path; fall back to the sibling checkpoint dir if moved."""
    recorded = Path(point["checkpoint"])
    recorded = recorded if recorded.is_absolute() else ROOT / recorded
    if recorded.exists():
        return recorded
    fallback = report_path.with_name(report_path.stem + "_checkpoints") / f"{mode}_{point['step']:04d}.pt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"checkpoint for {mode} step {point['step']} not found near {rel(report_path)}")


def load_report(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_replicate_reports(tag: str) -> list[tuple[int, Path]]:
    folder = RUNS_DIR / tag
    found = []
    for path in sorted(folder.glob("seed_*.json")):
        stem = path.stem
        if stem.count("_") != 1 or not stem.split("_")[1].isdigit():
            continue
        found.append((int(stem.split("_")[1]), path))
    return found
