"""Bounded from-random byte-language pilot; DEV diagnostics, uncertified prose."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from dataclasses import asdict
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from controlled_lm.language_pilot_data import mixed_byte_batch, world_fold
from controlled_lm.structured_decision import canonical_state_dict_digest
from controlled_lm.typed_language_model import BYTE_START, TypedLanguageModel
from qualify_language_gpu import objective

ROOT = Path(__file__).resolve().parents[1]
MODES = ("none", "internal", "output_only")
CONTINUATION_FIELDS = {"updates", "checkpoints", "prefix_reference_report", "prefix_reference_step", "resume_reference_report", "resume_reference_step"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def recursive_equal(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left, right))
    if type(left) is not type(right): return False
    if isinstance(left, dict): return left.keys() == right.keys() and all(recursive_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)): return len(left) == len(right) and all(recursive_equal(a, b) for a, b in zip(left, right))
    return left == right


def compare_prefix(reference_path, config, mode, step, checkpoint):
    checks = dict.fromkeys(("reference_completed", "config_equal", "reference_checkpoint_sha256_matches", "model_equal",
                           "optimizer_equal", "torch_rng_equal", "cuda_rng_equal", "batch_rng_equal", "batch_digest_equal", "data_hashes_equal", "checkpoint_metadata_equal"), False)
    result = {"reference_report": str(reference_path), "step": step, "checks": checks, "passed": False}
    try:
        current = torch.load(checkpoint, map_location="cpu", weights_only=True)
        result["current_canonical_model_sha256"] = canonical_state_dict_digest(current["model"])
        result["reference_report_sha256"] = sha(reference_path)
        reference = json.loads(reference_path.read_text()); ignored = CONTINUATION_FIELDS
        checks["reference_completed"] = reference["status"] == "complete"
        checks["config_equal"] = {key: value for key, value in config.items() if key not in ignored} == {key: value for key, value in reference["config"].items() if key not in ignored}
        run = next(run for run in reference["runs"] if run["mode"] == mode)
        checks["reference_completed"] &= run["status"] == "complete"
        point = next(point for point in run["curve"] if point["step"] == step)
        path = Path(point["checkpoint"]); path = path if path.is_absolute() else ROOT / path
        result.update(reference_checkpoint=str(path), reference_checkpoint_sha256=sha(path), reference_source_sha256=reference["source_sha256"])
        checks["reference_checkpoint_sha256_matches"] = result["reference_checkpoint_sha256"] == point["checkpoint_sha256"]
        if all(checks[key] for key in ("reference_completed", "config_equal", "reference_checkpoint_sha256_matches")):
            saved = torch.load(path, map_location="cpu", weights_only=True)
            checks["checkpoint_metadata_equal"] = current["mode"] == saved["mode"] == mode and current["step"] == saved["step"] == step
            for key in ("model", "optimizer", "torch_rng", "cuda_rng", "batch_rng"):
                checks[key + "_equal"] = recursive_equal(current[key], saved[key])
            checks["batch_digest_equal"] = current["batch_sha256"] == saved["batch_sha256"] == point["batch_sha256"]
            checks["data_hashes_equal"] = recursive_equal(current["data_sha256"], saved["data_sha256"])
        result["passed"] = all(checks.values())
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
    return result


def resume_identity(reference, config, source, data, runtime):
    def recipe(value): return {key: value for key, value in value.items() if key not in CONTINUATION_FIELDS}
    def fixed_sources(value): return {key: digest for key, digest in value.items() if key.replace("\\", "/") != "tools/train_language_pilot.py"}
    checks = {"completed_own_initial_pretraining": reference.get("status") == "complete" and reference.get("from_random_initialization") is True,
        "all_requested_modes_completed": all(any(run["mode"] == mode and run["status"] == "complete" and any(point["step"] == config.get("resume_reference_step", reference["config"]["updates"]) for point in run["curve"]) for run in reference["runs"]) for mode in config.get("semantic_modes", list(MODES))),
        "recipe_equal": recursive_equal(recipe(config), recipe(reference["config"])),
        "immutable_sources_equal": fixed_sources(source) == fixed_sources(reference["source_sha256"]),
        "data_equal": recursive_equal(data, reference["data_sha256"]),
        "runtime_equal": all(reference.get(key) == value for key, value in runtime.items()),
        "reference_source_data_unchanged": reference.get("source_unchanged") is True and reference.get("data_unchanged") is True}
    if not all(checks.values()): raise ValueError("strict resume identity mismatch: " + ",".join(key for key, passed in checks.items() if not passed))
    return checks


def rebuild_prefix(batches, step):
    digest = hashlib.sha256(); counts = dict.fromkeys(("used_tokens", "utf8_byte_targets", "eos_targets", "story_tokens", "world_tokens"), 0)
    for batch in batches[:step]:
        batch_hash(digest, batch); counts["used_tokens"] += int(batch.attention_mask.sum())
        counts["utf8_byte_targets"] += int((batch.targets >= BYTE_START).sum()); counts["eos_targets"] += int((batch.targets == 2).sum())
        counts["story_tokens"] += int(batch.attention_mask[~batch.world_rows].sum()); counts["world_tokens"] += int(batch.attention_mask[batch.world_rows].sum())
    return digest, counts


def sources():
    paths = [Path(__file__).resolve(), ROOT / "tools/qualify_language_gpu.py"]
    paths += [ROOT / "src/controlled_lm" / name for name in ("language_pilot_data.py", "typed_language_model.py",
        "typed_evidence_lane.py", "parallel_semantic.py", "semantic_transitions.py", "invariant_ssm.py",
        "parallel_ssm.py", "evidence_decoder.py", "policy.py", "structured_decision.py")]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def stories(path, limit=None):
    result = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)["story"]  # No generation/answer metadata is a model input.
            if type(value) is not str or not value:
                raise ValueError("story must be nonempty text")
            result.append(value)
            if limit and len(result) == limit:
                break
    return result


def batch_hash(digest, batch):
    for name in batch.__dataclass_fields__:
        value = getattr(batch, name)
        digest.update(name.encode())
        digest.update(json.dumps([asdict(policy) for policy in value], sort_keys=True).encode() if name == "policies"
                      else str((str(value.dtype), tuple(value.shape))).encode() + value.numpy().tobytes())


def forward(model, batch):
    return model(batch.input_ids, batch.attention_mask, observed_event_ids=batch.observed_event_ids,
                 trusted_event_mask=batch.trusted_event_mask, policies=batch.policies)


def pilot_objective(output, batch, world_answer_weight=0.):
    base = objective(output, batch)
    if world_answer_weight == 0: return base, None  # Preserve the original computation exactly.
    rows = batch.world_rows.nonzero().flatten(); positions = output.final_positions[rows]
    terminal_query_byte_ce = F.cross_entropy(output.raw_logits[rows, positions], batch.targets[rows, positions])
    return base + world_answer_weight * terminal_query_byte_ce, terminal_query_byte_ce


def training_probe(model, output, update, loss, terminal_query_byte_ce=None):
    def gradient_l2(parameters):
        gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
        return math.sqrt(sum(float(gradient.double().square().sum()) for gradient in gradients)) if gradients else None
    result = {"before_update": update, "loss_before_step": float(loss.detach()),
        "global_gradient_l2": gradient_l2(model.parameters()),
        "recurrent_layers": [{"layer": index, "gradient_l2": gradient_l2(core.parameters()),
            "parameter_gradient_l2": {name: gradient_l2([parameter]) for name, parameter in core.named_parameters()},
            "unused_parameters": [name for name, parameter in core.named_parameters() if parameter.grad is None],
            "core_state_abs_max": float(output.core_states[index].detach().abs().max())}
            for index, core in enumerate(model.recurrent_layers)], "gradient_clipping": False}
    if terminal_query_byte_ce is not None: result["terminal_query_byte_ce"] = float(terminal_query_byte_ce.detach())
    return result


def world_identity(batch):
    row = batch.world_rows.nonzero().item()
    events = batch.observed_event_ids[row][batch.trusted_event_mask[row]].tolist()
    if [(event - 12) // 8 for event in events] != list(range(4)):
        raise ValueError("world identity requires one observed SET per key in key order")
    return tuple((event - 12) % 8 for event in events)


def prepare_batches(train, dev, config, guard):
    length, story_rows = config.get("length", 128), config.get("story_rows", 3)
    rng = np.random.default_rng(config.get("dev_seed", 791017))
    story = [mixed_byte_batch(dev[i:i+3], length, rng, random_windows=False, terminal_windows=terminal, world_split="dev")
             for terminal in (False, True) for i in range(0, len(dev), 3)]
    world = [mixed_byte_batch(dev[:3], length, rng, random_windows=False, world_split="dev") for _ in range(config.get("dev_world_batches", 32))]
    rng = np.random.default_rng(config.get("data_seed", 791013)); training, states = [], []
    for _ in range(config["updates"]):
        guard(); training.append(mixed_byte_batch([train[int(i)] for i in rng.integers(len(train), size=story_rows)], length, rng, world_split="train"))
        states.append(rng.bit_generator.state)
    train_ids, dev_ids = [world_identity(batch) for batch in training], [world_identity(batch) for batch in story + world]
    eos = sum(int((batch.targets == 2).sum()) for batch in training)
    if set(train_ids) & set(dev_ids) or any(world_fold(values) < 20 for values in train_ids) or any(not 10 <= world_fold(values) < 20 for values in dev_ids):
        raise AssertionError("whole-ledger TRAIN/DEV partition violation")
    if not eos: raise AssertionError("planned training stream contains no EOS targets")
    audit = {"train_world_identities": sorted(set(train_ids)), "dev_world_identities": sorted(set(dev_ids)),
        "train_world_folds": dict(Counter(world_fold(values) for values in train_ids)), "dev_world_folds": dict(Counter(world_fold(values) for values in dev_ids)),
        "train_world_sequence_sha256": hashlib.sha256(json.dumps(train_ids).encode()).hexdigest(), "planned_train_eos_targets": eos, "overlap": 0,
        "policy": "whole ledger: TRAINfold>=20, DEV10..19, reserved0..9 unopened"}
    return training, states, story, world, audit


@torch.no_grad()
def evaluate(model, story_batches, world_batches, device, precision, guard):
    model.eval()
    story_nll = byte_nll = story_tokens = byte_tokens = 0
    query_correct = raw41_correct = relevance = covered = worlds = 0
    forbidden = 0.
    for is_story, batches in ((True, story_batches), (False, world_batches)):
        for cpu_batch in batches:
            guard(); batch = cpu_batch.to(device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=precision == "bf16_autocast"):
                output = forward(model, batch)
            if not torch.isfinite(output.raw_logits).all() or not torch.isfinite(output.proposal_logits).all():
                raise FloatingPointError("nonfinite DEV logits")
            if is_story:
                losses = F.cross_entropy(output.raw_logits.float().flatten(0, 1), batch.targets.flatten(),
                                         reduction="none").reshape_as(batch.targets)
                mask = (batch.targets != 0) & ~batch.world_rows[:, None]
                byte_mask = (batch.targets >= BYTE_START) & mask
                story_nll += float(losses[mask].sum()); story_tokens += int(mask.sum())
                byte_nll += float(losses[byte_mask].sum()); byte_tokens += int(byte_mask.sum())
                if output.final_evidence[~batch.world_rows].any():
                    raise AssertionError("fiction wrote evidence")
            else:
                row = batch.world_rows.nonzero().item(); position = output.final_positions[row]
                query_correct += int(output.raw_logits[row, position].argmax() == batch.targets[row, position])
                raw41_correct += int(output.proposal_logits[row].argmax() == batch.claim_targets[row])
                predicted = int(output.grounded.probabilities[row].argmax())
                relevance += int(predicted == batch.claim_targets[row]); covered += int(predicted != 40); worlds += 1
                invalid = output.grounded.probabilities.masked_select(~output.grounded.allowed)
                forbidden = max(forbidden, float(invalid.max()) if invalid.numel() else 0.)
    return {"story_mean_nll": story_nll / story_tokens, "story_bits_per_utf8_byte": byte_nll / byte_tokens / math.log(2),
            "story_target_tokens": story_tokens, "story_byte_targets": byte_tokens, "world_rows": worlds,
            "world_query_byte_accuracy": query_correct / worlds, "world_raw41_accuracy": raw41_correct / worlds,
            "grounded_requested_key_claim_accuracy": relevance / worlds, "grounded_nonabstention_coverage": covered / worlds,
            "max_forbidden_probability": forbidden}


@torch.no_grad()
def samples(model, prompts, count, seed, train_bytes, device, precision, guard):
    model.eval(); result = []; generator = torch.Generator(device=device).manual_seed(seed)
    for prompt in prompts:
        tokens = [1, *(BYTE_START + value for value in prompt.encode("utf-8"))]; generated = bytearray()
        for _ in range(count):
            guard()
            with torch.autocast(device, dtype=torch.bfloat16, enabled=precision == "bf16_autocast"):
                output = model(torch.tensor([tokens], device=device))  # Language lane only.
            logits = output.raw_logits[0, -1].float(); logits[:2] = -torch.inf
            probabilities = logits.softmax(-1)
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("nonfinite uncertified prose distribution")
            token = int(torch.multinomial(probabilities, 1, generator=generator))
            if token == 2:
                break
            tokens.append(token); generated.append(token - BYTE_START)
        prefix = bytes(generated[:32])
        result.append({"prompt": prompt, "continuation": generated.decode("utf-8", errors="replace"),
                       "generated_bytes": len(generated), "certification": "UNCERTIFIED raw prose",
                       "first32_bytes_exact_train_substring": any(prefix in text for text in train_bytes) if len(prefix) == 32 else None,
                       "copy_check_scope": "first32 bytes only; no near-copy or memorization guarantee"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda"); args = parser.parse_args()
    config = json.loads(args.config.read_text()); output = args.output.resolve()
    checkpoints = output.with_name(output.stem + "_checkpoints")
    if output.exists() or checkpoints.exists() or output.with_suffix(output.suffix + ".tmp").exists():
        raise FileExistsError("choose unique output and checkpoint paths")
    device, precision = args.device, config.get("precision", "fp32")
    world_answer_weight = config.get("world_answer_weight", 0.)
    if type(world_answer_weight) not in (int, float) or not math.isfinite(world_answer_weight) or not 0 <= world_answer_weight <= 1:
        raise ValueError("world_answer_weight must be finite numeric in[0,1], excluding bool")
    world_answer_weight = float(world_answer_weight)
    modes = config.get("semantic_modes", list(MODES)); recurrent_residual = config.get("recurrent_residual", False)
    if (type(modes) is not list or not modes or any(type(mode) is not str or mode not in MODES for mode in modes)
            or len(set(modes)) != len(modes)):
        raise ValueError("semantic_modes must be a nonempty unique subset of none/internal/output_only")
    if type(recurrent_residual) is not bool: raise ValueError("recurrent_residual must be bool")
    resume_path = config.get("resume_reference_report"); resume = None; start_step = 0
    if resume_path is not None:
        if type(resume_path) is not str: raise ValueError("resume_reference_report must be a report path")
        resume_path = ROOT / resume_path; resume = json.loads(resume_path.read_text())
        start_step = config.get("resume_reference_step", resume["config"]["updates"])
        if type(start_step) is not int or not 0 < start_step <= resume["config"]["updates"] or start_step >= config["updates"]: raise ValueError("resume step must precede the cumulative target and exist in the completed reference")
    points = sorted(set(config.get("checkpoints", [start_step, config["updates"]] if resume else [0, 250, 500, 1000])))
    if precision not in ("fp32", "bf16_autocast") or not points or points[0] != start_step or points[-1] != config["updates"]:
        raise ValueError("precision/checkpoints invalid; checkpoints must include start and final update")
    reference = config.get("prefix_reference_report"); reference_step = config.get("prefix_reference_step", 1000)
    if reference is not None and (type(reference) is not str or type(reference_step) is not int or reference_step <= 0 or reference_step not in points):
        raise ValueError("prefix reference requires a report path and a positive checkpoint step")
    cap = 5000 if resume else 3000
    if not 0 < config["updates"] <= cap or not 72 <= config.get("length", 128) <= 128:
        raise ValueError(f"bounded pretraining requires at most{cap} updates and length72..128")
    if config.get("projection_layout", "shared") != "shared":
        raise ValueError("pilot contrasts require shared bias-free evidence projection")
    if type(config.get("story_rows", 3)) is not int or not 3 <= config.get("story_rows", 3) <= 63:
        raise ValueError("story_rows must be an integer in3..63; one additional world row")
    if (not 0 < config.get("max_wall_seconds", 600) <= 600 or not 1 <= config.get("dev_documents", 32) <= 32
            or not 1 <= config.get("dev_world_batches", 32) <= 64 or not 0 <= config.get("sample_new_bytes", 96) <= 128):
        raise ValueError("pilot wall/DEV/sample caps exceeded")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    allocated = min(config.get("max_allocated_bytes", 8 * 1024 ** 3), 8 * 1024 ** 3)
    free = total = None
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.get_device_name() != "NVIDIA GeForce RTX 5070 Ti Laptop GPU":
            raise RuntimeError("only authorized local5070Ti Laptop GPU")
        torch.cuda.set_per_process_memory_fraction(.65); torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        free, total = torch.cuda.mem_get_info(); allocated = min(allocated, int(.65 * total))
    began = time.perf_counter()
    def guard():
        if time.perf_counter() - began > config.get("max_wall_seconds", 600):
            raise TimeoutError("pilot wall budget exhausted")
        if device == "cuda" and torch.cuda.max_memory_allocated() > allocated:
            raise MemoryError("pilot allocation budget exhausted")
    manifest_path = ROOT / config.get("manifest", "results/language_pilot_extraction.json")
    manifest = json.loads(manifest_path.read_text()); paths = {split: ROOT / manifest["splits"][split]["path"] for split in ("train", "dev")}
    def data_hashes():
        return {"manifest": sha(manifest_path), **{split: sha(path) for split, path in paths.items()}}
    data_before, source_before = data_hashes(), sources()
    if any(data_before[split] != manifest["splits"][split]["sha256"] for split in paths):
        raise ValueError("extracted story hashes differ from reviewed manifest")
    runtime = {"torch_version": str(torch.__version__), "device": device, "device_name": torch.cuda.get_device_name() if device == "cuda" else "cpu", "precision": precision}
    resume_checks = resume_identity(resume, config, source_before, data_before, runtime) if resume else None
    resume_report_sha = sha(resume_path) if resume else None
    train = stories(paths["train"]); dev = stories(paths["dev"], config.get("dev_documents", 32)); train_bytes = [text.encode() for text in train]
    training_batches, batch_states, story_batches, world_batches, world_audit = prepare_batches(train, dev, config, guard)
    prefix_digest, prefix_counts = rebuild_prefix(training_batches, start_step)
    length = config.get("length", 128)
    dev_hash = hashlib.sha256()
    for batch in story_batches + world_batches: batch_hash(dev_hash, batch)
    prompts = config.get("prompts", ["Once upon a time,", "The little dog"])
    if not 1 <= len(prompts) <= 3 or any(type(prompt) is not str or len(prompt.encode()) > 64 for prompt in prompts):
        raise ValueError("use one to three preregistered prompts, each at most64 UTF8 bytes")
    payload = {"status": "running", "development_only": True, "official_test_evaluated": False, "paid_compute": False,
        "from_random_initialization": True, "config": config, "config_sha256": sha(args.config), "source_sha256": source_before,
        "fresh_this_invocation": resume is None, "continuation_of_own_initial_pretraining": resume is not None, "start_step": start_step,
        "resume_identity_checks": resume_checks, "source_chain": ([] if resume is None else list(resume.get("source_chain", [])) + [{"report": str(resume_path), "report_sha256": resume_report_sha, "step": start_step, "source_sha256": resume["source_sha256"]}]),
        "data_sha256": data_before, "semantic_modes": modes, "recurrent_residual": recurrent_residual,
        "device": device, "device_name": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "torch_version": torch.__version__, "free_vram_bytes": free, "total_vram_bytes": total, "precision": precision, "master_parameters": "float32 Adam",
        "loss_weights_not_token_mixture": {"story_lm": .75, "world_lm": .25, "world_raw41": .2, "terminal_query_byte_ce": world_answer_weight}, "story_rows_per_update": config.get("story_rows", 3), "world_audit": world_audit,
        "gradient_clipping": False, "dev_selection": {"first_documents": len(dev), "prefix_length": length,
        "story_windows": ["fixed prefix", "fixed terminal including EOS"], "story_batches": len(story_batches),
        "story_eos_targets": sum(int((batch.targets[~batch.world_rows] == 2).sum()) for batch in story_batches),
        "world_batches": len(world_batches), "batch_sha256": dev_hash.hexdigest()}, "prompts": prompts, "runs": []}
    output.parent.mkdir(parents=True, exist_ok=True); checkpoints.mkdir()
    with output.open("x") as stream: json.dump(payload, stream)
    def preserve(path, step):
        temporary = path.with_suffix(".pt.tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "mode": mode, "step": step,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
            "batch_rng": rng.bit_generator.state, "config": config, "config_sha256": payload["config_sha256"],
            "last_training_loss": entry.get("last_training_loss"), "last_training_loss_before_update": entry.get("last_training_loss_before_update"),
            "original_initial_parameters_sha256": entry["initial_parameters_sha256"], "fresh_this_invocation": resume is None, "source_chain": payload["source_chain"],
            "source_sha256": source_before, "data_sha256": data_before, "batch_sha256": digest.hexdigest(), "development_only": True}, temporary)
        temporary.replace(path)
    def progress():
        entry.update(cumulative_updates=completed_updates, new_optimizer_updates=completed_updates - start_step,
                     cumulative_used_tokens=used_tokens, new_used_tokens=used_tokens - prefix_counts["used_tokens"], new_wall_seconds=time.perf_counter() - arm_began)
    initial_digest = None; matched_batches = {}
    state_ready = False
    try:
        for mode in modes:
            state_ready = False; arm_began = time.perf_counter()
            guard(); torch.manual_seed(config.get("seed", 791011)); rng = np.random.default_rng(config.get("data_seed", 791013))
            model = TypedLanguageModel(config["hidden_size"], config["native_size"], num_layers=config["num_layers"],
                recurrent_kind=config.get("recurrent_kind", "ssm"), semantic_mode=mode, ssm_backend=config.get("ssm_backend", "parallel"),
                projection_layout="shared", recurrent_residual=recurrent_residual).to(device, dtype=torch.float32)
            prior = config.get("ssm_gate_bias_range")
            if prior is not None:
                if model.recurrent_kind != "ssm" or len(prior) != 2 or not all(math.isfinite(value) for value in prior) or prior[0] > prior[1]: raise ValueError("invalid fresh SSM gate bias prior")
                with torch.no_grad():
                    for core in model.recurrent_layers: core.input_projection.bias[:model.hidden_size].copy_(torch.linspace(*prior, model.hidden_size, device=device))
            initial = canonical_state_dict_digest(model)
            if initial_digest is not None and initial != initial_digest: raise AssertionError("arm initialization mismatch")
            initial_digest = initial; optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", .0003))
            entry = {"mode": mode, "initial_parameters_sha256": initial, "parameters": sum(p.numel() for p in model.parameters()),
                "semantic_active_parameters": 0 if mode == "none" else sum(p.numel() for p in model.semantic_projections.parameters()),
                "fresh_trainable_ssm_gate_bias_range": prior, "recurrent_residual": recurrent_residual,
                "fresh_this_invocation": resume is None, "start_step": start_step, "new_optimizer_updates": 0, "status": "running", "training_probes": [], "curve": []}; payload["runs"].append(entry)
            digest = prefix_digest.copy(); used_tokens = prefix_counts["used_tokens"]; byte_targets = prefix_counts["utf8_byte_targets"]
            completed_updates = start_step; eos_targets = prefix_counts["eos_targets"]; story_tokens = prefix_counts["story_tokens"]; world_tokens = prefix_counts["world_tokens"]
            if resume:
                if sha(resume_path) != resume_report_sha: raise AssertionError("resume report changed")
                ref_run = next(run for run in resume["runs"] if run["mode"] == mode)
                ref_point = next(point for point in ref_run["curve"] if point["step"] == start_step)
                ref_path = Path(ref_point["checkpoint"]); ref_path = ref_path if ref_path.is_absolute() else ROOT / ref_path
                if ref_run["status"] != "complete" or sha(ref_path) != ref_point["checkpoint_sha256"]: raise ValueError("resume reference checkpoint mismatch")
                saved = torch.load(ref_path, map_location="cpu", weights_only=True)
                if saved["config"] != resume["config"] or saved["mode"] != mode or saved["step"] != start_step or saved["source_sha256"] != resume["source_sha256"] or saved["data_sha256"] != data_before: raise ValueError("resume checkpoint metadata mismatch")
                if initial != ref_run["initial_parameters_sha256"]: raise AssertionError("resume original initialization mismatch")
                rebuilt = {"batch_digest": digest.hexdigest() == saved["batch_sha256"] == ref_point["batch_sha256"], "batch_rng": recursive_equal(batch_states[start_step - 1], saved["batch_rng"]), "counters": all(ref_point[key] == value for key, value in prefix_counts.items())}
                entry["reconstructed_prefix_checks"] = rebuilt
                if not all(rebuilt.values()): raise AssertionError("resume reconstructed prefix mismatch")
                model.load_state_dict(saved["model"], strict=True); optimizer.load_state_dict(saved["optimizer"])
                rng.bit_generator.state = batch_states[start_step - 1]; torch.set_rng_state(saved["torch_rng"])
                if device == "cuda": torch.cuda.set_rng_state(saved["cuda_rng"])
                elif saved["cuda_rng"] is not None: raise ValueError("CPU resume cannot restore CUDA RNG")
                entry.update(last_training_loss=saved.get("last_training_loss"), last_training_loss_before_update=saved.get("last_training_loss_before_update"))
                del saved
            entry["loaded_start_parameters_sha256"] = canonical_state_dict_digest(model); state_ready = True; progress()
            for step in range(start_step, config["updates"] + 1):
                guard()
                if step in points:
                    progress()
                    checkpoint = checkpoints / f"{mode}_{step:04d}.pt"; preserve(checkpoint, step)  # Before DEV diagnostics.
                    point = {"step": step, "used_tokens": used_tokens, "utf8_byte_targets": byte_targets, "eos_targets": eos_targets, "story_tokens": story_tokens, "world_tokens": world_tokens, "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha(checkpoint), "batch_sha256": digest.hexdigest(), "last_training_loss": entry.get("last_training_loss")}; entry["curve"].append(point); write(output, payload)
                    key = str(step)
                    if key in matched_batches and matched_batches[key] != digest.hexdigest(): raise AssertionError("arm batch mismatch")
                    matched_batches[key] = digest.hexdigest()
                    if resume is not None and step == start_step:
                        point["resume_reference"] = compare_prefix(resume_path, config, mode, step, checkpoint); write(output, payload)
                        if not point["resume_reference"]["passed"]: raise AssertionError("exact resume reference agreement failed")
                    if reference is not None and step == reference_step:
                        point["prefix_reference"] = compare_prefix(ROOT / reference, config, mode, step, checkpoint); write(output, payload)
                        if not point["prefix_reference"]["passed"]: raise AssertionError("exact prefix reference agreement failed")
                    point["dev"] = evaluate(model, story_batches, world_batches, device, precision, guard)
                    point["samples"] = samples(model, prompts, config.get("sample_new_bytes", 96), config.get("sample_seed", 791019), train_bytes, device, precision, guard)
                    point["wall_seconds"] = time.perf_counter() - began; write(output, payload)
                if step == config["updates"]: break
                cpu_batch = training_batches[step]; rng.bit_generator.state = batch_states[step]
                batch_hash(digest, cpu_batch); batch = cpu_batch.to(device); model.train(); optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device, dtype=torch.bfloat16, enabled=precision == "bf16_autocast"):
                    training_output = forward(model, batch); loss, terminal_query_byte_ce = pilot_objective(training_output, batch, world_answer_weight)
                entry.update(last_training_loss=float(loss.detach()) if torch.isfinite(loss) else None,
                             last_training_loss_before_update=step + 1, last_training_loss_finite=bool(torch.isfinite(loss)))
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite loss; no clipping fallback")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError("nonfinite gradient")
                if step == start_step or step + 1 in points: entry["training_probes"].append(training_probe(model, training_output, step + 1, loss, terminal_query_byte_ce))
                del training_output
                optimizer.step(); completed_updates = step + 1
                used_tokens += int(cpu_batch.attention_mask.sum()); byte_targets += int((cpu_batch.targets >= BYTE_START).sum())
                eos_targets += int((cpu_batch.targets == 2).sum()); story_tokens += int(cpu_batch.attention_mask[~cpu_batch.world_rows].sum()); world_tokens += int(cpu_batch.attention_mask[cpu_batch.world_rows].sum())
            progress(); entry.update(status="complete", trained_batch_sha256=digest.hexdigest(), source_unchanged=sources() == source_before, data_unchanged=data_hashes() == data_before)
            if not entry["source_unchanged"] or not entry["data_unchanged"]: raise AssertionError("source/data changed during pilot")
            write(output, payload); del model, optimizer
            if device == "cuda": torch.cuda.empty_cache()
        payload["status"] = "complete"
    except Exception as error:
        if payload["runs"] and payload["runs"][-1]["status"] == "running": payload["runs"][-1]["status"] = "failed"
        if "model" in locals() and "optimizer" in locals() and state_ready:
            progress()
            interrupted = checkpoints / f"{mode}_interrupted.pt"
            try: preserve(interrupted, completed_updates); entry["interrupted_checkpoint"] = str(interrupted)
            except Exception as preservation_error: entry["preservation_error"] = str(preservation_error)
        payload.update(status="failed", error_type=type(error).__name__, error=str(error)); raise
    finally:
        payload.update(wall_seconds=time.perf_counter()-began, source_unchanged=sources() == source_before, data_unchanged=data_hashes() == data_before)
        payload["cumulative_invocation_wall_seconds"] = payload["wall_seconds"] + (resume.get("cumulative_invocation_wall_seconds", resume["wall_seconds"]) if resume else 0.)
        if device == "cuda": torch.cuda.synchronize(); payload["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        write(output, payload)


if __name__ == "__main__": main()
