"""Expanded DEV evaluation of every arm/replicate checkpoint (no training, no TEST).

World probe. The procedural world generator is mixed_byte_batch() in
src/controlled_lm/language_pilot_data.py: a ledger is 4 keys (A-D) with values
0..7 plus a queried key, 8^4 * 4 = 16,384 possible questions. Membership is by
ledger via world_fold(): fold >= 20 TRAIN, 10..19 DEV, 0..9 RESERVED (never
opened here). We enumerate EVERY DEV-fold ledger x 4 queries, drop the ledgers
the trainer already used in its in-run DEV batches (the original 8-row probe),
and check per replicate that none of the remaining ledgers occurs among that
run's actual training world identities (read from the raw trainer report).
Rows are rendered by render_world_row(), which is verified tensor-for-tensor
against mixed_byte_batch() before use.

Stories. All DEV documents (819 in the pinned extraction):
  trainer_windows : the trainer's own protocol (fixed 128-token prefix window +
                    fixed terminal window with EOS) applied to every DEV doc;
  tiled_full      : every UTF-8 byte of every DEV doc scored exactly once, in
                    BOS-prefixed windows of <= length-1 bytes (+ EOS at doc end).
Both report bits per UTF-8 byte (EOS excluded, as in the trainer) and counts.

Sanity replay. For each checkpoint the trainer's in-run DEV batches are rebuilt
(batch digest must equal the report's) and scored with the trainer's own
evaluate(); differences from the report's final DEV metrics are recorded.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MODES, RESULTS_DIR, ROOT, list_replicate_reports, load_report, rel,  # noqa: E402
                    resolve_checkpoint, setup_import_paths, sha256_file, write_json_new)

setup_import_paths()
from controlled_lm.language_pilot_data import LanguageBatch, mixed_byte_batch, world_fold  # noqa: E402
from controlled_lm.normalized_typed_language_model import NormalizedTypedLanguageModel  # noqa: E402
from controlled_lm.policy import Policy, READ  # noqa: E402
from controlled_lm.typed_language_model import BOS_ID, BYTE_START, EOS_ID  # noqa: E402

ABSTAIN_ID = 40
WORLD_POLICY = Policy(caps=READ, budget=8, witness_valid=True, allowed_operands=(0, 1, 2, 3))
STORY_POLICY = Policy()


def load_trainer_module():
    """Import the bound historical trainer as a library (its main() is not run)."""
    spec = importlib.util.spec_from_file_location("_routing_replication_trainer_lib", ROOT / "tools/train_language_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- worlds
def render_world_row(values, query, length):
    """Exactly the world row mixed_byte_batch() renders for (values, query)."""
    prefix = bytearray(b"Facts: ")
    positions = []
    for key, value in enumerate(values):
        prefix.extend(f"{chr(65 + key)}={value}; ".encode())
        positions.append(len(prefix))
    prefix.extend(f"\nQuestion: value of {chr(65 + query)}?\nAnswer: ".encode())
    context = [BOS_ID, *(BYTE_START + byte for byte in prefix)]
    if len(context) > length:
        raise ValueError("world prefix exceeds length")
    answer = BYTE_START + ord(str(values[query]))
    ids = torch.zeros(length, dtype=torch.long); targets = torch.zeros_like(ids); events = torch.zeros_like(ids)
    valid = torch.zeros(length, dtype=torch.bool); trusted = torch.zeros_like(valid)
    count = len(context)
    ids[:count] = torch.tensor(context); targets[:count] = torch.tensor([*context[1:], answer]); valid[:count] = True
    for key, position in enumerate(positions):
        events[position] = 12 + key * 8 + values[key]; trusted[position] = True
    return ids, targets, valid, events, trusted, query * 8 + values[query]


def world_batch(questions, length):
    rows = [render_world_row(values, query, length) for values, query in questions]
    stack = lambda index: torch.stack([row[index] for row in rows])
    return LanguageBatch(stack(0), stack(1), stack(2), stack(3), stack(4),
                         torch.tensor([row[5] for row in rows], dtype=torch.long),
                         torch.ones(len(rows), dtype=torch.bool), tuple(WORLD_POLICY for _ in rows))


def verify_renderer(length, draws=256, seed=20261021):
    """Compare render_world_row() with mixed_byte_batch()'s world row, all tensors."""
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        batch = mixed_byte_batch(["x"], length, rng, random_windows=False, world_split="dev")
        row = int(batch.world_rows.nonzero().item())
        events = batch.observed_event_ids[row][batch.trusted_event_mask[row]].tolist()
        values = [(event - 12) % 8 for event in events]
        query = int(batch.claim_targets[row]) // 8
        mine = render_world_row(values, query, length)
        theirs = (batch.input_ids[row], batch.targets[row], batch.attention_mask[row],
                  batch.observed_event_ids[row], batch.trusted_event_mask[row], int(batch.claim_targets[row]))
        if not all(torch.equal(a, b) for a, b in zip(mine[:5], theirs[:5])) or mine[5] != theirs[5]:
            raise AssertionError("render_world_row disagrees with mixed_byte_batch")
        if batch.policies[row] != WORLD_POLICY:
            raise AssertionError("world policy differs from mixed_byte_batch")
    return {"draws": draws, "seed": seed, "identical": True}


def build_question_set(trainer_dev_ledgers):
    dev_ledgers = [values for values in itertools.product(range(8), repeat=4) if 10 <= world_fold(list(values)) < 20]
    excluded = set(map(tuple, trainer_dev_ledgers))
    eval_ledgers = [values for values in dev_ledgers if values not in excluded]
    questions = [(values, query) for values in eval_ledgers for query in range(4)]
    return dev_ledgers, eval_ledgers, questions


# --------------------------------------------------------------------------- stories
def story_rows_batch(windows, length):
    """windows: list of (token list incl. BOS and optional EOS). Targets = tokens[1:]."""
    batch = len(windows)
    ids = torch.zeros(batch, length, dtype=torch.long); targets = torch.zeros_like(ids)
    valid = torch.zeros(batch, length, dtype=torch.bool)
    for row, tokens in enumerate(windows):
        count = min(length, len(tokens) - 1)
        ids[row, :count] = torch.tensor(tokens[:count]); targets[row, :count] = torch.tensor(tokens[1:count + 1])
        valid[row, :count] = True
    return LanguageBatch(ids, targets, valid, torch.zeros_like(ids), torch.zeros_like(valid),
                         torch.full((batch,), ABSTAIN_ID, dtype=torch.long), torch.zeros(batch, dtype=torch.bool),
                         tuple(STORY_POLICY for _ in range(batch)))


def tiled_windows(text, length):
    raw = text.encode("utf-8")
    span = length - 1
    windows = []
    for offset in range(0, len(raw), span):
        segment = raw[offset:offset + span]
        ending = [EOS_ID] if offset + len(segment) == len(raw) else []
        windows.append([BOS_ID, *(BYTE_START + byte for byte in segment), *ending])
    return windows


def trainer_window_batches(docs, length, dev_seed, chunk):
    rng = np.random.default_rng(dev_seed)  # consumed only by the (ignored) world row
    return [mixed_byte_batch(docs[i:i + chunk], length, rng, random_windows=False, terminal_windows=terminal,
                             world_split="dev")
            for terminal in (False, True) for i in range(0, len(docs), chunk)]


# --------------------------------------------------------------------------- scoring
def forward(model, batch):
    return model(batch.input_ids, batch.attention_mask, observed_event_ids=batch.observed_event_ids,
                 trusted_event_mask=batch.trusted_event_mask, policies=batch.policies)


@torch.no_grad()
def score_stories(model, batches, device):
    nll = byte_nll = 0.0
    tokens = byte_tokens = eos_tokens = rows = 0
    for cpu_batch in batches:
        batch = cpu_batch.to(device)
        output = forward(model, batch)
        if not torch.isfinite(output.raw_logits).all():
            raise FloatingPointError("nonfinite story logits")
        story = ~batch.world_rows
        if output.final_evidence[story].any():
            raise AssertionError("fiction wrote evidence")
        losses = F.cross_entropy(output.raw_logits.float().flatten(0, 1), batch.targets.flatten(),
                                 reduction="none").reshape_as(batch.targets)
        mask = (batch.targets != 0) & story[:, None]
        byte_mask = (batch.targets >= BYTE_START) & mask
        nll += float(losses[mask].double().sum()); tokens += int(mask.sum())
        byte_nll += float(losses[byte_mask].double().sum()); byte_tokens += int(byte_mask.sum())
        eos_tokens += int(((batch.targets == EOS_ID) & mask).sum()); rows += int(story.sum())
    return {"mean_nll_nats": nll / tokens, "bits_per_utf8_byte": byte_nll / byte_tokens / math.log(2),
            "target_tokens": tokens, "utf8_byte_targets": byte_tokens, "eos_targets": eos_tokens, "windows": rows}


@torch.no_grad()
def score_worlds(model, questions, length, device, batch_rows):
    typed = coverage = raw41 = query = 0
    answer_nll_bits = typed_target_prob = 0.0
    forbidden_max = forbidden_sum = 0.0
    per_key = {key: [0, 0] for key in "ABCD"}
    bits = {"typed": [], "raw41": [], "query_byte": []}
    for start in range(0, len(questions), batch_rows):
        chunk = questions[start:start + batch_rows]
        batch = world_batch(chunk, length).to(device)
        output = forward(model, batch)
        if not torch.isfinite(output.raw_logits).all() or not torch.isfinite(output.proposal_logits).all():
            raise FloatingPointError("nonfinite world logits")
        rows = torch.arange(len(chunk), device=device)
        positions = output.final_positions
        answer_logits = output.raw_logits[rows, positions].float()
        answer_targets = batch.targets[rows, positions]
        query_hit = answer_logits.argmax(-1) == answer_targets
        answer_nll_bits += float(F.cross_entropy(answer_logits, answer_targets, reduction="sum")) / math.log(2)
        raw_hit = output.proposal_logits.argmax(-1) == batch.claim_targets
        probabilities = output.grounded.probabilities
        allowed = output.grounded.allowed
        predicted = probabilities.argmax(-1)
        typed_hit = predicted == batch.claim_targets
        typed_target_prob += float(probabilities[rows, batch.claim_targets].sum())
        forbidden = torch.where(allowed, torch.zeros_like(probabilities), probabilities).sum(-1)
        forbidden_max = max(forbidden_max, float(forbidden.max())); forbidden_sum += float(forbidden.sum())
        typed += int(typed_hit.sum()); coverage += int((predicted != ABSTAIN_ID).sum())
        raw41 += int(raw_hit.sum()); query += int(query_hit.sum())
        for (values, key), hit in zip(chunk, typed_hit.tolist()):
            per_key["ABCD"[key]][0] += int(hit); per_key["ABCD"[key]][1] += 1
        bits["typed"] += typed_hit.int().tolist(); bits["raw41"] += raw_hit.int().tolist()
        bits["query_byte"] += query_hit.int().tolist()
    n = len(questions)
    return {"questions": n, "typed_requested_key_accuracy": typed / n, "typed_correct": typed,
            "typed_nonabstention_coverage": coverage / n, "typed_target_probability_mean": typed_target_prob / n,
            "raw41_accuracy": raw41 / n, "raw41_correct": raw41,
            "query_byte_accuracy": query / n, "query_byte_correct": query,
            "answer_byte_nll_bits_mean": answer_nll_bits / n,
            "forbidden_mass_max": forbidden_max, "forbidden_mass_mean": forbidden_sum / n,
            "typed_accuracy_by_query_key": {key: hits / total for key, (hits, total) in per_key.items() if total},
            "per_question_correct": {name: "".join(map(str, values)) for name, values in bits.items()}}


def build_model(config, mode, state, device):
    model = NormalizedTypedLanguageModel(config["hidden_size"], config["native_size"], num_layers=config["num_layers"],
                                         recurrent_kind=config["recurrent_kind"], semantic_mode=mode,
                                         ssm_backend=config["ssm_backend"], projection_layout="shared",
                                         recurrent_residual=config["recurrent_residual"])
    model.load_state_dict(state, strict=True)
    return model.to(device, dtype=torch.float32).eval()


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=None, help="default results/<tag>/eval_expanded.json")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step (default: final update)")
    parser.add_argument("--replicates", type=str, default=None, help="comma-separated subset (default: all complete)")
    parser.add_argument("--max-dev-docs", type=int, default=None, help="debug only: truncate DEV docs")
    parser.add_argument("--max-questions", type=int, default=None, help="debug only: truncate world questions")
    parser.add_argument("--batch-rows", type=int, default=64)
    parser.add_argument("--story-chunk", type=int, default=31, help="docs per trainer_windows batch (+1 world row)")
    args = parser.parse_args()
    output_path = args.output or RESULTS_DIR / args.tag / "eval_expanded.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {rel(output_path)}; pass a new --output")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if args.device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    began = time.perf_counter()
    trainer = load_trainer_module()

    reports = list_replicate_reports(args.tag)
    if args.replicates:
        wanted = {int(value) for value in args.replicates.split(",")}
        reports = [(index, path) for index, path in reports if index in wanted]
    loaded = []
    for index, path in reports:
        report = load_report(path)
        if report.get("status") != "complete":
            print(f"skipping replicate {index}: status {report.get('status')!r}", file=sys.stderr)
            continue
        loaded.append((index, path, report))
    if not loaded:
        raise SystemExit(f"no complete replicate reports under runs/{args.tag}/")

    # Every replicate must share data, dev selection and evaluation-relevant config.
    reference = loaded[0][2]
    config0 = reference["config"]
    shared_keys = ("manifest", "length", "dev_seed", "dev_documents", "dev_world_batches", "hidden_size",
                   "native_size", "num_layers", "recurrent_kind", "ssm_backend", "recurrent_residual",
                   "projection_layout", "precision", "updates", "pilot_stage")
    for index, _, report in loaded:
        if any(report["config"].get(key) != config0.get(key) for key in shared_keys):
            raise ValueError(f"replicate {index} config differs in evaluation-relevant keys")
        if report["data_sha256"] != reference["data_sha256"]:
            raise ValueError(f"replicate {index} trained on different data bytes")
        if report["dev_selection"]["batch_sha256"] != reference["dev_selection"]["batch_sha256"]:
            raise ValueError(f"replicate {index} used different in-run DEV batches")
    length = config0["length"]; step = args.step if args.step is not None else config0["updates"]

    manifest = json.loads((ROOT / config0["manifest"]).read_text())
    dev_path = ROOT / manifest["splits"]["dev"]["path"]
    if sha256_file(dev_path) != reference["data_sha256"]["dev"]:
        raise ValueError("DEV split bytes differ from those recorded by the training runs")
    dev_all = trainer.stories(dev_path)
    if args.max_dev_docs:
        dev_all = dev_all[:args.max_dev_docs]
    dev_first = dev_all[:config0["dev_documents"]]

    # In-run DEV batches, rebuilt exactly as tools/train_language_pilot.py prepare_batches() does.
    replay_rng = np.random.default_rng(config0["dev_seed"])
    replay_story = [mixed_byte_batch(dev_first[i:i + 3], length, replay_rng, random_windows=False,
                                     terminal_windows=terminal, world_split="dev")
                    for terminal in (False, True) for i in range(0, len(dev_first), 3)]
    replay_world = [mixed_byte_batch(dev_first[:3], length, replay_rng, random_windows=False, world_split="dev")
                    for _ in range(config0["dev_world_batches"])]
    replay_digest = hashlib.sha256()
    for batch in replay_story + replay_world:
        trainer.batch_hash(replay_digest, batch)
    replay_match = replay_digest.hexdigest() == reference["dev_selection"]["batch_sha256"]
    if not replay_match and not args.max_dev_docs:
        raise AssertionError("rebuilt in-run DEV batches do not match the report digest")
    trainer_dev_ledgers = sorted({trainer.world_identity(batch) for batch in replay_story + replay_world})
    if sorted(map(tuple, reference["world_audit"]["dev_world_identities"])) != trainer_dev_ledgers:
        raise AssertionError("trainer DEV ledgers differ from the report's world audit")

    renderer_check = verify_renderer(length)
    dev_ledgers, eval_ledgers, questions = build_question_set(trainer_dev_ledgers)
    if args.max_questions:
        questions = questions[:args.max_questions]
    eval_set = set(eval_ledgers)
    question_list = [[list(values), query] for values, query in questions]
    leakage = []
    for index, _, report in loaded:
        train_ids = {tuple(values) for values in report["world_audit"]["train_world_identities"]}
        leakage.append({"replicate": index, "distinct_train_ledgers": len(train_ids),
                        "eval_ledgers_in_train": len(train_ids & eval_set),
                        "trainer_dev_ledgers_in_train": len(train_ids & set(trainer_dev_ledgers)),
                        "train_ledgers_outside_train_folds": sum(world_fold(list(v)) < 20 for v in train_ids)})
    if any(row["eval_ledgers_in_train"] or row["train_ledgers_outside_train_folds"] for row in leakage):
        raise AssertionError("held-out world leakage detected: " + json.dumps(leakage))
    world_meta = {"generator": "controlled_lm.language_pilot_data.mixed_byte_batch (rendered by render_world_row)",
                  "renderer_check": renderer_check, "possible_ledgers": 8 ** 4,
                  "dev_fold_ledgers": len(dev_ledgers), "trainer_dev_ledgers_excluded": len(trainer_dev_ledgers),
                  "eval_ledgers": len(eval_ledgers), "queries_per_ledger": 4, "questions": len(questions),
                  "truncated_for_debug": bool(args.max_questions),
                  "eval_ledger_folds": sorted({world_fold(list(v)) for v in eval_ledgers}),
                  "reserved_folds_0_9_opened": False,
                  "question_list_sha256": hashlib.sha256(json.dumps(question_list).encode()).hexdigest(),
                  "question_list": question_list, "leakage_checks": leakage,
                  "overlap_with_any_training_world_ledger": sum(row["eval_ledgers_in_train"] for row in leakage),
                  "chance_levels": {"typed_requested_key": "1/5 if uniform over 4 supported singletons + abstain",
                                    "query_byte": "1/8 if uniform over digits 0..7", "raw41": "1/41"}}

    trainer_window_story = trainer_window_batches(dev_all, length, config0["dev_seed"], args.story_chunk)
    tiles = [window for text in dev_all for window in tiled_windows(text, length)]
    tiled_story = [story_rows_batch(tiles[i:i + args.batch_rows], length) for i in range(0, len(tiles), args.batch_rows)]
    story_meta = {"dev_documents_available": manifest["splits"]["dev"]["documents"],
                  "dev_documents_evaluated": len(dev_all),
                  "dev_text_utf8_bytes": sum(len(text.encode("utf-8")) for text in dev_all),
                  "dev_split_sha256": reference["data_sha256"]["dev"], "truncated_for_debug": bool(args.max_dev_docs),
                  "in_run_dev_documents": len(dev_first),
                  "protocols": {"trainer_windows": "fixed prefix + fixed terminal(EOS) 128-token windows per doc",
                                "tiled_full": f"every byte once; BOS + <= {length - 1} bytes (+EOS at doc end)"}}

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    models = []
    for index, path, report in loaded:
        for run in report["runs"]:
            mode = run["mode"]
            point = next(point for point in run["curve"] if point["step"] == step)
            checkpoint = resolve_checkpoint(path, point, mode)
            if sha256_file(checkpoint) != point["checkpoint_sha256"]:
                raise AssertionError(f"checkpoint digest mismatch: replicate {index} {mode}")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["mode"] != mode or saved["step"] != step or saved["config"] != report["config"]:
                raise AssertionError(f"checkpoint metadata mismatch: replicate {index} {mode}")
            model = build_model(report["config"], mode, saved["model"], device)
            del saved
            tick = time.perf_counter()
            replay = trainer.evaluate(model, replay_story, replay_world, device, report["config"]["precision"], lambda: None)
            recorded = point.get("dev") or {}
            replay_diff = {key: abs(replay[key] - recorded[key]) for key in replay if key in recorded}
            entry = {"replicate": index, "seed": report["config"]["seed"], "data_seed": report["config"]["data_seed"],
                     "mode": mode, "step": step, "checkpoint_sha256": point["checkpoint_sha256"],
                     "in_run_dev_replay": {"metrics": replay, "batch_digest_matches_report": replay_match,
                                           "max_abs_diff_vs_report": max(replay_diff.values()) if replay_diff else None},
                     "world": score_worlds(model, questions, length, device, args.batch_rows),
                     "story": {"trainer_windows": score_stories(model, trainer_window_story, device),
                               "tiled_full": score_stories(model, tiled_story, device)}}
            entry["seconds"] = round(time.perf_counter() - tick, 3)
            models.append(entry)
            print(f"replicate {index} {mode:11s} typed={entry['world']['typed_requested_key_accuracy']:.4f} "
                  f"query={entry['world']['query_byte_accuracy']:.4f} "
                  f"bpb_tiled={entry['story']['tiled_full']['bits_per_utf8_byte']:.5f} "
                  f"replay_diff={entry['in_run_dev_replay']['max_abs_diff_vs_report']}", flush=True)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    story_meta["trainer_windows_counts"] = {key: models[0]["story"]["trainer_windows"][key]
                                           for key in ("target_tokens", "utf8_byte_targets", "eos_targets", "windows")}
    story_meta["tiled_full_counts"] = {key: models[0]["story"]["tiled_full"][key]
                                      for key in ("target_tokens", "utf8_byte_targets", "eos_targets", "windows")}
    payload = {"artifact_kind": "ROUTING_REPLICATION_EXPANDED_DEV_EVAL", "tag": args.tag, "step": step,
               "pilot_stage": config0["pilot_stage"], "development_only": True, "official_test_evaluated": False,
               "device": device, "device_name": torch.cuda.get_device_name() if device == "cuda" else "cpu",
               "torch_version": torch.__version__, "eval_source_sha256": sha256_file(Path(__file__)),
               "replicates": [index for index, _, _ in loaded], "modes": list(MODES),
               "world_probe": world_meta, "stories": story_meta, "models": models,
               "wall_seconds": round(time.perf_counter() - began, 3)}
    write_json_new(output_path, payload)
    print(json.dumps({"output": rel(output_path), "models": len(models), "questions": len(questions),
                      "tiled_byte_targets": story_meta["tiled_full_counts"]["utf8_byte_targets"],
                      "wall_seconds": payload["wall_seconds"]}))


if __name__ == "__main__":
    main()
