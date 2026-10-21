"""Train-only local CUDA fit/timing of fresh byte-LM cores; no capability claim."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import copy
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from controlled_lm.language_pilot_data import mixed_byte_batch
from controlled_lm.typed_language_model import TypedLanguageModel
from controlled_lm.structured_decision import canonical_state_dict_digest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/language_gpu_qualification.json"


def objective(output, batch):
    losses = F.cross_entropy(output.raw_logits.flatten(0, 1), batch.targets.flatten(),
                             ignore_index=0, reduction="none").reshape_as(batch.targets)
    mask = batch.targets != 0
    story, world = mask & ~batch.world_rows[:, None], mask & batch.world_rows[:, None]
    return (.75 * losses[story].mean() + .25 * losses[world].mean()
            + .2 * F.cross_entropy(output.proposal_logits[batch.world_rows], batch.claim_targets[batch.world_rows]))


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-layout", choices=("per_layer", "shared"), default="per_layer")
    parser.add_argument("--story-rows", type=int, default=3)
    parser.add_argument("--gate-bias-max", type=float)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output
    if not 1 <= args.story_rows <= 63:
        raise ValueError("story rows must be1..63")
    if OUTPUT.exists():
        raise FileExistsError("preserve qualification; choose new version explicitly")
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != "NVIDIA GeForce RTX 5070 Ti Laptop GPU":
        raise RuntimeError("only authorized local laptop GPU")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.65)
    manifest = json.loads((ROOT / "results/language_pilot_extraction.json").read_text())
    source = ROOT / manifest["splits"]["train"]["path"]
    stories = []
    with source.open(encoding="utf-8") as stream:
        for _ in range(args.story_rows):
            stories.append(json.loads(next(stream))["story"])
    batch_cpu = mixed_byte_batch(stories, 128, np.random.default_rng(791001), random_windows=False)
    batch = batch_cpu.to("cuda")
    report = {"status": "running", "device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "parameters_from_random_initialization": True, "paid_compute": False,
              "training_data_only": True, "dev_or_test_evaluated": False,
              "meaning": "fit/timing/parity, not language quality or broad alignment",
              "configuration": {"hidden_size":192, "native_size":384, "num_layers":4,
                                "batch":args.story_rows+1, "length":128, "projection_layout":args.projection_layout,
                                "gate_bias_range":None if args.gate_bias_max is None else [0,args.gate_bias_max]},
              "data_manifest_sha256": hashlib.sha256((ROOT / "results/language_pilot_extraction.json").read_bytes()).hexdigest(),
              "runs": []}
    source_paths = [Path(__file__), *[ROOT / "src/controlled_lm" / name for name in
        ("typed_language_model.py", "language_pilot_data.py", "typed_evidence_lane.py", "parallel_semantic.py",
         "semantic_transitions.py", "invariant_ssm.py", "parallel_ssm.py", "evidence_decoder.py", "policy.py")]]
    source_hashes = lambda: {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    report["source_sha256_before"] = source_hashes()
    OUTPUT.write_text(json.dumps(report,indent=2)+"\n")
    began = time.perf_counter()
    for kind, backend in (("gru", "sequential"), ("ssm", "sequential"), ("ssm", "parallel")):
        for precision in ("fp32", "bf16_autocast"):
            if time.perf_counter()-began > 180:
                raise TimeoutError("local qualification aggregate180s budget")
            torch.manual_seed(791003)
            cpu = TypedLanguageModel(192,384,num_layers=4,recurrent_kind=kind,ssm_backend=backend,
                                     projection_layout=args.projection_layout)
            if kind == "ssm" and args.gate_bias_max is not None:
                with torch.no_grad():
                    for layer in cpu.recurrent_layers:
                        layer.input_projection.bias[:192].copy_(torch.linspace(0,args.gate_bias_max,192))
            model = copy.deepcopy(cpu).cuda().train()
            optimizer = torch.optim.Adam(model.parameters(),lr=.0003)
            torch.cuda.reset_peak_memory_stats()
            initial = canonical_state_dict_digest(model)
            parity = None
            if precision == "fp32":
                with torch.no_grad():
                    expected = cpu(batch_cpu.input_ids,batch_cpu.attention_mask,observed_event_ids=batch_cpu.observed_event_ids,
                                   trusted_event_mask=batch_cpu.trusted_event_mask,policies=batch_cpu.policies)
                    actual = model(batch.input_ids,batch.attention_mask,observed_event_ids=batch.observed_event_ids,
                                   trusted_event_mask=batch.trusted_event_mask,policies=batch.policies)
                parity = float((expected.raw_logits-actual.raw_logits.cpu()).abs().max())
                if parity > 1e-4:
                    raise AssertionError(f"fresh CPU/CUDA logit parity {parity}")
            elapsed = []
            for step in range(8):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=precision=="bf16_autocast"):
                    output = model(batch.input_ids,batch.attention_mask,observed_event_ids=batch.observed_event_ids,
                                   trusted_event_mask=batch.trusted_event_mask,policies=batch.policies)
                    loss = objective(output,batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss; no clipping")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("nonfinite gradient")
                optimizer.step()
                torch.cuda.synchronize()
                elapsed.append(time.perf_counter()-tick)
            forbidden = output.grounded.probabilities.masked_select(~output.grounded.allowed)
            row = {"core":kind,"backend":backend,"precision":precision,"parameters":sum(p.numel() for p in model.parameters()),
                   "initial_parameters_sha256":initial,"cpu_cuda_initial_max_abs_error":parity,
                   "optimizer_updates":8,"warmup_updates":3,"measured_updates":5,
                   "median_step_seconds":float(np.median(elapsed[3:])),
                   "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"last_loss":float(loss.detach()),
                   "max_forbidden_probability":float(forbidden.detach().max()),
                   "story_workspace_empty":bool(not output.final_evidence[~batch.world_rows].any()),
                   "world_workspace_facts":int(output.final_evidence[batch.world_rows].sum()),
                   "core_abs_max":max(float(x.detach().abs().max()) for x in output.core_states)}
            report["runs"].append(row)
            OUTPUT.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(row),flush=True)
            del output, loss, model, cpu, optimizer
            torch.cuda.empty_cache()
    report.update(status="complete",wall_seconds=time.perf_counter()-began,
                  source_unchanged=source_hashes()==report["source_sha256_before"])
    OUTPUT.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
