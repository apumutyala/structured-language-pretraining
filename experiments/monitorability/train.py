"""Train one arm/seed. Usage:
    python train.py --arm plain|structured --seed 0 --config configs/cpu_pilot.json [--steps N] [--device cpu|cuda]
Writes <out_dir>/<arm>_s<seed>/model.pt and train_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F

import env
from model import TinyLM


def load_config(path):
    with open(path) as f:
        return json.load(f)


def collate(eps, device):
    T = max(len(e.tokens) for e in eps)
    ids = torch.full((len(eps), T), env.PAD, dtype=torch.long)
    for i, e in enumerate(eps):
        ids[i, :len(e.tokens)] = torch.tensor(e.tokens)
    lengths = torch.tensor([len(e.tokens) for e in eps])
    target = torch.tensor([env.ANSWER_IDS[e.target_label] for e in eps])
    slots = torch.tensor([e.slot_labels for e in eps])
    return ids.to(device), lengths.to(device), target.to(device), slots.to(device)


def collate_train(eps, device):
    """Full teacher-forced sequences (primary + follow-up answers). Returns ids, lengths,
    primary positions, all supervised (batch_idx, pos, target) and slot labels."""
    seqs = [env.training_sequence(e) for e in eps]
    T = max(len(t) for t, _ in seqs)
    ids = torch.full((len(eps), T), env.PAD, dtype=torch.long)
    bi, pos, tgt = [], [], []
    for i, (t, sup) in enumerate(seqs):
        ids[i, :len(t)] = torch.tensor(t)
        for p_, a in sup:
            bi.append(i); pos.append(p_); tgt.append(a)
    prim = torch.tensor([len(e.tokens) - 1 for e in eps])
    lengths = torch.tensor([len(t) for t, _ in seqs])
    slots = torch.tensor([e.slot_labels for e in eps])
    return (ids.to(device), lengths.to(device), prim.to(device), torch.tensor(bi).to(device),
            torch.tensor(pos).to(device), torch.tensor(tgt).to(device), slots.to(device))


def build_model(cfg, arm, seed):
    torch.manual_seed(seed)
    m = cfg["model"]
    return TinyLM(len(env.VOCAB), d=m["d"], layers=m["layers"], heads=m["heads"],
                  mlp_mult=m.get("mlp_mult", 4), max_len=m.get("max_len", 160),
                  dropout=m.get("dropout", 0.0), n_slots=env.K_MAX + 1,
                  aux=(arm == "structured"), aux_seed=seed, n_tied=m.get("n_tied_heads", 0),
                  identity_recall=m.get("identity_recall", False))


@torch.no_grad()
def quick_val(model, val, device, bs=256):
    model.eval()
    out = {}
    correct = {c: [0, 0] for c in env.CONTEXTS}
    by_depth = {d: [0, 0] for d in range(env.MAX_DEPTH + 1)}
    ans = torch.tensor(env.ANSWER_IDS, device=device)
    for i in range(0, len(val), bs):
        chunk = val[i:i + bs]
        ids, lengths, _, _ = collate(chunk, device)
        logits, _ = model(ids, lengths)
        pred = logits[:, ans].argmax(-1).cpu().tolist()
        for e, p in zip(chunk, pred):
            correct[e.context][0] += int(p == e.target_label)
            correct[e.context][1] += 1
            if e.context == "H":
                by_depth[e.depth][0] += int(p == e.target_label)
                by_depth[e.depth][1] += 1
    model.train()
    for c, (a, n) in correct.items():
        out[f"val_target_acc_{c}"] = a / max(n, 1)
    for dd, (a, n) in by_depth.items():
        out[f"val_H_acc_depth{dd}"] = a / max(n, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["plain", "structured"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "cpu_pilot.json"))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr = cfg["train"]
    steps = args.steps or tr["steps"]
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    out_dir = args.out_dir or os.path.join(HERE, cfg.get("ckpt_dir", "checkpoints"), cfg["name"])
    run_dir = os.path.join(out_dir, f"{args.arm}_s{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    model = build_model(cfg, args.arm, args.seed).to(device)
    n_trunk = model.trunk_params()
    n_total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], betas=(0.9, 0.98),
                            weight_decay=tr["weight_decay"])
    warm = tr["warmup"]

    def lr_at(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, steps - warm)
        return tr.get("min_lr_frac", 0.1) + (1 - tr.get("min_lr_frac", 0.1)) * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    # identical data stream for both arms (depends on seed only)
    data_rng = random.Random(1_000_003 * (args.seed + 1))
    val_rng = random.Random(424242)
    val = [env.make_episode(val_rng, split="train") for _ in range(tr.get("val_n", 600))]
    mix = tuple(cfg["data"]["mix"])
    nxq = cfg["data"].get("n_extra_q", 0)  # honest follow-up questions after the primary answer
    aux_w = tr["aux_weight"] if args.arm == "structured" else 0.0
    # optional depth curriculum (identical for both arms): early steps sample depth <= cur_max
    cur = cfg["data"].get("curriculum")  # e.g. {"until_frac": 0.2, "max_depth": 1}
    log = []
    t0 = time.time()
    run_ans = run_aux = 0.0
    for step in range(steps):
        if cur and step < cur["until_frac"] * steps:
            eps = [env.make_episode(data_rng, split="train", mix=mix,
                                    depth=data_rng.randrange(cur["max_depth"] + 1), n_extra_q=nxq)
                   for _ in range(tr["batch"])]
        else:
            eps = [env.make_episode(data_rng, split="train", mix=mix, n_extra_q=nxq) for _ in range(tr["batch"])]
        ids, lengths, prim, sb, sp, st, slots = collate_train(eps, device)
        logits, hs = model.forward_train(ids, lengths, prim, sb, sp)
        loss_ans = F.cross_entropy(logits, st)  # mean over all supervised answers (primary + follow-ups)
        loss = loss_ans
        if aux_w > 0:
            loss_aux = model.aux_loss(hs, slots)
            loss = loss + aux_w * loss_aux
            run_aux = 0.98 * run_aux + 0.02 * loss_aux.item() if step else loss_aux.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
        opt.step(); sched.step()
        run_ans = 0.98 * run_ans + 0.02 * loss_ans.item() if step else loss_ans.item()
        if (step + 1) % tr["eval_every"] == 0 or step + 1 == steps:
            rec = {"step": step + 1, "time_s": round(time.time() - t0, 1),
                   "loss_answer_ema": round(run_ans, 4), "loss_aux_ema": round(run_aux, 4),
                   "lr": sched.get_last_lr()[0]}
            rec.update({k: round(v, 4) for k, v in quick_val(model, val, device).items()})
            log.append(rec)
            print(json.dumps(rec), flush=True)
    wall = time.time() - t0
    torch.save({"state_dict": model.state_dict(), "config": cfg, "arm": args.arm, "seed": args.seed,
                "steps": steps}, os.path.join(run_dir, "model.pt"))
    metrics = {"arm": args.arm, "seed": args.seed, "config_name": cfg["name"], "config": cfg,
               "steps": steps, "device": args.device, "threads": args.threads,
               "params_trunk": n_trunk, "params_total": n_total, "train_wall_s": round(wall, 1),
               "log": log, "torch": torch.__version__}
    with open(os.path.join(run_dir, "train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"done {args.arm} s{args.seed} wall={wall:.0f}s params_trunk={n_trunk}")


if __name__ == "__main__":
    main()
