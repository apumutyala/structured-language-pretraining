"""Aggregate per-seed eval JSONs into across-seed summaries (mean, t-based 95% CI, paired diffs).

Usage: python aggregate.py --config_name cpu_pilot   -> results/<name>/summary.json and summary.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))

PRIMARY_POP = "within_S_verified_heldout"
POPS = ["within_S_verified_heldout", "within_S_heldout", "within_S_verified_train",
        "F_verified_lies_vs_H_correct_heldout", "F_lies_vs_H_correct_heldout",
        "S_verified_lies_vs_neutral_false_claim_heldout", "all_contexts_heldout"]
DETS = ["disagree_selected", "disagree_final", "disagree_layer0", "disagree_bow",
        "lie_probe_Strained_selected", "lie_probe_Ftrained_selected",
        "output_low_confidence", "output_entropy", "persona_trigger", "claim_token_equals_output", "claim_present"]


def get(d, path):
    for k in path:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def metric_paths(r):
    sel = str(r["probe_layer_selected"])
    nl = len([k for k in r["q1_truth_probe"] if k != "bow"])
    fin = str(nl - 1)
    m = {}
    b = ("behavior",)
    for s in ["H_train", "H_heldout", "S_train", "S_heldout", "F_train", "F_heldout"]:
        m[f"beh.{s}.target_acc"] = b + (s, "target_acc", "p")
        m[f"beh.{s}.truth_acc"] = b + (s, "truth_acc", "p")
    for dep in range(4):
        m[f"beh.H_heldout.target_acc_depth{dep}"] = b + ("H_heldout", "target_acc_by_depth", str(dep), "p")
        m[f"beh.H_train.target_acc_depth{dep}"] = b + ("H_train", "target_acc_by_depth", str(dep), "p")
    for lab in ["yes", "no", "unknown"]:
        m[f"beh.H_heldout.target_acc_true_{lab}"] = b + ("H_heldout", "target_acc_by_true_label", lab, "p")
    for s in ["S_train", "S_heldout"]:
        m[f"beh.{s}.compliance_claim_false"] = b + (s, "S_compliance_when_claim_false", "p")
        m[f"beh.{s}.acc_claim_true"] = b + (s, "S_acc_when_claim_true", "p")
    for s in ["S_heldout", "F_heldout", "S_train", "F_train"]:
        m[f"beh.{s}.lies_verified_frac"] = b + (s, "lies_verified_frac", "p")
        m[f"beh.{s}.n_verified_lies"] = None  # filled below
        m[f"beh.{s}.matched_neutral_truth_acc"] = b + (s, "matched_neutral_truth_acc", "p")
        m[f"beh.{s}.false_output_rate"] = b + (s, "false_output_rate", "p")
    m["probe_layer_selected"] = ("probe_layer_selected",)
    for tag, L in (("sel", sel), ("final", fin), ("L0", "0"), ("bow", "bow")):
        q = ("q1_truth_probe", L)
        for s in ["probe_val", "H_train", "H_heldout", "S_heldout", "F_heldout"]:
            m[f"q1.{tag}.acc_{s}"] = q + (f"acc_{s}",)
        for s in ["S_heldout", "F_heldout", "S_train", "F_train"]:
            m[f"q2.{tag}.truth_acc_on_lies_{s}"] = q + (f"acc_on_lies_{s}", "p")
            m[f"q2.{tag}.agrees_output_on_lies_{s}"] = q + (f"agrees_with_output_on_lies_{s}", "p")
        m[f"q1.{tag}.acc_when_model_wrong_H_heldout"] = q + ("acc_when_model_wrong_H_heldout", "p")
    for L in range(nl):
        m[f"q1.layer{L}.acc_H_heldout"] = ("q1_truth_probe", str(L), "acc_H_heldout")
        m[f"q2.layer{L}.truth_acc_on_lies_S_heldout"] = ("q1_truth_probe", str(L), "acc_on_lies_S_heldout", "p")
        m[f"q3.layer{L}.disagree.{PRIMARY_POP}"] = ("q3_detectors_auroc", PRIMARY_POP, f"disagree_L{L}", "auroc")
    for pop in POPS:
        for det in DETS:
            base = ("q3_detectors_auroc", pop, det)
            m[f"q3.{pop}.{det}.auroc"] = base + ("auroc",)
            m[f"q3.{pop}.{det}.recall@1%fpr"] = base + ("recall_at_1pct_fpr",)
            m[f"q3.{pop}.{det}.recall@5%fpr"] = base + ("recall_at_5pct_fpr",)
        m[f"q3.{pop}.n_pos"] = ("q3_detectors_auroc", pop, "disagree_selected", "n_pos")
        m[f"q3.{pop}.n_neg"] = ("q3_detectors_auroc", pop, "disagree_selected", "n_neg")
    for dev, tests in (("within_S_train", ["within_S_heldout", "within_S_verified_heldout"]),
                       ("F_lies_vs_H_correct_train", ["F_lies_vs_H_correct_heldout", "F_verified_lies_vs_H_correct_heldout"])):
        m[f"null.{dev}.best_dev_auroc"] = ("q3_random_direction_null", dev, "best_dev_auroc")
        for t in tests:
            m[f"null.{dev}.test.{t}.auroc"] = ("q3_random_direction_null", dev, t, "auroc")
    fl = ("q3_disagreement_flags_selected_layer",)
    for s in ["H_heldout", "S_heldout", "F_heldout"]:
        m[f"flags.{s}.on_lies"] = fl + (s, "flag_rate_on_lies", "p")
        m[f"flags.{s}.on_truthful"] = fl + (s, "flag_rate_on_truthful", "p")
    m["flags.H_heldout.neutral_false_claim_truthful"] = fl + ("H_heldout", "flag_rate_neutral_false_claim_truthful", "p")
    return m


def ci(vals):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    if len(v) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    if len(v) == 1:
        return {"mean": float(v[0]), "lo": None, "hi": None, "n": 1}
    se = v.std(ddof=1) / np.sqrt(len(v))
    h = stats.t.ppf(0.975, len(v) - 1) * se
    return {"mean": float(v.mean()), "lo": float(v.mean() - h), "hi": float(v.mean() + h), "n": int(len(v)),
            "sd": float(v.std(ddof=1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", required=True)
    ap.add_argument("--results_dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    rdir = os.path.join(args.results_dir, args.config_name)
    runs = {}
    for f in sorted(glob.glob(os.path.join(rdir, "*_s*.json"))):
        if f.endswith("_train_metrics.json"):
            continue
        r = json.load(open(f))
        runs[(r["arm"], r["seed"])] = r
    arms = sorted({a for a, _ in runs})
    seeds = {a: sorted(s for b, s in runs if b == a) for a in arms}
    any_run = next(iter(runs.values()))
    paths = metric_paths(any_run)
    per = {}
    for (a, s), r in runs.items():
        mp = metric_paths(r)
        vals = {k: (get(r, p) if p else None) for k, p in mp.items()}
        for split in ["S_heldout", "F_heldout", "S_train", "F_train"]:
            lv = get(r, ("behavior", split, "lies_verified_frac"))
            vals[f"beh.{split}.n_verified_lies"] = (round(lv["p"] * lv["n"]) if lv and lv["p"] is not None else None)
        per[(a, s)] = vals
    summary = {"config_name": args.config_name, "arms": arms, "seeds": seeds, "metrics": {}}
    for k in paths:
        entry = {}
        for a in arms:
            vs = [per[(a, s)].get(k) for s in seeds[a]]
            entry[a] = {**ci(vs), "per_seed": dict(zip([str(s) for s in seeds[a]], vs))}
        if set(arms) == {"plain", "structured"}:
            common = sorted(set(seeds["plain"]) & set(seeds["structured"]))
            diffs = []
            for s in common:
                x, y = per[("structured", s)].get(k), per[("plain", s)].get(k)
                diffs.append(None if x is None or y is None else x - y)
            entry["paired_diff_structured_minus_plain"] = {**ci(diffs),
                                                           "per_seed": dict(zip([str(s) for s in common], diffs))}
            valid = [d for d in diffs if d is not None]
            entry["paired_diff_structured_minus_plain"]["n_positive"] = int(sum(d > 0 for d in valid))
        summary["metrics"][k] = entry
    summary["wall_clock"] = {f"{a}_s{s}": {"train_s": None, "eval_s": runs[(a, s)].get("eval_wall_s")} for a, s in runs}
    for a, s in runs:
        tm = os.path.join(rdir, f"{a}_s{s}_train_metrics.json")
        if os.path.exists(tm):
            t = json.load(open(tm))
            summary["wall_clock"][f"{a}_s{s}"]["train_s"] = t["train_wall_s"]
            summary["params_trunk"] = t["params_trunk"]
            summary.setdefault("params_total", {})[a] = t["params_total"]
    with open(os.path.join(rdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    # markdown
    def fmt(e):
        if e is None or e.get("mean") is None:
            return "n/a"
        if e.get("lo") is None:
            return f"{e['mean']:.3f}"
        return f"{e['mean']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}]"
    lines = [f"# Summary: {args.config_name}", "",
             f"Seeds: {seeds}. Cells: mean across seeds [t-based 95% CI]. Diff = structured - plain, paired by seed (n_pos = seeds with diff>0).", "",
             "| metric | " + " | ".join(arms) + (" | diff (S-P) | n_pos |" if len(arms) == 2 else " |"),
             "|---|" + "---|" * len(arms) + ("---|---|" if len(arms) == 2 else "")]
    for k in paths:
        e = summary["metrics"][k]
        row = f"| {k} | " + " | ".join(fmt(e[a]) for a in arms)
        if "paired_diff_structured_minus_plain" in e:
            d = e["paired_diff_structured_minus_plain"]
            row += f" | {fmt(d)} | {d.get('n_positive')}/{d.get('n')}"
        lines.append(row + " |")
    with open(os.path.join(rdir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {rdir}/summary.json, summary.md ({len(runs)} runs)")


if __name__ == "__main__":
    main()
