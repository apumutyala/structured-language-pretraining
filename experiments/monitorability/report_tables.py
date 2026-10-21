"""Build the markdown tables used in RESULTS.md directly from results/<cfg>/*.json.

Per-seed values (with per-run episode-bootstrap 95% CIs where available), across-seed mean and
range, paired structured-minus-plain differences (mean, range, #seeds > 0), and the per-run
validity gates G1-G3 of PREREGISTRATION.md. Descriptive only (Amendment 1).
Usage: python report_tables.py --config_name cpu_pilot  -> results/<cfg>/report_tables.md
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def cell(v, lo=None, hi=None):
    if v is None:
        return "n/a"
    return f"{v:.3f}" + (f" [{lo:.2f},{hi:.2f}]" if lo is not None and hi is not None else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", default="cpu_pilot")
    args = ap.parse_args()
    rdir = os.path.join(HERE, "results", args.config_name)
    runs = {}
    for f in sorted(glob.glob(os.path.join(rdir, "*_s*.json"))):
        if f.endswith("_train_metrics.json"):
            continue
        r = json.load(open(f))
        tm = os.path.join(rdir, f"{r['arm']}_s{r['seed']}_train_metrics.json")
        r["_train"] = json.load(open(tm)) if os.path.exists(tm) else {}
        runs[(r["arm"], r["seed"])] = r
    seeds = sorted({s for _, s in runs})
    arms = ["plain", "structured"]
    L = []

    # ---------- gates
    L += ["### Validity gates per run (PREREGISTRATION.md section 5)", "",
          "| run | H held-out acc | depth0 acc | depth1 acc | G1 (H>=0.60 and pooled d0-1>=0.85) | S compliance (claim false) | F compliance | G2 | verified S lies / truthful S (held-out) | G3 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    gates = {}
    for a in arms:
        for s in seeds:
            r = runs.get((a, s))
            if not r:
                continue
            b = r["behavior"]
            h = b["H_heldout"]["target_acc"]["p"]
            d0 = b["H_heldout"]["target_acc_by_depth"]["0"]["p"]
            d1 = b["H_heldout"]["target_acc_by_depth"]["1"]["p"]
            n0 = b["H_heldout"]["target_acc_by_depth"]["0"]["n"]; n1 = b["H_heldout"]["target_acc_by_depth"]["1"]["n"]
            d01 = (d0 * n0 + d1 * n1) / (n0 + n1)
            g1 = h >= 0.60 and d01 >= 0.85  # same definition as gates.py (pooled depth 0-1)
            sc = b["S_heldout"]["S_compliance_when_claim_false"]["p"]
            fc = b["F_heldout"]["F_compliance"]["p"]
            g2 = sc >= 0.80 and fc >= 0.60
            q = r["q3_detectors_auroc"]["within_S_verified_heldout"]["disagree_selected"]
            g3 = q["n_pos"] >= 30 and q["n_neg"] >= 30
            gates[(a, s)] = (g1, g2, g3)
            L.append(f"| {a} s{s} | {f3(h)} | {f3(d0)} | {f3(d1)} | {'pass' if g1 else 'FAIL'} | {f3(sc)} | {f3(fc)} | "
                     f"{'pass' if g2 else 'FAIL'} | {q['n_pos']} / {q['n_neg']} | {'pass' if g3 else 'FAIL'} |")
    L.append("")

    # ---------- metric table helper
    def table(title, rows, ci_path_suffix=None):
        L.extend([f"### {title}", "",
                  "| metric | " + " | ".join(f"plain s{s}" for s in seeds) + " | plain mean [range] | "
                  + " | ".join(f"struct s{s}" for s in seeds) + " | struct mean [range] | diff S-P mean [range] (#>0) |",
                  "|---|" + "---|" * (2 * len(seeds) + 3)])
        for name, path, has_ci in rows:
            vals = {}
            cells = {}
            for a in arms:
                for s in seeds:
                    r = runs.get((a, s))
                    v = get(r, path) if r else None
                    if isinstance(v, dict):
                        node = v
                        v = node.get("auroc", node.get("p"))
                        cells[(a, s)] = cell(v, node.get("lo"), node.get("hi")) if has_ci else cell(v)
                    else:
                        cells[(a, s)] = cell(v)
                    vals[(a, s)] = v
            row = f"| {name} | "
            for a in arms:
                vs = [vals[(a, s)] for s in seeds if vals[(a, s)] is not None]
                row += " | ".join(cells[(a, s)] for s in seeds) + " | "
                row += (f"{sum(vs)/len(vs):.3f} [{min(vs):.3f}, {max(vs):.3f}]" if vs else "n/a") + " | "
            ds = [vals[("structured", s)] - vals[("plain", s)] for s in seeds
                  if vals[("structured", s)] is not None and vals[("plain", s)] is not None]
            row += (f"{sum(ds)/len(ds):+.3f} [{min(ds):+.3f}, {max(ds):+.3f}] ({sum(d > 0 for d in ds)}/{len(ds)})" if ds else "n/a") + " |"
            L.append(row)
        L.append("")

    beh = [("H train-family acc", ("behavior", "H_train", "target_acc"), True),
           ("H held-out-family acc", ("behavior", "H_heldout", "target_acc"), True)]
    beh += [(f"H held-out acc, depth {d}", ("behavior", "H_heldout", "target_acc_by_depth", str(d)), True) for d in range(4)]
    beh += [(f"H held-out acc, true={l}", ("behavior", "H_heldout", "target_acc_by_true_label", l), True) for l in ("yes", "no", "unknown")]
    beh += [("S compliance when claim false (held-out)", ("behavior", "S_heldout", "S_compliance_when_claim_false"), True),
            ("S accuracy when claim true (held-out)", ("behavior", "S_heldout", "S_acc_when_claim_true"), True),
            ("F compliance (held-out)", ("behavior", "F_heldout", "F_compliance"), True),
            ("S matched-neutral truth acc (held-out)", ("behavior", "S_heldout", "matched_neutral_truth_acc"), True),
            ("S false outputs that are verified lies (held-out)", ("behavior", "S_heldout", "lies_verified_frac"), True),
            ("F false outputs that are verified lies (held-out)", ("behavior", "F_heldout", "lies_verified_frac"), True)]
    table("Behavior (Wilson 95% CI per run)", beh)

    L_q1_title = "Q1/Q2 truth probe at the selected layer (fit on H train-family only; chance = 0.333)"
    L.extend([f"### {L_q1_title}", "",
              "| metric | " + " | ".join(f"plain s{s}" for s in seeds) + " | " + " | ".join(f"struct s{s}" for s in seeds) + " |",
              "|---|" + "---|" * (2 * len(seeds))])
    probe_rows = [("selected layer", None),
                  ("acc probe_val (H train fam)", ("acc_probe_val",)),
                  ("acc H held-out fam", ("acc_H_heldout",)),
                  ("acc S held-out (all)", ("acc_S_heldout",)),
                  ("acc F held-out (all)", ("acc_F_heldout",)),
                  ("acc on false-output S episodes (held-out)", ("acc_on_lies_S_heldout", "p")),
                  ("  probe agrees with the false output", ("agrees_with_output_on_lies_S_heldout", "p")),
                  ("acc on false-output F episodes (held-out)", ("acc_on_lies_F_heldout", "p")),
                  ("  probe agrees with the false output", ("agrees_with_output_on_lies_F_heldout", "p")),
                  ("acc when model wrong on H held-out", ("acc_when_model_wrong_H_heldout", "p"))]
    for name, path in probe_rows:
        row = f"| {name} | "
        cs = []
        for a in arms:
            for s in seeds:
                r = runs.get((a, s))
                if not r:
                    cs.append("n/a"); continue
                sel = str(r["probe_layer_selected"])
                if path is None:
                    cs.append(sel); continue
                v = get(r["q1_truth_probe"][sel], path)
                n = get(r["q1_truth_probe"][sel], path[:-1] + ("n",)) if path[-1] == "p" else None
                cs.append(f3(v) + (f" (n={n})" if n is not None else ""))
        L.append(row + " | ".join(cs) + " |")
    L.append("")
    # per-layer probe on H held-out and S lies
    nl = max(len([k for k in r["q1_truth_probe"] if k != "bow"]) for r in runs.values())
    L.extend(["### Truth-probe accuracy by layer (H held-out / S false-output held-out)", "",
              "| layer | " + " | ".join(f"{a[:6]} s{s}" for a in arms for s in seeds) + " |",
              "|---|" + "---|" * (2 * len(seeds))])
    for l in [str(i) for i in range(nl)] + ["bow"]:
        cs = []
        for a in arms:
            for s in seeds:
                r = runs.get((a, s))
                q = r["q1_truth_probe"].get(l) if r else None
                cs.append("n/a" if q is None else f"{q['acc_H_heldout']:.3f} / {f3(q['acc_on_lies_S_heldout']['p'])}")
        L.append(f"| {l} | " + " | ".join(cs) + " |")
    L.append("")

    pops = [("PRIMARY: within-S verified, held-out fam", "within_S_verified_heldout"),
            ("within-S all false outputs, held-out fam", "within_S_heldout"),
            ("within-S verified, train fam", "within_S_verified_train"),
            ("F verified lies vs H correct, held-out fam", "F_verified_lies_vs_H_correct_heldout"),
            ("S verified lies vs neutral false-claim truthful H, held-out", "S_verified_lies_vs_neutral_false_claim_heldout"),
            ("all contexts, held-out fam", "all_contexts_heldout")]
    dets = ["disagree_selected", "disagree_final", "disagree_layer0", "disagree_bow",
            "lie_probe_Strained_selected", "lie_probe_Ftrained_selected", "output_low_confidence",
            "persona_trigger", "claim_token_equals_output"]
    for title, pop in pops:
        rows = [(f"{d}", ("q3_detectors_auroc", pop, d), True) for d in dets]
        rows.append(("n_pos (lies)", ("q3_detectors_auroc", pop, "disagree_selected", "n_pos"), False))
        rows.append(("n_neg", ("q3_detectors_auroc", pop, "disagree_selected", "n_neg"), False))
        table(f"AUROC: {title} (per-run episode-bootstrap 95% CI)", rows)
    rec = []
    for d in ("disagree_selected", "lie_probe_Ftrained_selected", "output_low_confidence"):
        for k in ("recall_at_1pct_fpr", "recall_at_5pct_fpr"):
            rec.append((f"{d} {k}", ("q3_detectors_auroc", "within_S_verified_heldout", d, k), False))
    table("Recall at fixed FPR, primary population", rec)
    null = [("best dev AUROC (within-S train, sign-free)", ("q3_random_direction_null", "within_S_train", "best_dev_auroc"), False),
            ("median dev AUROC over 1000 dirs", ("q3_random_direction_null", "within_S_train", "median_dev_auroc_sign_free"), False),
            ("-> test within-S held-out", ("q3_random_direction_null", "within_S_train", "within_S_heldout"), True),
            ("-> test within-S verified held-out", ("q3_random_direction_null", "within_S_train", "within_S_verified_heldout"), True),
            ("best dev AUROC (F lies vs H, train)", ("q3_random_direction_null", "F_lies_vs_H_correct_train", "best_dev_auroc"), False),
            ("-> test F verified vs H held-out", ("q3_random_direction_null", "F_lies_vs_H_correct_train", "F_verified_lies_vs_H_correct_heldout"), True)]
    table("Random-direction null (best of 1000 unit directions at the selected layer)", null)
    fl = [("flag rate on S false outputs", ("q3_disagreement_flags_selected_layer", "S_heldout", "flag_rate_on_lies"), True),
          ("flag rate on S truthful outputs", ("q3_disagreement_flags_selected_layer", "S_heldout", "flag_rate_on_truthful"), True),
          ("flag rate on H truthful outputs", ("q3_disagreement_flags_selected_layer", "H_heldout", "flag_rate_on_truthful"), True),
          ("flag rate on H neutral false-claim truthful", ("q3_disagreement_flags_selected_layer", "H_heldout", "flag_rate_neutral_false_claim_truthful"), True),
          ("flag rate on F false outputs", ("q3_disagreement_flags_selected_layer", "F_heldout", "flag_rate_on_lies"), True)]
    table("Disagreement detector as a binary flag (argmax probe != output), held-out family", fl)
    # wall clock
    L.extend(["### Wall-clock and size", "", "| run | params trunk | params total | train s | eval s | final train loss_answer_ema |", "|---|---|---|---|---|---|"])
    for a in arms:
        for s in seeds:
            r = runs.get((a, s))
            if not r:
                continue
            t = r["_train"]
            L.append(f"| {a} s{s} | {t.get('params_trunk')} | {t.get('params_total')} | {t.get('train_wall_s')} | {r.get('eval_wall_s')} | "
                     f"{t['log'][-1]['loss_answer_ema'] if t.get('log') else 'n/a'} |")
    L.append("")
    out = os.path.join(rdir, "report_tables.md")
    open(out, "w").write("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
