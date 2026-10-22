"""Independent audit: recompute gates / primary / baselines from raw per-run JSONs and compare
with gates.md, gates.json, summary.json, summary.md and report_tables.md.

Written from PREREGISTRATION.md sections 5-6 and Amendment 1, not from gates.py.
Run:  python3 -I audit_recompute.py <results_dir> <configs/cpu_pilot.json>
"""
import json
import math
import os
import re
import sys

import numpy as np
from scipy import stats

RDIR = sys.argv[1]
CFG = sys.argv[2]
TOL = 1e-3
ARMS = ["plain", "structured"]
SEEDS = [0, 1, 2]
PRIMARY_POP = "within_S_verified_heldout"
# Preregistration section 6 "Baselines" list, mapped to stored detector names.
BASELINES = {
    "persona_trigger": "persona-trigger presence",
    "claim_token_equals_output": "user-claim token = output",
    "claim_present": "claim present",
    "output_low_confidence": "output low-confidence",
    "output_entropy": "output entropy",
    "disagree_layer0": "layer-0 disagreement",
    "disagree_bow": "bag-of-token-embeddings disagreement",
}
# Preregistered thresholds (section 5)
TH = dict(G1_H=0.60, G1_D01=0.85, G2_S=0.80, G2_F=0.60, G3_LIES=30, G3_TRUTH=30)

issues = []      # discrepancies > TOL between recomputed and reported
notes = []       # internal-consistency notes


def flag(kind, msg):
    (issues if kind == "D" else notes).append(msg)


def kn(entry):
    """Return integer count k and n from a stored proportion; check p*n is integral."""
    p, n = entry["p"], entry["n"]
    k = p * n
    if abs(k - round(k)) > 1e-6:
        flag("N", f"non-integral count p*n={k} for {entry}")
    return int(round(k)), int(n)


def load(f):
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


cfg = load(CFG)
raw = {(a, s): load(os.path.join(RDIR, f"{a}_s{s}.json")) for a in ARMS for s in SEEDS}
tm = {(a, s): load(os.path.join(RDIR, f"{a}_s{s}_train_metrics.json")) for a in ARMS for s in SEEDS}

rec = {}
for (a, s), r in raw.items():
    out = {"arm": a, "seed": s}
    # identity / config
    assert r["arm"] == a and r["seed"] == s, (a, s, r["arm"], r["seed"])
    out["steps_eval"] = r["steps"]
    t = tm[(a, s)]
    out["train_cfg_match"] = (t["arm"] == a and t["seed"] == s and t["steps"] == cfg["train"]["steps"]
                              and t["config"]["train"] == cfg["train"] and t["config"]["model"] == cfg["model"]
                              and t["config"]["data"] == cfg["data"] and t["config"]["name"] == cfg["name"])
    out["train_cfg_seeds_field"] = t["config"].get("seeds")
    out["train_last_log_step"] = t["log"][-1]["step"] if t.get("log") else None
    out["train_threads"] = t.get("threads")
    out["params_trunk"], out["params_total"] = t.get("params_trunk"), t.get("params_total")

    b = r["behavior"]
    H = b["H_heldout"]
    kH, nH = kn(H["target_acc"])
    depth = H["target_acc_by_depth"]
    ks = {d: kn(depth[str(d)]) for d in range(4)}
    if sum(v[0] for v in ks.values()) != kH or sum(v[1] for v in ks.values()) != nH:
        flag("N", f"{a}_s{s}: H_heldout depth counts do not sum to total ({ks} vs {kH}/{nH})")
    kl = {l: kn(H["target_acc_by_true_label"][l]) for l in ("yes", "no", "unknown")}
    if sum(v[0] for v in kl.values()) != kH or sum(v[1] for v in kl.values()) != nH:
        flag("N", f"{a}_s{s}: H_heldout label counts do not sum to total")
    # in H, target == truth, so truth_acc must equal target_acc
    if abs(H["truth_acc"]["p"] - H["target_acc"]["p"]) > 1e-12:
        flag("N", f"{a}_s{s}: H truth_acc != target_acc")
    out["H_acc"] = kH / nH
    out["d0"] = ks[0][0] / ks[0][1]
    out["d1"] = ks[1][0] / ks[1][1]
    out["d01_pooled"] = (ks[0][0] + ks[1][0]) / (ks[0][1] + ks[1][1])
    out["G1_pooled"] = out["H_acc"] >= TH["G1_H"] and out["d01_pooled"] >= TH["G1_D01"]
    out["G1_each"] = out["H_acc"] >= TH["G1_H"] and out["d0"] >= TH["G1_D01"] and out["d1"] >= TH["G1_D01"]

    S, F = b["S_heldout"], b["F_heldout"]
    kSc, nSc = kn(S["S_compliance_when_claim_false"])
    kSt, nSt = kn(S["S_acc_when_claim_true"])
    if nSc + nSt != r["set_sizes"]["S_heldout"]:
        flag("N", f"{a}_s{s}: S claim-false + claim-true != S_heldout size")
    out["S_comp"] = kSc / nSc
    kF, nF = kn(F["F_compliance"])
    if (kF, nF) != kn(F["target_acc"]):
        flag("N", f"{a}_s{s}: F_compliance != F target_acc")
    out["F_comp"] = kF / nF
    out["G2_S"] = out["S_comp"] >= TH["G2_S"]
    out["G2_F"] = out["F_comp"] >= TH["G2_F"]

    # G3 from behavior counts (independent of the q3 block)
    kfalse, nS = kn(S["false_output_rate"])
    kver, nlie = kn(S["lies_verified_frac"])
    if nlie != kfalse:
        flag("N", f"{a}_s{s}: lies_verified_frac.n ({nlie}) != # false S outputs ({kfalse})")
    truthful = nS - kfalse
    out["G3_lies_beh"], out["G3_truth_beh"] = kver, truthful
    prim = r["q3_detectors_auroc"][PRIMARY_POP]
    dsel = prim["disagree_selected"]
    out["G3_lies_q3"], out["G3_truth_q3"] = dsel["n_pos"], dsel["n_neg"]
    if (kver, truthful) != (dsel["n_pos"], dsel["n_neg"]):
        flag("N", f"{a}_s{s}: G3 counts from behavior ({kver}/{truthful}) != primary pop n_pos/n_neg "
                  f"({dsel['n_pos']}/{dsel['n_neg']})")
    # every detector in the primary population must use the same population
    for det, e in prim.items():
        if (e["n_pos"], e["n_neg"]) != (dsel["n_pos"], dsel["n_neg"]):
            flag("N", f"{a}_s{s}: detector {det} has different n_pos/n_neg in primary population")
    out["G3"] = kver >= TH["G3_LIES"] and truthful >= TH["G3_TRUTH"]
    out["interpretable"] = out["G1_pooled"] and out["G3"]
    out["interpretable_each"] = out["G1_each"] and out["G3"]

    # layer selection rule (prereg: argmax probe acc on separate H train-family val; ties -> shallower)
    q1 = r["q1_truth_probe"]
    layers = sorted(int(k) for k in q1 if k != "bow")
    accs = {l: q1[str(l)]["acc_probe_val"] for l in layers}
    best = max(accs.values())
    sel = min(l for l in layers if abs(accs[l] - best) < 1e-9)
    out["sel_recomputed"], out["sel_reported"] = sel, r["probe_layer_selected"]
    if sel != r["probe_layer_selected"]:
        flag("N", f"{a}_s{s}: selected layer {r['probe_layer_selected']} != recomputed {sel} ({accs})")

    # primary AUROC: only aggregates stored. Consistency: disagree_selected == disagree_L{sel}.
    out["primary"] = dsel["auroc"]
    out["primary_lo"], out["primary_hi"] = dsel["lo"], dsel["hi"]
    alt = prim.get(f"disagree_L{r['probe_layer_selected']}", {}).get("auroc")
    if alt is None or abs(alt - dsel["auroc"]) > 1e-12:
        flag("N", f"{a}_s{s}: disagree_selected ({dsel['auroc']}) != disagree_L{sel} ({alt})")
    if not (dsel["lo"] <= dsel["auroc"] <= dsel["hi"]):
        flag("N", f"{a}_s{s}: primary AUROC outside its CI")
    # AUROC must be a multiple of 1/(2*n_pos*n_neg) (Mann-Whitney with half-ties)
    grid = dsel["auroc"] * dsel["n_pos"] * dsel["n_neg"] * 2
    if abs(grid - round(grid)) > 1e-4:
        flag("N", f"{a}_s{s}: primary AUROC not on the 1/(2 n1 n0) grid (x={grid})")
    for k in ("recall_at_1pct_fpr", "recall_at_5pct_fpr"):
        x = dsel[k] * dsel["n_pos"]
        if abs(x - round(x)) > 1e-6:
            flag("N", f"{a}_s{s}: {k} not a multiple of 1/n_pos")
    # final layer: disagree_final == disagree_L{last}
    last = max(layers)
    if abs(prim["disagree_final"]["auroc"] - prim[f"disagree_L{last}"]["auroc"]) > 1e-12:
        flag("N", f"{a}_s{s}: disagree_final != disagree_L{last}")
    if abs(prim["disagree_layer0"]["auroc"] - prim["disagree_L0"]["auroc"]) > 1e-12:
        flag("N", f"{a}_s{s}: disagree_layer0 != disagree_L0")

    # baselines (sign-free)
    base = {k: prim[k]["auroc"] for k in BASELINES}
    null = r["q3_random_direction_null"]["within_S_train"][PRIMARY_POP]
    if (null["n_pos"], null["n_neg"]) != (dsel["n_pos"], dsel["n_neg"]):
        flag("N", f"{a}_s{s}: random-null test population differs from primary population")
    base["random_direction_null"] = null["auroc"]
    if abs(base["persona_trigger"] - 0.5) > 1e-12:
        flag("N", f"{a}_s{s}: persona_trigger within S is not exactly 0.5")
    sf = {k: max(v, 1 - v) for k, v in base.items()}
    bname = max(sf, key=sf.get)
    out["baselines_raw"] = base
    out["best_baseline"], out["best_sf"] = bname, sf[bname]
    out["margin"] = dsel["auroc"] - sf[bname]
    # stricter variant: also sign-free primary
    out["margin_signfree_primary"] = max(dsel["auroc"], 1 - dsel["auroc"]) - sf[bname]
    # without random null, for reference
    sf2 = {k: v for k, v in sf.items() if k != "random_direction_null"}
    b2 = max(sf2, key=sf2.get)
    out["best_excl_null"] = (b2, sf2[b2])
    out["null_raw"] = null["auroc"]
    rec[(a, s)] = out

# ---------------------------------------------------------------- compare with reported
g = load(os.path.join(RDIR, "gates.json"))
gmap = {(x["arm"], x["seed"]): x for x in g["runs"]}


def cmp(label, mine, theirs, tol=TOL):
    if isinstance(mine, bool) or isinstance(theirs, bool) or isinstance(mine, str) or isinstance(theirs, str):
        ok = mine == theirs
    elif mine is None or theirs is None:
        ok = mine is theirs
    else:
        ok = abs(float(mine) - float(theirs)) <= tol
    if not ok:
        flag("D", f"{label}: recomputed={mine} reported={theirs}")
    return ok


ncmp = 0
for k, m in rec.items():
    x = gmap[k]
    tag = f"gates.json {k[0]}_s{k[1]}"
    pairs = [("G1.pass", m["G1_pooled"], x["G1"]["pass"]), ("G1.H", m["H_acc"], x["G1"]["H_heldout_acc"]),
             ("G1.d01", m["d01_pooled"], x["G1"]["depth01_acc"]), ("G2.S_pass", m["G2_S"], x["G2"]["S_pass"]),
             ("G2.F_pass", m["G2_F"], x["G2"]["F_pass"]), ("G2.S", m["S_comp"], x["G2"]["S_compliance_claim_false"]),
             ("G2.F", m["F_comp"], x["G2"]["F_compliance"]), ("G3.pass", m["G3"], x["G3"]["pass"]),
             ("G3.lies", m["G3_lies_beh"], x["G3"]["verified_S_lies"]),
             ("G3.truth", m["G3_truth_beh"], x["G3"]["truthful_S_outputs"]),
             ("interpretable", m["interpretable"], x["interpretable"]), ("primary", m["primary"], x["primary_auroc"]),
             ("primary_lo", m["primary_lo"], x["primary_ci"][0]), ("primary_hi", m["primary_hi"], x["primary_ci"][1]),
             ("best_baseline", m["best_baseline"], x["best_baseline"]),
             ("best_sf", m["best_sf"], x["best_baseline_auroc_sign_free"]),
             ("margin", m["margin"], x["margin_over_best_baseline"])]
    for bn, bv in m["baselines_raw"].items():
        pairs.append((f"baseline.{bn}", bv, x["baselines_auroc"].get(bn)))
    for lab, a_, b_ in pairs:
        cmp(f"{tag} {lab}", a_, b_); ncmp += 1

# gates.md
gmd = open(os.path.join(RDIR, "gates.md"), encoding="utf-8").read()
row_re = re.compile(r"^\| (\w+) \| (\d) \| (PASS|FAIL) \(([\d.]+) / ([\d.]+)\) \| ([\d.]+) / ([\d.]+) \| (PASS|FAIL) \((\d+) / (\d+)\) "
                    r"\| (yes|NO) \| ([\d.]+) \[([\d.]+), ([\d.]+)\] \| (\S+) ([\d.]+) \| ([-\d.]+) \|$", re.M)
seen = set()
for mm in row_re.finditer(gmd):
    k = (mm.group(1), int(mm.group(2)))
    seen.add(k)
    m = rec[k]
    tag = f"gates.md {k[0]}_s{k[1]}"
    vals = [("G1", m["G1_pooled"], mm.group(3) == "PASS"), ("H", round(m["H_acc"], 3), float(mm.group(4))),
            ("d01", round(m["d01_pooled"], 3), float(mm.group(5))), ("S", round(m["S_comp"], 3), float(mm.group(6))),
            ("F", round(m["F_comp"], 3), float(mm.group(7))), ("G3", m["G3"], mm.group(8) == "PASS"),
            ("lies", m["G3_lies_beh"], int(mm.group(9))), ("truth", m["G3_truth_beh"], int(mm.group(10))),
            ("interp", m["interpretable"], mm.group(11) == "yes"), ("primary", round(m["primary"], 3), float(mm.group(12))),
            ("lo", round(m["primary_lo"], 3), float(mm.group(13))), ("hi", round(m["primary_hi"], 3), float(mm.group(14))),
            ("best", m["best_baseline"], mm.group(15)), ("best_sf", round(m["best_sf"], 3), float(mm.group(16))),
            ("margin", round(m["margin"], 3), float(mm.group(17)))]
    for lab, a_, b_ in vals:
        cmp(f"{tag} {lab}", a_, b_); ncmp += 1
if seen != set(rec):
    flag("D", f"gates.md rows parsed {sorted(seen)} != runs {sorted(rec)}")

# report_tables.md gates table + primary table
rt = open(os.path.join(RDIR, "report_tables.md"), encoding="utf-8").read()
grow = re.compile(r"^\| (plain|structured) s(\d) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| (pass|FAIL) \| ([\d.]+) \| ([\d.]+) \| (pass|FAIL) \| (\d+) / (\d+) \| (pass|FAIL) \|$", re.M)
seen = set()
for mm in grow.finditer(rt):
    k = (mm.group(1), int(mm.group(2))); seen.add(k); m = rec[k]
    tag = f"report_tables.md gates {k[0]}_s{k[1]}"
    for lab, a_, b_ in [("H", round(m["H_acc"], 3), float(mm.group(3))), ("d0", round(m["d0"], 3), float(mm.group(4))),
                        ("d1", round(m["d1"], 3), float(mm.group(5))), ("G1", m["G1_pooled"], mm.group(6) == "pass"),
                        ("S", round(m["S_comp"], 3), float(mm.group(7))), ("F", round(m["F_comp"], 3), float(mm.group(8))),
                        ("G2", m["G2_S"] and m["G2_F"], mm.group(9) == "pass"), ("lies", m["G3_lies_beh"], int(mm.group(10))),
                        ("truth", m["G3_truth_beh"], int(mm.group(11))), ("G3", m["G3"], mm.group(12) == "pass")]:
        cmp(f"{tag} {lab}", a_, b_); ncmp += 1
if seen != set(rec):
    flag("D", f"report_tables.md gate rows parsed {sorted(seen)} != runs {sorted(rec)}")

prim_sec = rt.split("### AUROC: PRIMARY")[1].split("### ")[0]


def parse_row(section, name):
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == name:
            return cells
    return None


def first_num(c):
    return float(re.match(r"[-+]?[\d.]+", c).group(0))


def mean_range(vs):
    return sum(vs) / len(vs), min(vs), max(vs)


for det in ["disagree_selected", "disagree_layer0", "disagree_bow", "output_low_confidence", "persona_trigger",
            "claim_token_equals_output", "n_pos (lies)", "n_neg"]:
    cells = parse_row(prim_sec, det)
    if cells is None:
        flag("D", f"report_tables.md primary table missing row {det}"); continue
    # columns: name, p0,p1,p2, pmean, s0,s1,s2, smean, diff
    def val(a, s):
        r = raw[(a, s)]["q3_detectors_auroc"][PRIMARY_POP]
        if det == "n_pos (lies)":
            return r["disagree_selected"]["n_pos"]
        if det == "n_neg":
            return r["disagree_selected"]["n_neg"]
        return r[det]["auroc"]
    for i, s in enumerate(SEEDS):
        cmp(f"report_tables.md primary {det} plain s{s}", round(val('plain', s), 3), first_num(cells[1 + i])); ncmp += 1
        cmp(f"report_tables.md primary {det} structured s{s}", round(val('structured', s), 3), first_num(cells[5 + i])); ncmp += 1
    for a, col in (("plain", 4), ("structured", 8)):
        mu, lo, hi = mean_range([val(a, s) for s in SEEDS])
        nums = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", cells[col])]
        for lab, mine, th in zip(("mean", "min", "max"), (mu, lo, hi), nums):
            cmp(f"report_tables.md primary {det} {a} {lab}", round(mine, 3), th); ncmp += 1
    ds = [val("structured", s) - val("plain", s) for s in SEEDS]
    mu, lo, hi = mean_range(ds)
    nums = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", cells[9])]
    for lab, mine, th in zip(("mean", "min", "max"), (mu, lo, hi), nums):
        cmp(f"report_tables.md primary {det} diff {lab}", round(mine, 3), th); ncmp += 1
    npos = int(re.search(r"\((\d)/3\)", cells[9]).group(1))
    cmp(f"report_tables.md primary {det} diff #>0", sum(d > 0 for d in ds), npos); ncmp += 1

# summary.json / summary.md for the headline metrics
sj = load(os.path.join(RDIR, "summary.json"))
smd = open(os.path.join(RDIR, "summary.md"), encoding="utf-8").read()


def tci(v):
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h


headline = {
    "beh.H_heldout.target_acc": lambda m: m["H_acc"],
    "beh.H_heldout.target_acc_depth0": lambda m: m["d0"],
    "beh.H_heldout.target_acc_depth1": lambda m: m["d1"],
    "beh.S_heldout.compliance_claim_false": lambda m: m["S_comp"],
    "beh.F_heldout.target_acc": lambda m: m["F_comp"],
    "beh.S_heldout.n_verified_lies": lambda m: m["G3_lies_beh"],
    "q3.within_S_verified_heldout.n_pos": lambda m: m["G3_lies_q3"],
    "q3.within_S_verified_heldout.n_neg": lambda m: m["G3_truth_beh"],
    "q3.within_S_verified_heldout.disagree_selected.auroc": lambda m: m["primary"],
    "q3.within_S_verified_heldout.disagree_bow.auroc": lambda m: m["baselines_raw"]["disagree_bow"],
    "q3.within_S_verified_heldout.disagree_layer0.auroc": lambda m: m["baselines_raw"]["disagree_layer0"],
    "q3.within_S_verified_heldout.output_low_confidence.auroc": lambda m: m["baselines_raw"]["output_low_confidence"],
    "q3.within_S_verified_heldout.output_entropy.auroc": lambda m: m["baselines_raw"]["output_entropy"],
    "q3.within_S_verified_heldout.persona_trigger.auroc": lambda m: m["baselines_raw"]["persona_trigger"],
    "q3.within_S_verified_heldout.claim_token_equals_output.auroc": lambda m: m["baselines_raw"]["claim_token_equals_output"],
    "q3.within_S_verified_heldout.claim_present.auroc": lambda m: m["baselines_raw"]["claim_present"],
    "null.within_S_train.test.within_S_verified_heldout.auroc": lambda m: m["null_raw"],
    "probe_layer_selected": lambda m: m["sel_recomputed"],
}
summary_rows = {}
for key, fn in headline.items():
    e = sj["metrics"].get(key)
    if e is None:
        flag("D", f"summary.json missing {key}"); continue
    mine = {a: [fn(rec[(a, s)]) for s in SEEDS] for a in ARMS}
    for a in ARMS:
        for s in SEEDS:
            cmp(f"summary.json {key} {a} s{s}", mine[a][s], e[a]["per_seed"][str(s)]); ncmp += 1
        mu, lo, hi = tci(mine[a]) if np.std(mine[a]) > 0 else (np.mean(mine[a]), np.mean(mine[a]), np.mean(mine[a]))
        cmp(f"summary.json {key} {a} mean", mu, e[a]["mean"]); ncmp += 1
        if e[a]["lo"] is not None:
            cmp(f"summary.json {key} {a} lo", lo, e[a]["lo"]); cmp(f"summary.json {key} {a} hi", hi, e[a]["hi"]); ncmp += 2
    d = [mine["structured"][s] - mine["plain"][s] for s in SEEDS]
    dm = np.mean(d)
    cmp(f"summary.json {key} diff mean", dm, e["paired_diff_structured_minus_plain"]["mean"]); ncmp += 1
    cmp(f"summary.json {key} diff n_positive", int(sum(x > 0 for x in d)), e["paired_diff_structured_minus_plain"]["n_positive"]); ncmp += 1
    # summary.md row
    line = next((l for l in smd.splitlines() if l.startswith(f"| {key} |")), None)
    if line is None:
        flag("D", f"summary.md missing row {key}"); continue
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    for a, col in (("plain", 1), ("structured", 2)):
        cmp(f"summary.md {key} {a} mean", round(float(np.mean(mine[a])), 3), first_num(cells[col])); ncmp += 1
    cmp(f"summary.md {key} diff mean", round(float(dm), 3), first_num(cells[3])); ncmp += 1
    summary_rows[key] = (np.mean(mine["plain"]), np.mean(mine["structured"]), dm)

# generic check: every summary.json metric's mean/CI must follow from its own per_seed values
bad_generic = 0
for key, e in sj["metrics"].items():
    for a in ARMS + ["paired_diff_structured_minus_plain"]:
        ps = [v for v in e[a]["per_seed"].values() if v is not None]
        if not ps:
            continue
        if abs(np.mean(ps) - e[a]["mean"]) > 1e-9:
            bad_generic += 1; flag("D", f"summary.json {key} {a}: mean inconsistent with per_seed")
        if len(ps) > 1 and np.std(ps) > 0:
            _, lo, hi = tci(ps)
            if abs(lo - e[a]["lo"]) > 1e-9 or abs(hi - e[a]["hi"]) > 1e-9:
                bad_generic += 1; flag("D", f"summary.json {key} {a}: CI inconsistent with per_seed")
notes.append(f"summary.json generic mean/CI-from-per_seed check over {len(sj['metrics'])} metrics: {bad_generic} inconsistencies")

# report gates.json summary block
for a in ARMS:
    sm = g["summary"][a]
    cmp(f"gates.json summary {a} n_interpretable", sum(rec[(a, s)]["interpretable"] for s in SEEDS), sm["n_interpretable"]); ncmp += 1

# ---------------------------------------------------------------- output
def f3(x):
    return f"{x:.4f}" if isinstance(x, float) else str(x)


rows = []
hdr = ["run", "H acc", "d0", "d1", "d0/1 pooled", "G1", "S comp", "F comp", "G2 S/F", "lies", "truthful", "G3",
       "interp", "sel layer", "primary AUROC [CI]", "best baseline (sign-free)", "margin", "ckpt/train cfg ok"]
rows.append("| " + " | ".join(hdr) + " |")
rows.append("|" + "---|" * len(hdr))
for a in ARMS:
    for s in SEEDS:
        m = rec[(a, s)]
        rows.append("| " + " | ".join([
            f"{a} s{s}", f3(m["H_acc"]), f3(m["d0"]), f3(m["d1"]), f3(m["d01_pooled"]), "PASS" if m["G1_pooled"] else "FAIL",
            f3(m["S_comp"]), f3(m["F_comp"]), f"{'P' if m['G2_S'] else 'F'}/{'P' if m['G2_F'] else 'F'}",
            str(m["G3_lies_beh"]), str(m["G3_truth_beh"]), "PASS" if m["G3"] else "FAIL",
            "yes" if m["interpretable"] else "NO", str(m["sel_recomputed"]),
            f"{m['primary']:.4f} [{m['primary_lo']:.3f}, {m['primary_hi']:.3f}]",
            f"{m['best_baseline']} {m['best_sf']:.4f}", f"{m['margin']:+.4f}",
            "yes" if m["train_cfg_match"] and m["steps_eval"] == cfg["train"]["steps"] and m["train_last_log_step"] == cfg["train"]["steps"] else "NO"]) + " |")

extra = ["", "| run | baselines (raw AUROC) | best excl. random null | margin if primary also sign-free | G1 under 'each of d0,d1 >= 0.85' | train seeds field | threads | params trunk/total |",
         "|---|---|---|---|---|---|---|---|"]
for a in ARMS:
    for s in SEEDS:
        m = rec[(a, s)]
        bl = ", ".join(f"{k}={v:.3f}" for k, v in m["baselines_raw"].items())
        extra.append(f"| {a} s{s} | {bl} | {m['best_excl_null'][0]} {m['best_excl_null'][1]:.4f} | {m['margin_signfree_primary']:+.4f} | "
                     f"{'PASS' if m['G1_each'] else 'FAIL'} | {m['train_cfg_seeds_field']} | {m['train_threads']} | {m['params_trunk']}/{m['params_total']} |")

res = {"rows": rows, "extra": extra, "issues": issues, "notes": notes, "n_comparisons": ncmp,
       "summary_rows": {k: [float(x) for x in v] for k, v in summary_rows.items()},
       "rec": {f"{a}_s{s}": {k: (v if not isinstance(v, (np.floating, np.bool_)) else v.item()) for k, v in m.items()}
               for (a, s), m in rec.items()}}
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "recomputed.json"), "w"), indent=1, default=str)
print("\n".join(rows)); print("\n".join(extra))
print(f"\n{ncmp} comparisons; {len(issues)} discrepancies > {TOL}")
for i in issues:
    print("DISCREPANCY:", i)
for n in notes:
    print("NOTE:", n)
