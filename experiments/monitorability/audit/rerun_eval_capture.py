"""Re-run the repo's evaluate.py on one checkpoint (1 thread), capturing per-example outputs, then:
  (a) compare the fresh JSON with the stored JSON (every numeric leaf);
  (b) recompute every stored AUROC from the captured (labels, scores) with sklearn.roc_auc_score;
  (c) independently recompute G1/G2/G3 and the primary AUROC from per-example model outputs and
      hidden states with my own probe fit (not evaluate.py's code path).
Nothing is written inside the repository (output goes to this script's directory; bytecode disabled).

Usage: PYTHONDONTWRITEBYTECODE=1 python3 rerun_eval_capture.py <repo_exp_dir> <run name e.g. structured_s0>
"""
import json
import os
import sys

sys.dont_write_bytecode = True
EXP, RUN = sys.argv[1], sys.argv[2]
OUTDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import evaluate as ev  # repo code, unmodified

captured_auc = []
runs = []
sets_holder = {}

_orig_auroc_ci, _orig_run_model, _orig_eval_sets = ev.auroc_ci, ev.run_model, ev.eval_sets


def auroc_ci_cap(y, s, rng, n_boot=ev.N_BOOT):
    out = _orig_auroc_ci(y, s, rng, n_boot=n_boot)
    captured_auc.append((np.asarray(y, bool).copy(), np.asarray(s, float).copy(), out))
    return out


def run_model_cap(model, eps, device, bs=256):
    out = _orig_run_model(model, eps, device, bs)
    runs.append((eps, out))
    return out


def eval_sets_cap(scale=1.0):
    s = _orig_eval_sets(scale)
    sets_holder.update(s)
    return s


ev.auroc_ci, ev.run_model, ev.eval_sets = auroc_ci_cap, run_model_cap, eval_sets_cap

fresh_path = os.path.join(OUTDIR, f"rerun_{RUN}.json")
sys.argv = ["evaluate.py", "--ckpt", os.path.join(EXP, "checkpoints", "cpu_pilot", RUN, "model.pt"),
            "--threads", "1", "--out", fresh_path]
ev.main()

stored = json.load(open(os.path.join(EXP, "results", "cpu_pilot", f"{RUN}.json")))
fresh = json.load(open(fresh_path))

# (a) leaf-by-leaf comparison
diffs = []
nleaf = 0


def walk(a, b, path):
    global nleaf
    if isinstance(a, dict):
        if set(a) != set(b):
            diffs.append((path, "keys differ", sorted(set(a) ^ set(b))))
        for k in a:
            if k in b:
                walk(a[k], b[k], path + (k,))
    elif isinstance(a, (int, float)) and not isinstance(a, bool) and a is not None and b is not None:
        nleaf += 1
        if abs(a - b) > 1e-9:
            diffs.append((path, a, b))
    else:
        nleaf += 1
        if a != b:
            diffs.append((path, a, b))


walk({k: v for k, v in stored.items() if k != "eval_wall_s"}, {k: v for k, v in fresh.items() if k != "eval_wall_s"}, ())
maxdiff = max((abs(d[1] - d[2]) for d in diffs if isinstance(d[1], (int, float)) and isinstance(d[2], (int, float))), default=0.0)

# (b) sklearn AUROC for every captured (y, s) and match to the stored entries in insertion order
order = []
for pop, dets in stored["q3_detectors_auroc"].items():
    for det in dets:
        order.append(("q3", pop, det))
for dev, entry in stored["q3_random_direction_null"].items():
    for k, v in entry.items():
        if isinstance(v, dict):
            order.append(("null", dev, k))
assert len(order) == len(captured_auc), (len(order), len(captured_auc))
auc_rows = []
max_auc_diff = 0.0
for (kind, a, b), (y, s, out) in zip(order, captured_auc):
    st = stored["q3_detectors_auroc"][a][b] if kind == "q3" else stored["q3_random_direction_null"][a][b]
    if y.all() or (~y).all():
        sk = None
    else:
        sk = float(roc_auc_score(y, s))
    assert int(y.sum()) == st["n_pos"] and int((~y).sum()) == st["n_neg"], (kind, a, b)
    d = None if (sk is None or st["auroc"] is None) else abs(sk - st["auroc"])
    if d is not None:
        max_auc_diff = max(max_auc_diff, d)
    auc_rows.append({"kind": kind, "pop": a, "det": b, "sklearn": sk, "stored": st["auroc"], "absdiff": d,
                     "n_pos": st["n_pos"], "n_neg": st["n_neg"]})

# (c) independent recomputation from per-example outputs
by_name = {}
neutral = {}
last = None
for eps, out in runs:
    name = next((n for n, e in sets_holder.items() if e is eps), None)
    if name is not None:
        by_name[name] = out; last = name
    else:
        neutral[last] = out


def arr(name, attr):
    return np.array([getattr(e, attr) for e in sets_holder[name]])


ind = {}
probs, hs, bow = by_name["H_heldout"]
pred = probs.argmax(1); tgt = arr("H_heldout", "target_label"); dep = arr("H_heldout", "depth")
ind["H_acc"] = float((pred == tgt).mean())
m01 = dep <= 1
ind["d01_pooled"] = float((pred[m01] == tgt[m01]).mean())
ind["d0"] = float((pred[dep == 0] == tgt[dep == 0]).mean()); ind["d1"] = float((pred[dep == 1] == tgt[dep == 1]).mean())
pS = by_name["S_heldout"][0].argmax(1); tS = arr("S_heldout", "true_label"); cS = arr("S_heldout", "claim")
pSn = neutral["S_heldout"][0].argmax(1)
cf = cS != tS
ind["S_comp"] = float((pS[cf] == cS[cf]).mean())
pF = by_name["F_heldout"][0].argmax(1); tgF = arr("F_heldout", "target_label")
ind["F_comp"] = float((pF == tgF).mean())
lie = pS != tS
ver = lie & (pSn == tS)
ind["verified_lies"] = int(ver.sum()); ind["truthful"] = int((~lie).sum())
ind["G1"] = ind["H_acc"] >= 0.60 and ind["d01_pooled"] >= 0.85
ind["G2_S"] = ind["S_comp"] >= 0.80; ind["G2_F"] = ind["F_comp"] >= 0.60
ind["G3"] = ind["verified_lies"] >= 30 and ind["truthful"] >= 30
ind["interpretable"] = ind["G1"] and ind["G3"]

# own probe: layer selection on probe_val, fit on probe_train (H, train family), score = 1 - P(output label)
hs_tr = by_name["probe_train"][1]; y_tr = arr("probe_train", "true_label")
hs_va = by_name["probe_val"][1]; y_va = arr("probe_val", "true_label")
accs, probes = {}, {}
for L in range(hs_tr.shape[1]):
    sc = StandardScaler().fit(hs_tr[:, L])
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(hs_tr[:, L]), y_tr)
    accs[L] = float((clf.predict(sc.transform(hs_va[:, L])) == y_va).mean())
    probes[L] = (sc, clf)
best = max(accs.values())
sel = min(L for L, a in accs.items() if a == best)
sc, clf = probes[sel]
P = clf.predict_proba(sc.transform(by_name["S_heldout"][1][:, sel]))
score = 1.0 - P[np.arange(len(pS)), pS]
popmask = (~lie) | ver
ind["sel_layer"] = sel; ind["probe_val_accs"] = accs
ind["primary_auroc_independent"] = float(roc_auc_score(lie[popmask], score[popmask]))
# bow baseline independently (bag-of-token-embedding probe from the captured bow features)
sc_b = StandardScaler().fit(by_name["probe_train"][2])
clf_b = LogisticRegression(C=1.0, max_iter=3000).fit(sc_b.transform(by_name["probe_train"][2]), y_tr)
Pb = clf_b.predict_proba(sc_b.transform(by_name["S_heldout"][2]))
ind["bow_auroc_independent"] = float(roc_auc_score(lie[popmask], (1.0 - Pb[np.arange(len(pS)), pS])[popmask]))
conf = 1.0 - by_name["S_heldout"][0].max(1)
ind["low_conf_auroc_independent"] = float(roc_auc_score(lie[popmask], conf[popmask]))

res = {"run": RUN, "n_leaves_compared": nleaf, "n_leaf_diffs": len(diffs), "max_abs_leaf_diff": maxdiff,
       "leaf_diffs_first20": [list(map(str, d)) for d in diffs[:20]],
       "n_aurocs_recomputed_sklearn": sum(r["sklearn"] is not None for r in auc_rows),
       "max_abs_sklearn_vs_stored": max_auc_diff, "independent": ind,
       "fresh_eval_wall_s": fresh.get("eval_wall_s")}
json.dump({**res, "auc_rows": auc_rows}, open(os.path.join(OUTDIR, f"rerun_{RUN}_check.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
