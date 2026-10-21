"""Non-learned behavioral baselines on the exact evaluation sets used by evaluate.py.

shallow_heuristic: if the queried name is a fact -> its polarity; elif some rule concludes it ->
that rule's polarity (first such rule); else unknown. No premise checking.
Usage: python baselines.py --out results/cpu_pilot/heuristic_baseline.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import env
from evaluate import eval_sets


def shallow(ep):
    w = [env.VOCAB[t] for t in ep.tokens]
    q = (w[-5], w[-4])
    fi = w.index("facts")
    end = len(w)
    for p in env.NEUTRAL_PERSONAS + (env.PERSONA_S, env.PERSONA_F, "question"):
        if p in w[fi:]:
            end = min(end, fi + w[fi:].index(p))
    facts = w[fi + 2:end]
    for i in range(len(facts) - 2):
        if (facts[i + 1], facts[i + 2]) == q:
            return env.YES if facts[i] == "true" else env.NO
    rules = w[3:fi]
    for i in range(1, len(rules) - 2):
        if rules[i - 1] == "then" and (rules[i + 1], rules[i + 2]) == q:
            return env.YES if rules[i] == "true" else env.NO
    return env.UNK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "cpu_pilot", "heuristic_baseline.json"))
    args = ap.parse_args()
    sets = eval_sets()
    out = {"description": "shallow heuristic: queried name is a fact -> its polarity; elif a rule concludes it -> "
                          "that rule's polarity (first such rule, premises ignored); else unknown"}
    for name in ("H_train", "H_heldout"):
        eps = sets[name]
        pr = [shallow(e) for e in eps]
        acc = sum(p == e.true_label for p, e in zip(pr, eps)) / len(eps)
        by = {}
        for d in range(4):
            sub = [(p, e) for p, e in zip(pr, eps) if e.depth == d]
            by[d] = sum(p == e.true_label for p, e in sub) / len(sub)
        out[name] = {"acc": acc, "acc_by_depth": by, "n": len(eps)}
        out[name]["majority_class_acc"] = max(sum(e.true_label == c for e in eps) for c in range(3)) / len(eps)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
