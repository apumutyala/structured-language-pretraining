"""Paired across-seed analysis of the routing replication (see PREREGISTRATION.md).

Inputs : results/<tag>/eval_expanded.json (+ results/<tag>/train_summary_seed_XX.json)
Outputs: results/<tag>/analysis.json and results/<tag>/analysis.md (refuses to overwrite)

Unit of analysis = replicate (seed pair). For each metric and contrast
(internal - none, output_only - none; internal - output_only exploratory) it
reports per-arm mean/SD, mean paired difference, SD, a percentile bootstrap 95%
CI over replicates, a two-sided paired t-test, an exact two-sided sign test and
an exact sign-flip permutation test. Holm correction is applied across the two
PRIMARY internal-vs-none tests. Replicate 0 (original seeds) is excluded from
the primary analysis unless --include-original is given.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ORIGINAL_RECEIPT, RESULTS_DIR, ROOT, rel, write_json_new  # noqa: E402

REGISTERED_STAGE = "matched_stability"
# name, extractor, direction that counts as "better", role, smallest effect size of interest (SESOI)
METRICS = [
    ("typed_accuracy", lambda m: m["world"]["typed_requested_key_accuracy"], "higher", "primary", 0.02),
    ("story_bpb_tiled_full", lambda m: m["story"]["tiled_full"]["bits_per_utf8_byte"], "lower", "primary", 0.01),
    ("query_byte_accuracy", lambda m: m["world"]["query_byte_accuracy"], "higher", "secondary", 0.02),
    ("answer_byte_nll_bits", lambda m: m["world"]["answer_byte_nll_bits_mean"], "lower", "secondary", 0.05),
    ("typed_target_probability", lambda m: m["world"]["typed_target_probability_mean"], "higher", "secondary", 0.02),
    ("raw41_accuracy", lambda m: m["world"]["raw41_accuracy"], "higher", "secondary", 0.02),
    ("story_bpb_trainer_windows_all_dev", lambda m: m["story"]["trainer_windows"]["bits_per_utf8_byte"], "lower",
     "secondary", 0.01),
    ("story_bpb_in_run_12_docs", lambda m: m["in_run_dev_replay"]["metrics"]["story_bits_per_utf8_byte"], "lower",
     "secondary", 0.01),
    ("typed_nonabstention_coverage", lambda m: m["world"]["typed_nonabstention_coverage"], "higher", "diagnostic", None),
    ("forbidden_mass_max", lambda m: m["world"]["forbidden_mass_max"], "lower", "diagnostic", None),
]
CONTRASTS = [("internal", "none", "primary"), ("output_only", "none", "secondary"),
             ("internal", "output_only", "exploratory")]
ALPHA = 0.05
CEILING = 0.995


# ------------------------------------------------------------------ statistics (no SciPy dependency)
def _betacf(a, b, x):
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        delta = d * c; h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc_regularized(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def t_two_sided_p(t, df):
    return betainc_regularized(df / 2.0, 0.5, df / (df + t * t))


def t_quantile_975(df):
    low, high = 0.0, 100.0
    for _ in range(200):
        mid = (low + high) / 2
        if t_two_sided_p(mid, df) > 0.05:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def sign_test_p(diffs):
    positive = sum(d > 0 for d in diffs); negative = sum(d < 0 for d in diffs)
    n = positive + negative
    if n == 0:
        return 1.0, positive, negative
    k = min(positive, negative)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail), positive, negative


def sign_flip_p(diffs, rng, exact_limit=16, draws=200000):
    values = np.asarray(diffs, dtype=float)
    observed = abs(values.mean())
    if len(values) <= exact_limit:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(values))))
        means = np.abs((signs * values).mean(1))
        return float((means >= observed - 1e-12).mean()), "exact"
    signs = rng.choice((1.0, -1.0), size=(draws, len(values)))
    means = np.abs((signs * values).mean(1))
    return float(((means >= observed - 1e-12).sum() + 1) / (draws + 1)), "monte_carlo"


def paired_stats(a, b, rng, bootstrap):
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if n > 1 else None
    result = {"n": n, "mean_diff": mean, "sd_diff": sd, "diffs": diffs}
    if n > 1:
        samples = np.asarray(diffs)[rng.integers(0, n, size=(bootstrap, n))].mean(1)
        result["bootstrap_ci95"] = [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
        if sd and sd > 0:
            t = mean / (sd / math.sqrt(n))
            result["t"] = t; result["t_p_two_sided"] = t_two_sided_p(t, n - 1)
            half = t_quantile_975(n - 1) * sd / math.sqrt(n)
            result["t_ci95"] = [mean - half, mean + half]
            result["cohens_dz"] = mean / sd
        else:
            result["t"] = None
            result["t_p_two_sided"] = 1.0 if mean == 0 else 0.0
            result["t_ci95"] = [mean, mean]
            result["cohens_dz"] = None
            result["note"] = "all paired differences identical; t undefined"
    else:
        result.update(bootstrap_ci95=None, t=None, t_p_two_sided=None, t_ci95=None, cohens_dz=None)
    p, positive, negative = sign_test_p(diffs)
    result.update(sign_test_p_two_sided=p, positive=positive, negative=negative, ties=n - positive - negative)
    result["sign_flip_p_two_sided"], result["sign_flip_method"] = sign_flip_p(diffs, rng)
    return result


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [None] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def verdict(stats, direction, sesoi, adjusted_p):
    """Preregistered decision rule (PREREGISTRATION.md section 5)."""
    ci = stats.get("t_ci95")
    boot = stats.get("bootstrap_ci95")
    if ci is None or adjusted_p is None:
        return "insufficient replicates"
    sign = 1 if direction == "higher" else -1
    favorable = sign * stats["mean_diff"] > 0
    boot_excludes_zero = boot is not None and (boot[0] > 0 or boot[1] < 0)
    if adjusted_p < ALPHA and boot_excludes_zero:
        if favorable:
            return ("SUPPORTED: improvement, at least SESOI" if sign * stats["mean_diff"] >= sesoi
                    else "SUPPORTED (statistically) but below SESOI: practically negligible improvement")
        return "REVERSED: routed arm significantly worse than the reference arm"
    if -sesoi < ci[0] and ci[1] < sesoi:
        return "NULL (equivalence): 95% CI within +/-SESOI"
    return "NULL (inconclusive): no significant difference, CI not within +/-SESOI"


# ------------------------------------------------------------------ assembly
def fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float) and (abs(value) >= 1e4 or (value != 0 and abs(value) < 1e-4)):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def reproduction_check(evaluation, summaries):
    """Replicate 0 vs the original derived receipt (exact digests, metric deltas)."""
    receipt_path = ROOT / ORIGINAL_RECEIPT
    summary = summaries.get(0)
    if summary is None or not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    rows = []
    for run in summary["runs"]:
        original = next(r for r in receipt["runs"] if r["mode"] == run["mode"])
        final_new, final_old = run["curve"][-1], original["curve"][-1]
        rows.append({"mode": run["mode"],
                     "initial_parameters_sha256_equal": run["initial_parameters_sha256"] == original["initial_parameters_sha256"],
                     "trained_batch_sha256_equal": run["trained_batch_sha256"] == original["trained_batch_sha256"],
                     "final_checkpoint_sha256_equal": final_new["checkpoint_sha256"] == final_old["checkpoint_sha256"],
                     "final_dev_bpb": final_new["dev"]["story_bits_per_utf8_byte"],
                     "original_final_dev_bpb": final_old["dev"]["story_bits_per_utf8_byte"],
                     "abs_diff_bpb": abs(final_new["dev"]["story_bits_per_utf8_byte"] - final_old["dev"]["story_bits_per_utf8_byte"]),
                     "typed_acc": final_new["dev"]["grounded_requested_key_claim_accuracy"],
                     "original_typed_acc": final_old["dev"]["grounded_requested_key_claim_accuracy"]})
    return {"receipt": rel(receipt_path), "config_sha256_equal": summary["config_sha256"] == receipt["original_config_sha256"],
            "runs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--eval", type=Path, default=None, help="default results/<tag>/eval_expanded.json")
    parser.add_argument("--include-original", action="store_true", help="include replicate 0 in the paired analysis")
    parser.add_argument("--replicates", type=str, default="1-10",
                        help="primary replicate indices, e.g. '1-10' or '1-8,11,12' (preregistered: 1-10)")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261021)
    parser.add_argument("--out-prefix", type=Path, default=None, help="default results/<tag>/analysis")
    args = parser.parse_args()
    results_dir = RESULTS_DIR / args.tag
    eval_path = args.eval or results_dir / "eval_expanded.json"
    prefix = args.out_prefix or results_dir / "analysis"
    json_path, md_path = prefix.with_suffix(".json"), prefix.with_suffix(".md")
    for path in (json_path, md_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {rel(path)}; pass --out-prefix")
    evaluation = json.loads(eval_path.read_text())
    summaries = {}
    for path in sorted(results_dir.glob("train_summary_seed_*.json")):
        summary = json.loads(path.read_text())
        summaries[summary["replicate"]] = summary

    by_replicate = {}
    for model in evaluation["models"]:
        by_replicate.setdefault(model["replicate"], {})[model["mode"]] = model
    requested = set()
    for part in args.replicates.split(","):
        low, _, high = part.partition("-")
        requested.update(range(int(low), int(high or low) + 1))
    if args.include_original:
        requested.add(0)
    included = sorted(index for index, arms in by_replicate.items()
                      if index in requested and set(arms) >= {"none", "internal", "output_only"})
    missing = sorted(requested - set(included))
    truncated = (evaluation["world_probe"].get("truncated_for_debug", False)
                 or evaluation["stories"].get("truncated_for_debug", False))
    confirmatory = evaluation["pilot_stage"] == REGISTERED_STAGE and evaluation["step"] == 1000 and not truncated
    rng = np.random.default_rng(args.bootstrap_seed)

    arm_table, contrasts = {}, []
    for name, extract, direction, role, sesoi in METRICS:
        arm_table[name] = {}
        for mode in ("none", "internal", "output_only"):
            values = [extract(by_replicate[i][mode]) for i in included]
            arm_table[name][mode] = {"values": values, "mean": float(np.mean(values)) if values else None,
                                     "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        for left, right, contrast_role in CONTRASTS:
            stats = paired_stats(arm_table[name][left]["values"], arm_table[name][right]["values"], rng, args.bootstrap)
            contrasts.append({"metric": name, "contrast": f"{left} - {right}", "metric_role": role,
                              "contrast_role": contrast_role, "better": direction, "sesoi": sesoi, **stats})
    primary = [c for c in contrasts if c["metric_role"] == "primary" and c["contrast"] == "internal - none"]
    adjusted = holm([c["t_p_two_sided"] if c["t_p_two_sided"] is not None else 1.0 for c in primary])
    for contrast, p in zip(primary, adjusted):
        contrast["holm_p"] = p
        contrast["verdict"] = verdict(contrast, contrast["better"], contrast["sesoi"], p)
    for contrast in contrasts:
        if contrast["metric_role"] == "secondary" and contrast["contrast_role"] in ("primary", "secondary"):
            contrast["verdict_unadjusted"] = verdict(contrast, contrast["better"], contrast["sesoi"],
                                                     contrast["t_p_two_sided"])
    typed_none = arm_table["typed_accuracy"]["none"]["mean"]
    ceiling = typed_none is not None and typed_none >= CEILING
    if ceiling:
        for contrast in primary:
            if contrast["metric"] == "typed_accuracy":
                contrast["verdict"] += " [CEILING: none-arm mean >= 0.995, superiority beyond SESOI untestable]"
    supported = any(c["verdict"].startswith("SUPPORTED: improvement") for c in primary)
    reversed_ = any(c["verdict"].startswith("REVERSED") for c in primary)
    overall = ("H1 MIXED: one primary endpoint improved, the other significantly worse" if supported and reversed_
               else "H1 SUPPORTED" if supported
               else "H1 REVERSED on at least one primary endpoint" if reversed_
               else "H1 NOT SUPPORTED (null)")
    if not confirmatory:
        overall = "NOT CONFIRMATORY (stage/step differ from preregistration or eval truncated): " + overall
    if len(included) < 8:
        overall = f"INCONCLUSIVE: only {len(included)} complete replicates (< 8 preregistered minimum); " + overall
    elif len(included) < 10:
        overall += f" [only {len(included)} replicates; preregistered N=10]"
    failed = sorted(i for i, s in summaries.items() if s["status"] != "complete")
    if missing:
        overall += f" [requested but missing/incomplete replicates: {missing}]"
    if failed:
        overall += f" [failed replicates reported, not replaced: {failed}]"

    world = evaluation["world_probe"]; stories = evaluation["stories"]
    payload = {"artifact_kind": "ROUTING_REPLICATION_ANALYSIS", "eval_file": rel(eval_path), "tag": args.tag,
               "pilot_stage": evaluation["pilot_stage"], "step": evaluation["step"], "confirmatory": confirmatory,
               "replicates_requested": sorted(requested), "replicates_missing": missing,
               "replicates_included": included, "include_original": args.include_original,
               "failed_replicates": {str(i): {"status": summaries[i]["status"], "error_type": summaries[i]["error_type"],
                                             "arm_status": {r["mode"]: r["status"] for r in summaries[i]["runs"]}}
                                     for i in failed},
               "bootstrap": {"resamples": args.bootstrap, "seed": args.bootstrap_seed, "method": "percentile over replicates"},
               "alpha": ALPHA, "multiplicity": "Holm over the two primary internal-none t-tests",
               "overall": overall, "typed_ceiling": ceiling,
               "data_checks": {"questions": world["questions"], "eval_ledgers": world["eval_ledgers"],
                               "overlap_with_training_world_ledgers": world["overlap_with_any_training_world_ledger"],
                               "renderer_identical": world["renderer_check"]["identical"],
                               "dev_documents_evaluated": stories["dev_documents_evaluated"],
                               "dev_text_utf8_bytes": stories["dev_text_utf8_bytes"],
                               "tiled_full_byte_targets": stories["tiled_full_counts"]["utf8_byte_targets"],
                               "trainer_windows_byte_targets": stories["trainer_windows_counts"]["utf8_byte_targets"],
                               "max_in_run_replay_abs_diff": max((m["in_run_dev_replay"]["max_abs_diff_vs_report"] or 0.0)
                                                                 for m in evaluation["models"])},
               "arms": arm_table, "contrasts": contrasts,
               "reproduction_of_original": reproduction_check(evaluation, summaries),
               "training": {str(i): {"seed": s["seed"], "data_seed": s["data_seed"], "status": s["status"],
                                     "wall_seconds": s["wall_seconds"], "peak_allocated_bytes": s["peak_allocated_bytes"],
                                     "device_name": s["device_name"]} for i, s in sorted(summaries.items())}}

    lines = [f"# Routing replication analysis ({args.tag})", "",
             f"**Overall (preregistered decision): {overall}**", "",
             f"Stage `{evaluation['pilot_stage']}`, step {evaluation['step']}, replicates {included} "
             f"(n={len(included)}; replicate 0 {'included' if args.include_original else 'excluded'}). "
             f"World probe: {world['questions']} held-out DEV-fold questions over {world['eval_ledgers']} ledgers, "
             f"overlap with training ledgers = {world['overlap_with_any_training_world_ledger']}. "
             f"Stories: {stories['dev_documents_evaluated']} DEV docs, {stories['dev_text_utf8_bytes']:,} UTF-8 bytes; "
             f"tiled byte targets {stories['tiled_full_counts']['utf8_byte_targets']:,}.", "",
             "## Per-arm mean (SD) across replicates", "",
             "| Metric | Role | none | internal | output_only |", "| --- | --- | ---: | ---: | ---: |"]
    for name, _, _, role, _ in METRICS:
        cells = [f"{fmt(arm_table[name][m]['mean'])} ({fmt(arm_table[name][m]['sd'])})" for m in ("none", "internal", "output_only")]
        lines.append(f"| {name} | {role} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired differences", "",
              "| Metric | Contrast | Mean diff | SD | Bootstrap 95% CI | t-test p | Holm p | Sign test (+/-/0, p) | Sign-flip p | Verdict |",
              "| --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: | --- |"]
    for c in contrasts:
        if c["metric_role"] == "diagnostic":
            continue
        boot = c.get("bootstrap_ci95")
        boot_text = f"[{fmt(boot[0])}, {fmt(boot[1])}]" if boot else "n/a"
        lines.append(f"| {c['metric']} | {c['contrast']} | {fmt(c['mean_diff'])} | {fmt(c['sd_diff'])} | {boot_text} | "
                     f"{fmt(c['t_p_two_sided'])} | {fmt(c.get('holm_p'))} | {c['positive']}/{c['negative']}/{c['ties']}, "
                     f"{fmt(c['sign_test_p_two_sided'])} | {fmt(c['sign_flip_p_two_sided'])} | "
                     f"{c.get('verdict') or ((c['verdict_unadjusted'] + ' (unadjusted)') if c.get('verdict_unadjusted') else c['contrast_role'])} |")
    diag = [c for c in contrasts if c["metric_role"] == "diagnostic" and c["contrast"] == "internal - none"]
    lines += ["", "Diagnostics: " + "; ".join(
        f"{name} none/internal/output_only means = " + "/".join(fmt(arm_table[name][m]["mean"]) for m in ("none", "internal", "output_only"))
        for name in ("typed_nonabstention_coverage", "forbidden_mass_max")) +
        ". Forbidden mass is zero by construction of the exact decoder; nonzero would indicate a bug."]
    lines.append(f"Max |in-run DEV replay - report| over checkpoints: {fmt(payload['data_checks']['max_in_run_replay_abs_diff'], 8)}.")
    reproduction = payload["reproduction_of_original"]
    if reproduction:
        lines += ["", "## Replicate 0 vs original receipt", "",
                  f"Config SHA equal: {reproduction['config_sha256_equal']}.", "",
                  "| Arm | init digest equal | batch digest equal | final ckpt equal | final dev bpb | original | abs diff | typed acc / original |",
                  "| --- | --- | --- | --- | ---: | ---: | ---: | --- |"]
        for row in reproduction["runs"]:
            lines.append(f"| {row['mode']} | {row['initial_parameters_sha256_equal']} | {row['trained_batch_sha256_equal']} | "
                         f"{row['final_checkpoint_sha256_equal']} | {fmt(row['final_dev_bpb'], 6)} | "
                         f"{fmt(row['original_final_dev_bpb'], 6)} | {fmt(row['abs_diff_bpb'], 6)} | "
                         f"{row['typed_acc']} / {row['original_typed_acc']} |")
    lines.append("")
    write_json_new(json_path, payload)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
