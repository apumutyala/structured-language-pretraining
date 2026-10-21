"""Print the report table for a finished config (copied verbatim from CODEX_TASKS.md step (c)).

Usage (from experiments/monitorability):  python report_table.py gpu_scale
"""
import json
import sys

name = sys.argv[1] if len(sys.argv) > 1 else "gpu_scale"
m = json.load(open(f"results/{name}/summary.json", encoding="utf-8"))["metrics"]
rows = [('Honest acc, held-out names', 'beh.H_heldout.target_acc'),
        ('S compliance (claim false)', 'beh.S_heldout.compliance_claim_false'),
        ('F compliance', 'beh.F_heldout.target_acc'),
        ('# verified S lies (held-out)', 'beh.S_heldout.n_verified_lies'),
        ('Truth-probe acc, H held-out', 'q1.sel.acc_H_heldout'),
        ('Truth-probe acc on S lies', 'q2.sel.truth_acc_on_lies_S_heldout'),
        ('PRIMARY disagree AUROC within-S verified held-out', 'q3.within_S_verified_heldout.disagree_selected.auroc'),
        ('  recall@5%FPR', 'q3.within_S_verified_heldout.disagree_selected.recall@5%fpr'),
        ('F-trained lie probe -> within-S verified', 'q3.within_S_verified_heldout.lie_probe_Ftrained_selected.auroc'),
        ('S-trained lie probe -> F verified vs H', 'q3.F_verified_lies_vs_H_correct_heldout.lie_probe_Strained_selected.auroc'),
        ('Persona heuristic, within-S', 'q3.within_S_verified_heldout.persona_trigger.auroc'),
        ('Claim-token=output, within-S', 'q3.within_S_verified_heldout.claim_token_equals_output.auroc'),
        ('Random-direction null (dev-selected) -> within-S verified', 'null.within_S_train.test.within_S_verified_heldout.auroc')]


def f(e):
    if e.get('mean') is None:
        return 'n/a'
    return f"{e['mean']:.3f}" if e.get('lo') is None else f"{e['mean']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}]"


print('| metric | plain | structured | diff (S-P) | seeds diff>0 |')
print('|---|---|---|---|---|')
for label, key in rows:
    e = m[key]
    d = e.get('paired_diff_structured_minus_plain', {})
    print(f"| {label} | {f(e['plain'])} | {f(e['structured'])} | {f(d) if d else 'n/a'} | {d.get('n_positive')}/{d.get('n')} |")
