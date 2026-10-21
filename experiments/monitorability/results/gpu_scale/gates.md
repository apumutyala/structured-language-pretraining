# Gates and primary-vs-baseline: gpu_scale

| arm | seed | G1 (H acc / d0-1) | G2 S / F | G3 lies / truthful | interpretable | primary AUROC [CI] | best baseline (sign-free) | margin |
|---|---|---|---|---|---|---|---|---|
| plain | 0 | FAIL (0.597 / 0.571) | 1.000 / 0.728 | PASS (1193 / 975) | NO | 0.558 [0.533, 0.583] | disagree_bow 0.720 | -0.163 |
| structured | 0 | FAIL (0.602 / 0.591) | 1.000 / 0.728 | PASS (1227 / 975) | NO | 0.667 [0.646, 0.688] | disagree_bow 0.718 | -0.051 |

Across interpretable runs (t-based 95% CI):

- plain: 0/1 interpretable; primary AUROC n/a [n/a, n/a]; margin over best baseline n/a [n/a, n/a]
- structured: 0/1 interpretable; primary AUROC n/a [n/a, n/a]; margin over best baseline n/a [n/a, n/a]
- structured - plain (interpretable pairs, n=0): n/a [n/a, n/a]
