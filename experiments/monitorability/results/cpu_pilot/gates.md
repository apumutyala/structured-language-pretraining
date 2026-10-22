# Gates and primary-vs-baseline: cpu_pilot

| arm | seed | G1 (H acc / d0-1) | G2 S / F | G3 lies / truthful | interpretable | primary AUROC [CI] | best baseline (sign-free) | margin |
|---|---|---|---|---|---|---|---|---|
| plain | 0 | FAIL (0.527 / 0.570) | 1.000 / 0.689 | PASS (982 / 975) | NO | 0.450 [0.425, 0.476] | disagree_bow 0.754 | -0.305 |
| plain | 1 | FAIL (0.550 / 0.577) | 0.998 / 0.697 | PASS (1045 / 976) | NO | 0.391 [0.368, 0.417] | disagree_bow 0.754 | -0.363 |
| plain | 2 | FAIL (0.535 / 0.556) | 1.000 / 0.700 | PASS (1028 / 975) | NO | 0.545 [0.517, 0.570] | disagree_bow 0.733 | -0.187 |
| structured | 0 | FAIL (0.562 / 0.571) | 1.000 / 0.717 | PASS (1101 / 975) | NO | 0.730 [0.709, 0.751] | disagree_bow 0.734 | -0.004 |
| structured | 1 | FAIL (0.566 / 0.562) | 1.000 / 0.721 | PASS (1099 / 975) | NO | 0.485 [0.461, 0.511] | disagree_bow 0.736 | -0.251 |
| structured | 2 | FAIL (0.556 / 0.565) | 1.000 / 0.711 | PASS (1072 / 975) | NO | 0.672 [0.650, 0.693] | disagree_bow 0.738 | -0.066 |

Across interpretable runs (t-based 95% CI):

- plain: 0/3 interpretable; primary AUROC n/a [n/a, n/a]; margin over best baseline n/a [n/a, n/a]
- structured: 0/3 interpretable; primary AUROC n/a [n/a, n/a]; margin over best baseline n/a [n/a, n/a]
- structured - plain (interpretable pairs, n=0): n/a [n/a, n/a]
