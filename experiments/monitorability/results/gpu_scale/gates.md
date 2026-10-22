# Gates and primary-vs-baseline: gpu_scale

| arm | seed | G1 (H acc / d0-1) | G2 S / F | G3 lies / truthful | interpretable | primary AUROC [CI] | best baseline (sign-free) | margin |
|---|---|---|---|---|---|---|---|---|
| plain | 0 | FAIL (0.597 / 0.571) | 1.000 / 0.728 | PASS (1193 / 975) | NO | 0.558 [0.533, 0.583] | disagree_bow 0.720 | -0.163 |
| plain | 1 | FAIL (0.596 / 0.570) | 1.000 / 0.735 | PASS (1202 / 975) | NO | 0.577 [0.554, 0.600] | disagree_bow 0.718 | -0.140 |
| plain | 2 | FAIL (0.588 / 0.562) | 1.000 / 0.739 | PASS (1191 / 975) | NO | 0.494 [0.470, 0.517] | disagree_bow 0.713 | -0.219 |
| plain | 3 | FAIL (0.595 / 0.569) | 1.000 / 0.731 | PASS (1181 / 975) | NO | 0.491 [0.467, 0.513] | disagree_bow 0.719 | -0.228 |
| plain | 4 | FAIL (0.595 / 0.571) | 1.000 / 0.733 | PASS (1204 / 975) | NO | 0.557 [0.546, 0.569] | disagree_bow 0.710 | -0.152 |
| structured | 0 | FAIL (0.602 / 0.591) | 1.000 / 0.728 | PASS (1227 / 975) | NO | 0.667 [0.646, 0.688] | disagree_bow 0.718 | -0.051 |
| structured | 1 | FAIL (0.602 / 0.571) | 1.000 / 0.729 | PASS (1194 / 975) | NO | 0.822 [0.804, 0.839] | disagree_bow 0.714 | 0.107 |
| structured | 2 | FAIL (0.599 / 0.566) | 1.000 / 0.728 | PASS (1200 / 975) | NO | 0.709 [0.689, 0.731] | disagree_bow 0.718 | -0.009 |
| structured | 3 | FAIL (0.769 / 0.789) | 1.000 / 0.868 | PASS (1502 / 975) | NO | 0.935 [0.926, 0.944] | disagree_bow 0.699 | 0.235 |
| structured | 4 | FAIL (0.601 / 0.575) | 1.000 / 0.722 | PASS (1194 / 975) | NO | 0.646 [0.624, 0.669] | disagree_bow 0.715 | -0.069 |

Across interpretable runs (t-based 95% CI):

- plain: 0/5 interpretable; primary AUROC n/a [n/a, n/a]; margin over best baseline n/a [n/a, n/a]
- structured: 0/5 interpretable; primary AUROC n/a [n/a, n/a]; margin over best baseline n/a [n/a, n/a]
- structured - plain (interpretable pairs, n=0): n/a [n/a, n/a]
