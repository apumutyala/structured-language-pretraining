# Routing replication analysis (main)

**Overall (preregistered decision): H1 REVERSED on at least one primary endpoint**

Stage `matched_stability`, step 1000, replicates [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (n=10; replicate 0 excluded). World probe: 1552 held-out DEV-fold questions over 388 ledgers, overlap with training ledgers = 0. Stories: 819 DEV docs, 1,048,472 UTF-8 bytes; tiled byte targets 1,048,472.

## Per-arm mean (SD) across replicates

| Metric | Role | none | internal | output_only |
| --- | --- | ---: | ---: | ---: |
| typed_accuracy | primary | 0.9648 (0.0275) | 0.9238 (0.0364) | 0.9622 (0.0327) |
| story_bpb_tiled_full | primary | 1.8546 (0.0068) | 1.9165 (0.0085) | 1.8618 (0.0086) |
| query_byte_accuracy | secondary | 0.3488 (0.0397) | 0.3646 (0.0392) | 0.3515 (0.0451) |
| answer_byte_nll_bits | secondary | 2.4334 (0.1954) | 2.4129 (0.2574) | 2.4357 (0.2637) |
| typed_target_probability | secondary | 0.7629 (0.0370) | 0.7362 (0.0263) | 0.7658 (0.0390) |
| raw41_accuracy | secondary | 0.3298 (0.0468) | 0.4889 (0.0758) | 0.3575 (0.0604) |
| story_bpb_trainer_windows_all_dev | secondary | 1.8839 (0.0094) | 1.9467 (0.0111) | 1.8913 (0.0105) |
| story_bpb_in_run_12_docs | secondary | 1.8124 (0.0146) | 1.8727 (0.0170) | 1.8215 (0.0160) |
| typed_nonabstention_coverage | diagnostic | 1.0000 (0.0000) | 1.0000 (0.0000) | 1.0000 (0.0000) |
| forbidden_mass_max | diagnostic | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.0000 (0.0000) |

## Paired differences

| Metric | Contrast | Mean diff | SD | Bootstrap 95% CI | t-test p | Holm p | Sign test (+/-/0, p) | Sign-flip p | Verdict |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: | --- |
| typed_accuracy | internal - none | -0.0410 | 0.0171 | [-0.0514, -0.0316] | 3.384e-05 | 3.384e-05 | 0/10/0, 0.0020 | 0.0020 | REVERSED: routed arm significantly worse than the reference arm |
| typed_accuracy | output_only - none | -0.0026 | 0.0100 | [-0.0088, 0.0028] | 0.4355 | n/a | 5/5/0, 1.0000 | 0.4746 | secondary |
| typed_accuracy | internal - output_only | -0.0385 | 0.0141 | [-0.0469, -0.0306] | 1.192e-05 | n/a | 0/10/0, 0.0020 | 0.0020 | exploratory |
| story_bpb_tiled_full | internal - none | 0.0619 | 0.0075 | [0.0577, 0.0665] | 8.044e-10 | 1.609e-09 | 10/0/0, 0.0020 | 0.0020 | REVERSED: routed arm significantly worse than the reference arm |
| story_bpb_tiled_full | output_only - none | 0.0072 | 0.0037 | [0.0051, 0.0095] | 0.0002 | n/a | 10/0/0, 0.0020 | 0.0020 | secondary |
| story_bpb_tiled_full | internal - output_only | 0.0547 | 0.0065 | [0.0510, 0.0586] | 6.734e-10 | n/a | 10/0/0, 0.0020 | 0.0020 | exploratory |
| query_byte_accuracy | internal - none | 0.0158 | 0.0223 | [0.0026, 0.0289] | 0.0523 | n/a | 7/3/0, 0.3438 | 0.0586 | NULL (inconclusive): no significant difference, CI not within +/-SESOI (unadjusted) |
| query_byte_accuracy | output_only - none | 0.0026 | 0.0177 | [-0.0083, 0.0126] | 0.6478 | n/a | 5/4/1, 1.0000 | 0.6523 | NULL (equivalence): 95% CI within +/-SESOI (unadjusted) |
| query_byte_accuracy | internal - output_only | 0.0131 | 0.0230 | [-0.0006, 0.0268] | 0.1039 | n/a | 6/4/0, 0.7539 | 0.1074 | exploratory |
| answer_byte_nll_bits | internal - none | -0.0205 | 0.0831 | [-0.0661, 0.0317] | 0.4549 | n/a | 3/7/0, 0.3438 | 0.4512 | NULL (inconclusive): no significant difference, CI not within +/-SESOI (unadjusted) |
| answer_byte_nll_bits | output_only - none | 0.0023 | 0.0834 | [-0.0338, 0.0582] | 0.9338 | n/a | 3/7/0, 0.3438 | 0.9902 | NULL (inconclusive): no significant difference, CI not within +/-SESOI (unadjusted) |
| answer_byte_nll_bits | internal - output_only | -0.0228 | 0.0503 | [-0.0517, 0.0076] | 0.1855 | n/a | 3/7/0, 0.3438 | 0.1816 | exploratory |
| typed_target_probability | internal - none | -0.0267 | 0.0226 | [-0.0406, -0.0140] | 0.0046 | n/a | 1/9/0, 0.0215 | 0.0039 | REVERSED: routed arm significantly worse than the reference arm (unadjusted) |
| typed_target_probability | output_only - none | 0.0028 | 0.0093 | [-0.0029, 0.0081] | 0.3584 | n/a | 6/4/0, 0.7539 | 0.3477 | NULL (equivalence): 95% CI within +/-SESOI (unadjusted) |
| typed_target_probability | internal - output_only | -0.0295 | 0.0245 | [-0.0445, -0.0162] | 0.0041 | n/a | 0/10/0, 0.0020 | 0.0020 | exploratory |
| raw41_accuracy | internal - none | 0.1591 | 0.0421 | [0.1336, 0.1825] | 7.982e-07 | n/a | 10/0/0, 0.0020 | 0.0020 | SUPPORTED: improvement, at least SESOI (unadjusted) |
| raw41_accuracy | output_only - none | 0.0276 | 0.0185 | [0.0160, 0.0380] | 0.0011 | n/a | 9/1/0, 0.0215 | 0.0039 | SUPPORTED: improvement, at least SESOI (unadjusted) |
| raw41_accuracy | internal - output_only | 0.1314 | 0.0303 | [0.1132, 0.1486] | 2.421e-07 | n/a | 10/0/0, 0.0020 | 0.0020 | exploratory |
| story_bpb_trainer_windows_all_dev | internal - none | 0.0628 | 0.0095 | [0.0574, 0.0685] | 5.876e-09 | n/a | 10/0/0, 0.0020 | 0.0020 | REVERSED: routed arm significantly worse than the reference arm (unadjusted) |
| story_bpb_trainer_windows_all_dev | output_only - none | 0.0074 | 0.0042 | [0.0049, 0.0098] | 0.0004 | n/a | 10/0/0, 0.0020 | 0.0020 | REVERSED: routed arm significantly worse than the reference arm (unadjusted) |
| story_bpb_trainer_windows_all_dev | internal - output_only | 0.0554 | 0.0070 | [0.0513, 0.0595] | 1.290e-09 | n/a | 10/0/0, 0.0020 | 0.0020 | exploratory |
| story_bpb_in_run_12_docs | internal - none | 0.0603 | 0.0128 | [0.0528, 0.0679] | 1.226e-07 | n/a | 10/0/0, 0.0020 | 0.0020 | REVERSED: routed arm significantly worse than the reference arm (unadjusted) |
| story_bpb_in_run_12_docs | output_only - none | 0.0091 | 0.0078 | [0.0044, 0.0134] | 0.0052 | n/a | 9/1/0, 0.0215 | 0.0098 | REVERSED: routed arm significantly worse than the reference arm (unadjusted) |
| story_bpb_in_run_12_docs | internal - output_only | 0.0513 | 0.0107 | [0.0446, 0.0571] | 1.021e-07 | n/a | 10/0/0, 0.0020 | 0.0020 | exploratory |

Diagnostics: typed_nonabstention_coverage none/internal/output_only means = 1.0000/1.0000/1.0000; forbidden_mass_max none/internal/output_only means = 0.0000/0.0000/0.0000. Forbidden mass is zero by construction of the exact decoder; nonzero would indicate a bug.
Max |in-run DEV replay - report| over checkpoints: 0.00000000.

## Replicate 0 vs original receipt

Config SHA equal: True.

| Arm | init digest equal | batch digest equal | final ckpt equal | final dev bpb | original | abs diff | typed acc / original |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| none | True | True | False | 1.818783 | 1.818783 | 0.000000 | 1.0 / 1.0 |
| internal | True | True | False | 1.865863 | 1.865863 | 0.000000 | 0.875 / 0.875 |
| output_only | True | True | False | 1.819111 | 1.819111 | 0.000000 | 1.0 / 1.0 |

