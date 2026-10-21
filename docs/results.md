# Evidence and failures

All listed model results are small development diagnostics, not held-out
confirmation or peer-reviewed architecture results. Source audits are
AI-assisted reviews. The exact CPU interface tests establish bounded software
behavior; they do not establish useful learned language, general safety or
confidentiality.

The unnormalized local pilot used a four-layer shared-projection SSM, FP32,
length 128, 31 story rows plus one procedural world row per update, and one
seed. All arms consumed the same batch digest through 1,000 cumulative initial
training updates (a continuation of their own random-init lineage). Each had
4,027,000 used target tokens. At the last development check:

| Routing arm | Story bits / UTF-8 byte | World query-byte accuracy | Requested-key typed accuracy |
| --- | ---: | ---: | ---: |
| None | 2.4694725266 | 0.125 | 0.875 |
| Internal | 2.1741064555 | 0.500 | 0.875 |
| Output-only | **9199.6580425566** | 0.250 | 0.625 |

The story denominator was 3,060 UTF-8-byte targets, plus 12 EOS targets
(3,072 target tokens); the procedural check had eight rows. Typed nonabstention
coverage was 1.0 and maximum forbidden probability was 0.0 in these finite
checks. The output-only instability invalidates a clean routing-effect claim.
One seed and this tiny selected development evaluation cannot support a paired
architecture-confidence interval, useful language claim or H100 projection.
The aggregate recorded continuation-plus-prefix invocation wall time was
181.32 seconds; this includes three local arms and is not a funded-scale cost.

A parameter-free RMS variant has separate one-update CPU enablement and
eight-update local CUDA qualification receipts. Their development story bpb
values at update eight are approximately 5.1476 / 5.1655 / 5.1501 (see the exact
derived receipt for authoritative values). Eight updates only qualify a short
numerical path. Longer matched stability testing remains separate, and no
positive result should be inferred from the original failed comparator.

A newer normalized 1,000-update local run completed all three coupled
random-init arms in 192.87 seconds with approximately 1.199 GiB peak allocation.
It used 4,027,000 valid target tokens per arm. Final development story bpb was
1.818783 / 1.865863 / 1.819111 for none/internal/output-only. All declared
stability/learning gates were reported passed in that run. An independent AI-assisted
15-checkpoint audit passed, verifying saved-state/Adam clocks, paired data/counters,
source/lineage and the declared gates. It performed no new GPU metric replay. This closes
a local training-reliability gap at this setting; it does not demonstrate a
semantic-integration or architecture advantage. There is still one seed,
12 development stories and eight world rows. Requested-key typed accuracy was
8/8, 7/8 and 8/8; raw 41-way accuracy 0/8, 2/8 and 0/8; raw query-byte accuracy
was 1/8 for all arms. Forbidden mass was zero on those evaluated finite paths.
The failed earlier output-only result remains in the table above. The newer
run is recorded in its own derived receipt rather than substituted into it.

The three-bit workflow reference retains original Lean sources and bounded
historical differential evidence against independent finite/Python semantics.
It proves conditional invariants for that declared mathematical system, not
Python refinement, truthful remote observations, the new RECORD kernel or a
language model. The standalone RECORD kernel and same-information adapter have
bounded CPU/adversarial checks and AI-assisted independent reviews; the code
and tests are rerunnable here. They are interface infrastructure, not a trained
strong external-control comparison.

## Receipt and source provenance

`results/*.derived.json` are explicitly selected, redacted derived summaries.
Each includes the SHA-256 of its original private report and describes the
transformation. They preserve the failed arm and evaluation denominators.
Original logs, host paths, corpus text, sample continuations and checkpoints are
not exported. A redacted summary has different bytes and a different digest;
it is never described as the immutable original report. `PUBLIC_MANIFEST.json`
binds the exported bytes separately. Original source/report digests are
provenance identifiers, not a claim that every historical artifact is shipped.

The normalized wrapper retains its three required bound-source files exactly.
`structured_decision.py` is a documented digest-only excerpt with both retained
function-byte hashes and its original full-source digest in `PROVENANCE.json`.
The public trainer ledger therefore differs from the original broader ledger.
Native-platform extraction manifests may also differ in path separators or
runtime metadata, even when TRAIN/dev corpus bytes match the expected hashes.
Reproduction means the explicit public recipe and checks; it does not mean
bit-for-bit regeneration of unavailable historical full reports/checkpoints.
