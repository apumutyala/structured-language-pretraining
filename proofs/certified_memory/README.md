# Three-bit workflow reference

These exact Lean 4.8.0 sources model finite possible-world workflow memory,
policy grants and attempt budgets. `SoundMemory.lean` supplies generic
conditional inclusion/action/grounded-assertion results. `WorkflowKernel.lean`
instantiates three completion bits and executable update/release predicates;
`WorkflowKernelCli.lean` is a trusted-host finite test adapter.

Run `python tools/build_lean.py --lean lean` from the repository root with Lean
4.8.0 installed. See docs/reproduction.md for runtime and environment overrides.
No binaries are committed. Theorems assume the declared prior/transition and
truthful-or-contradictory observation condition; false but consistent receipts
are excluded. This is not transport authentication, Python refinement, a
language guarantee or a proof of the newer supported_state RECORD code.
