# Setup deviations and execution notes

- The initial prescribed data-dependency installation in the original GPU environment failed before extraction or training because that interpreter has no pip module. The failure was preserved and reported on issue #1; no bootstrap, interpreter substitution or seed run was attempted before clarification.
- Owner READY setup amendment 1 (2026-10-21T18:49:19Z) explicitly authorized the existing data-only interpreter for extraction only. Its PyArrow version was checked as exactly 26.0.0. Extraction succeeded in this runner checkout; no corpus was copied from another checkout.
- The original GPU interpreter performed the standard-library extraction verification and all resumed routing tests, preflight, training, evaluation and analysis commands. The failed pip-install line was skipped exactly as amended. Existing CPU and GPU environments were preserved.
- Extraction verification returned `train_dev_sha256_verified: true` and `official_test_model_evaluated: false`. The GPU-environment package tests passed 8 tests; the optional SciPy reference test was skipped because SciPy is absent. No dependency was added to work around that optional skip.
- The routing preregistration, experiment code, configs, seeds, data splits and endpoints were unchanged. The execution checkout fast-forwarded from the original routing registration commit to the same branch's monitorability-package additions before launch; routing sources were identical. Exact execution HEAD is recorded in the generated run log.
- The prescribed training command's console output was also retained in an ignored local driver log. No training flags or hyperparameters were changed.

The explicitly requested local interpreter paths and full initial setup error are recorded in the issue reports; this committed note omits absolute host paths.
