# Amendment 1 to the monitorability preregistration

Written by the project lead on 2026-10-21 at about 12:31 PDT. The two seed-0 main runs (launched 12:25:47 PDT) were still in training, and no main-run endpoint had been computed or seen.

1. **Registration timing.** `PREREGISTRATION.md` was committed at 12:27:37 PDT, which is 110 s after the seed-0 main runs started and before any main-run result existed. Its final edit (12:25:56) recorded the launch time.
2. **Seeds.** The CPU study is extended from seed 0 only to **seeds 0, 1 and 2 per arm**. The configuration (`configs/cpu_pilot.json`), code, gates, endpoints and analysis are identical. Seeds 1 and 2 run after seed 0 finishes, using the same `run_main.sh` invocation pattern. Endpoints are reported per seed. They are also summarized across the 3 seeds as mean, range and paired structured − plain differences, and these summaries are **descriptive only**. n=3 is too small for the GPU-scale decision rule, which stays the confirmatory test.
3. **Gates per seed.** The G1–G3 validity gates apply to each run separately. A seed that fails G1 or G3 is reported and labelled as not interpretable, never dropped.

Nothing else changes.
