"""Cross-platform equivalent of run_main.sh (orchestration only; no analysis logic).

Trains and evaluates both arms for each seed (arms interleaved per seed), copies
train metrics into results/<config name>/, then runs aggregate.py.

Usage (from experiments/monitorability):
    python run_main.py configs/gpu_scale.json cuda 8 0 1 2 3 4
"""
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd, log_path):
    print(" ".join(cmd), flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
    if code != 0:
        raise SystemExit(f"command failed with exit code {code}: {' '.join(cmd)}")


def main():
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    cfg, device, threads, seeds = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    name = json.loads((HERE / cfg).read_text(encoding="utf-8"))["name"]
    (HERE / "results" / name).mkdir(parents=True, exist_ok=True)
    (HERE / "logs").mkdir(exist_ok=True)
    py = sys.executable
    for seed in seeds:
        for arm in ("plain", "structured"):
            print(f"=== {name} {arm} seed {seed} {datetime.now().isoformat(timespec='seconds')}", flush=True)
            run([py, "train.py", "--arm", arm, "--seed", seed, "--config", cfg,
                 "--device", device, "--threads", threads],
                HERE / "logs" / f"{name}_{arm}_s{seed}_train.log")
            shutil.copyfile(HERE / "checkpoints" / name / f"{arm}_s{seed}" / "train_metrics.json",
                            HERE / "results" / name / f"{arm}_s{seed}_train_metrics.json")
            run([py, "evaluate.py", "--ckpt", str(Path("checkpoints") / name / f"{arm}_s{seed}" / "model.pt"),
                 "--device", device, "--threads", threads,
                 "--out", str(Path("results") / name / f"{arm}_s{seed}.json")],
                HERE / "logs" / f"{name}_{arm}_s{seed}_eval.log")
    run([py, "aggregate.py", "--config_name", name], HERE / "logs" / f"{name}_aggregate.log")
    print(f"=== done {datetime.now().isoformat(timespec='seconds')}", flush=True)


if __name__ == "__main__":
    main()
