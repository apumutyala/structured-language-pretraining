#!/usr/bin/env bash
# Train + evaluate both arms for a list of seeds, then aggregate.
# Usage: bash run_main.sh <config.json> <device> <threads> <seed> [<seed> ...]
# Example (CPU pilot):  bash run_main.sh configs/cpu_pilot.json cpu 2 0 1 2 3 4
# Example (GPU):        bash run_main.sh configs/gpu_scale.json cuda 8 0 1 2 3 4 5 6 7 8 9
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
CFG="$1"; DEV="$2"; THR="$3"; shift 3
NAME="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['name'])" "$CFG")"
mkdir -p "results/$NAME" logs
for SEED in "$@"; do
  for ARM in plain structured; do
    echo "=== $NAME $ARM seed $SEED $(date -Is)"
    python train.py --arm "$ARM" --seed "$SEED" --config "$CFG" --device "$DEV" --threads "$THR" \
      2>&1 | tee "logs/${NAME}_${ARM}_s${SEED}_train.log"
    cp "checkpoints/$NAME/${ARM}_s${SEED}/train_metrics.json" "results/$NAME/${ARM}_s${SEED}_train_metrics.json"
    python evaluate.py --ckpt "checkpoints/$NAME/${ARM}_s${SEED}/model.pt" --device "$DEV" --threads "$THR" \
      --out "results/$NAME/${ARM}_s${SEED}.json" 2>&1 | tee "logs/${NAME}_${ARM}_s${SEED}_eval.log"
  done
done
python aggregate.py --config_name "$NAME"
echo "=== done $(date -Is)"
