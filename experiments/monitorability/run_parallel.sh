#!/usr/bin/env bash
# CPU run pattern actually used for the cpu_pilot main runs (2-core machine):
# for each seed, train both arms in parallel with 1 thread each, then evaluate both in parallel.
# Usage: bash run_parallel.sh <config.json> <seed> [<seed> ...]
#        SKIP_TRAIN=1 bash run_parallel.sh ...   # evaluate existing checkpoints only
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
CFG="$1"; shift
NAME="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['name'])" "$CFG")"
mkdir -p "results/$NAME" logs
for SEED in "$@"; do
  if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    echo "=== train $NAME seed $SEED $(date -Is)"
    for ARM in plain structured; do
      python train.py --arm "$ARM" --seed "$SEED" --config "$CFG" --threads 1 \
        > "logs/${NAME}_${ARM}_s${SEED}_train.log" 2>&1 &
    done
    wait
  fi
  echo "=== eval $NAME seed $SEED $(date -Is)"
  for ARM in plain structured; do
    cp "checkpoints/$NAME/${ARM}_s${SEED}/train_metrics.json" "results/$NAME/${ARM}_s${SEED}_train_metrics.json"
    python evaluate.py --ckpt "checkpoints/$NAME/${ARM}_s${SEED}/model.pt" --threads 1 \
      --out "results/$NAME/${ARM}_s${SEED}.json" > "logs/${NAME}_${ARM}_s${SEED}_eval.log" 2>&1 &
  done
  wait
  tail -n 1 logs/${NAME}_*_s${SEED}_eval.log
done
python aggregate.py --config_name "$NAME"
echo "=== done $(date -Is)"
