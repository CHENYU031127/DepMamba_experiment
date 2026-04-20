#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --dataset lmvd \
  --train True \
  --if_wandb False \
  --tqdm_able False
