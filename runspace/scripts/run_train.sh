#!/usr/bin/env bash
cd /data/bfys/shenzh/ML/cs336/assignment1-basics/runspace/scripts || exit 1

# Activate conda 
source /data/bfys/shenzh/software/miniconda/etc/profile.d/conda.sh || exit 1
conda activate ML || exit 1

export WANDB_API_KEY=3c7f187c3f4b555f875198c659e26b021cb9eb9c
# Execute training
uv run python train_lm.py --config config_examples/TinyStory.json --use-wandb