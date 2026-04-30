#!/bin/bash
#SBATCH --job-name=extract_data
#SBATCH --output=/scratch/%u/htr/extract_data/logs/extract_%j.out
#SBATCH --error=/scratch/%u/htr/extract_data/logs/extract_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

set -x
PROJECT_DIR=/scratch/vrabel/htr/extract_data
mkdir -p "$PROJECT_DIR/logs"

# Pick a config (override by exporting CONFIG before sbatch, or pass as $1).
CONFIG="${1:-${CONFIG:-configs/gpt2.json}}"

# Per-run output dir: runs/<jobid>_<configname>/
CONFIG_NAME="$(basename "$CONFIG" .json)"
RUN_REL="runs/${SLURM_JOB_ID}_${CONFIG_NAME}"
RUN_DIR="$PROJECT_DIR/$RUN_REL"
mkdir -p "$RUN_DIR"
cp "$PROJECT_DIR/$CONFIG" "$RUN_DIR/config.json"

# Optional HF token for gated models (Llama-3.x). Skip entirely if you don't
# touch a gated model. Sourced from a file so `set -x` never echoes the value.
# if [ -f "$PROJECT_DIR/.hf_token" ]; then
#   set +x
#   export HF_TOKEN="$(cat "$PROJECT_DIR/.hf_token")"
#   set -x
# fi

enroot start --root \
  --mount=$PROJECT_DIR:/workspace \
  ps_opt \
  bash -lc "
    set -eux
    cd /workspace
    export WANDB_ENTITY=h_tr
    uv run python run.py '$CONFIG' '$RUN_REL/run.h5'
  "
