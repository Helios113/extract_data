#!/bin/bash
# Usage: ./submit_all.sh [config_dir]
# Submits one sbatch job per JSON config found under config_dir (default: configs/).
# Jobs are constrained to ruapehu or mahuia nodes with 1 GPU.

set -euo pipefail

CONFIG_DIR="${1:-configs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t CONFIGS < <(find "$CONFIG_DIR" -name "*.json" | sort)

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No JSON configs found under $CONFIG_DIR" >&2
  exit 1
fi

echo "Submitting ${#CONFIGS[@]} jobs from $CONFIG_DIR"

for CONFIG in "${CONFIGS[@]}"; do
  JOB_ID=$(sbatch \
    --job-name="extract_$(basename "$CONFIG" .json)" \
    --output="$SCRIPT_DIR/logs/extract_%j.out" \
    --error="$SCRIPT_DIR/logs/extract_%j.err" \
    --gres=gpu:1 \
    --time=12:00:00 \
    --cpus-per-task=8 \
    --ntasks=1 \
    --partition=hopper \
    --wrap="cd $SCRIPT_DIR && H5=\$(python3 -c \"import json; print(json.load(open('$CONFIG'))['output'])\") && uv run python run.py '$CONFIG' && uv run python upload.py push -y \"\$H5\"" \
    --parsable)
  echo "  submitted job $JOB_ID for $CONFIG"
done
