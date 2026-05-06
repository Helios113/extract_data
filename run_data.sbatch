#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --partition=hopper
set -euo pipefail


export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

uv run python scripts/run_analyze_id.py configs/results/analyze_id_llama_sing_small.json