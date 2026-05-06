uv run python run.py configs/custom/custom_qwen_dims_2layer_Rlarge_noise0p2_d30_n1024_s64.json --no-upload

uv run python run.py configs/custom/custom_gpt.json --no-upload



uv run python scripts/analyze_id.py \
  out/custom/pythia.h5 \
  results/id_est/custom/qwen_lr.csv \
  --ess-k 25 --no-entropy --pos 10