REPO_ID=${1:-RoboDojo_sim_arx-x5_v30}

export HF_HOME=${HF_HOME:-/xspark-cache/shared}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/xspark-cache/shared/lerobot}

python util_scripts/compute_norm_stats_single.py \
  --action_mode delta \
  --chunk_size 50 \
  --repo_id ${REPO_ID}