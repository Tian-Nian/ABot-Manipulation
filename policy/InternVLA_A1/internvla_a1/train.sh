repo_id=${1:-RoboDojo_sim_arx-x5_v30}
GPU_ID=${2:-0,1,2,3,4,5,6,7}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export HF_HOME=${HF_HOME:-/xspark-cache/shared}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/xspark-cache/shared/lerobot}
export COSMOS_PATH="/mnt/xspark-data/xspark_shared/model_weights/Cosmos-Tokenizer-CI8x8"
export QWEN3_2B_PATH="/mnt/xspark-data/xspark_shared/model_weights/Qwen3-VL-2B-Instruct"
export PROC_PER_NODE=${PROC_PER_NODE:-8}

bash launch/internvla_a1_3b_finetune.sh ${repo_id} delta true