#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="meta-llama/Llama-3.1-8B"

JSONLS=(
  "train_jsonl/full_low_10k.jsonl"
  "train_jsonl/high7_low3_10k.jsonl"
  "train_jsonl/high5_low5_10k.jsonl"
  "train_jsonl/high3_low7_10k.jsonl"
  "train_jsonl/mixup7_ori3_1_5b_10k.jsonl"
  "train_jsonl/mixup5_ori5_1_5b_10k.jsonl" 
  "train_jsonl/mixup3_ori7_1_5b_10k.jsonl"
)

LORA_RANK=64
LORA_ALPHA=16
LORA_DROPOUT=0.1
LR=1e-4
EPOCHS=5
MAX_LEN=2048
WARMUP=0.03
SEED=42
PER_DEVICE=1
GRAD_ACCUM=64
GPUS=("0" "1")
USE_VLLM=true
SHUTDOWN_AFTER=true

mkdir -p sft_output merged

slugify() {
  local p="$1"
  local base="${p##*/}"; base="${base%.*}"
  echo "$base" | sed -E 's/[^A-Za-z0-9]+/_/g'
}

port_seq() { echo $((29501 + $1)); }

declare -a MODELS_TO_EVAL=()
declare -a TAGS_TO_EVAL=()

idx=0
for TRAIN_JSONL in "${JSONLS[@]}"; do
  NAME="$(slugify "$TRAIN_JSONL")"
  OUT_SFT="sft_output/lora_llama3_${NAME}"   
  OUT_MERGED="merged/lora_llama3_${NAME}"    
  MASTER_PORT="$(port_seq "$idx")"

  echo "========== [TRAIN] ${NAME} | GPUs={${GPUS[*]}} | master_port=${MASTER_PORT} =========="

  CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")" torchrun \
    --nproc_per_node="${#GPUS[@]}" \
    --master_port="${MASTER_PORT}" \
    finetune.py \
      --model_name_or_path "${BASE_MODEL}" \
      --train_file "${TRAIN_JSONLS:-$TRAIN_JSONL}" \
      --output_dir "${OUT_SFT}" \
      --use_lora \
      --lora_rank "${LORA_RANK}" \
      --lora_alpha "${LORA_ALPHA}" \
      --lora_dropout "${LORA_DROPOUT}" \
      --per_device_train_batch_size "${PER_DEVICE}" \
      --gradient_accumulation_steps "${GRAD_ACCUM}" \
      --learning_rate "${LR}" \
      --num_train_epochs "${EPOCHS}" \
      --max_seq_length "${MAX_LEN}" \
      --warmup_ratio "${WARMUP}" \
      --seed "${SEED}" \
      --logging_steps 50

  python merge_lora.py \
    --lora_model_name_or_path "${OUT_SFT}" \
    --base_model_name_or_path "${BASE_MODEL}" \
    --output_dir "${OUT_MERGED}" \
    --save_tokenizer \
    --use_fast_tokenizer

  rm -rf "${OUT_SFT}"

  MODELS_TO_EVAL+=("${OUT_MERGED}")
  TAGS_TO_EVAL+=("lora_llama3_${NAME}")      

  idx=$((idx + 1))
done

echo "========== [EVAL] starting evaluations =========="
for i in "${!MODELS_TO_EVAL[@]}"; do
  MODEL="${MODELS_TO_EVAL[$i]}"
  TAG="${TAGS_TO_EVAL[$i]}"
  GPU="${GPUS[$(( i % ${#GPUS[@]} ))]}"

  echo "[EVAL] ${TAG} -> GPU ${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" bash run_eval.sh \
    --model "${MODEL}" \
    --tag   "${TAG}" \
    --gpu   0 \
    --use_vllm "${USE_VLLM}" &
done

wait
echo "All evaluations finished."

if [ "${SHUTDOWN_AFTER}" = true ]; then
  /usr/bin/shutdown
fi