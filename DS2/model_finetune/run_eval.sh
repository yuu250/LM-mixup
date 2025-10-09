#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
# bash run_eval.sh \
# --model Qwen/Qwen2.5-1.5B-Instruct \
# --tag   qwen2.5_1.5b_baseline
#
# Optional:
#   --tokenizer /path/to/tokenizer   (default = same as --model)
#   --data_root raw_data/eval        (default below)
#   --use_vllm true|false            (default false)
#   --gpu 0                          (default 0)

MODEL_PATH=""
TOKENIZER_PATH=""
RUN_TAG="run_$(date +%Y%m%d_%H%M%S)"
DATA_ROOT="DS2/eval"
USE_VLLM="false"
GPU_ID="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--base_model_name_or_path)
      MODEL_PATH="$2"; shift 2;;
    --tokenizer|--tokenizer_name_or_path)
      TOKENIZER_PATH="$2"; shift 2;;
    --tag|--output_dir)
      RUN_TAG="$2"; shift 2;;
    --data_root)
      DATA_ROOT="$2"; shift 2;;
    --use_vllm)
      USE_VLLM="$2"; shift 2;;
    --gpu)
      GPU_ID="$2"; shift 2;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "${MODEL_PATH}" ]]; then
  echo "ERROR: Please set --model <MODEL_PATH>"
  exit 1
fi
if [[ -z "${TOKENIZER_PATH}" ]]; then
  TOKENIZER_PATH="${MODEL_PATH}"
fi

echo "========== EVAL CONFIG =========="
echo "GPU_ID            : ${GPU_ID}"
echo "MODEL_PATH        : ${MODEL_PATH}"
echo "TOKENIZER_PATH    : ${TOKENIZER_PATH}"
echo "RUN_TAG           : ${RUN_TAG}"
echo "DATA_ROOT         : ${DATA_ROOT}"
echo "USE_VLLM          : ${USE_VLLM}"
echo "================================="

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

RESULT_ROOT="results/${RUN_TAG}"
mkdir -p "${RESULT_ROOT}"

maybe_vllm() {
  if [[ "${USE_VLLM}" == "true" ]]; then
    echo "--use_vllm"
  else
    echo ""
  fi
}

####################################
# 1) MMLU —— factual knowledge
####################################
echo ""
echo ">>> [1/5] Running MMLU..."
SAVE_DIR="${RESULT_ROOT}/mmlu"
mkdir -p "${SAVE_DIR}"

python -m eval.mmlu.run_eval \
  --ntrain 0 \
  --data_dir "${DATA_ROOT}/mmlu" \
  --save_dir "${SAVE_DIR}" \
  --model_name_or_path "${MODEL_PATH}" \
  --tokenizer_name_or_path "${TOKENIZER_PATH}" \
  --eval_batch_size 8

echo ">>> MMLU done. Results at: ${SAVE_DIR}"

####################################
# 2) GSM8K —— math reasoning
####################################
echo ""
echo ">>> [2/5] Running GSM8K..."
SAVE_DIR="${RESULT_ROOT}/gsm"
mkdir -p "${SAVE_DIR}"

python -m eval.gsm.run_eval \
  --data_dir "${DATA_ROOT}/gsm/" \
  --max_num_examples 200 \
  --save_dir "${SAVE_DIR}" \
  --model_name_or_path "${MODEL_PATH}" \
  --tokenizer_name_or_path "${TOKENIZER_PATH}" \
  --n_shot 8 \
  $(maybe_vllm)

echo ">>> GSM8K done. Results at: ${SAVE_DIR}"

####################################
# 3) BBH —— big-bench hard
####################################
echo ""
echo ">>> [3/5] Running BBH..."
SAVE_DIR="${RESULT_ROOT}/bbh"
mkdir -p "${SAVE_DIR}"

python -m eval.bbh.run_eval \
  --data_dir "${DATA_ROOT}/bbh" \
  --save_dir "${SAVE_DIR}" \
  --model_name_or_path "${MODEL_PATH}" \
  --tokenizer_name_or_path "${TOKENIZER_PATH}" \
  --max_num_examples_per_task 40 \
  $(maybe_vllm)

echo ">>> BBH done. Results at: ${SAVE_DIR}"

####################################
# 4) TruthfulQA —— truthfulness
####################################
echo ""
echo ">>> [4/5] Running TruthfulQA..."
SAVE_DIR="${RESULT_ROOT}/truthfulqa"
mkdir -p "${SAVE_DIR}"

python -m eval.truthfulqa.run_eval \
  --data_dir "${DATA_ROOT}/truthfulqa" \
  --save_dir "${SAVE_DIR}" \
  --model_name_or_path "${MODEL_PATH}" \
  --tokenizer_name_or_path "${TOKENIZER_PATH}" \
  --metrics truth info mc \
  --preset qa \
  --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
  --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
  --eval_batch_size 20 \
  --load_in_8bit

echo ">>> TruthfulQA done. Results at: ${SAVE_DIR}"

####################################
# 5) TyDiQA —— multilinguality
####################################
echo ""
echo ">>> [5/5] Running TyDiQA..."
SAVE_DIR="${RESULT_ROOT}/tydiqa"
mkdir -p "${SAVE_DIR}"

python -m eval.tydiqa.run_eval \
  --data_dir "${DATA_ROOT}/tydiqa/" \
  --n_shot 1 \
  --max_num_examples_per_lang 100 \
  --max_context_length 512 \
  --save_dir "${SAVE_DIR}" \
  --model_name_or_path "${MODEL_PATH}" \
  --tokenizer_name_or_path "${TOKENIZER_PATH}" \
  --eval_batch_size 20 \
  --load_in_8bit

echo ">>> TyDiQA done. Results at: ${SAVE_DIR}"

echo ""
echo "All evals finished."
echo "Result root: ${RESULT_ROOT}"