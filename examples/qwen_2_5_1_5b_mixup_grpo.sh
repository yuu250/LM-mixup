#!/bin/bash
set -x

export PYTHONUNBUFFERED=1

BASE_DIR=sft/sft_output/qwen_2_5_1_5b_cold_start

MODEL_PATH=$(ls -d ${BASE_DIR}/checkpoint-* | sort -V | tail -1)

echo "Using MODEL_PATH=${MODEL_PATH}"

python3 -m verl.trainer.main \
    config=examples/config_mixup_bayes.yaml \
    data.max_response_length=2400 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen_mixup_1_5b_grpo