#!/bin/bash
set -x

export PYTHONUNBUFFERED=1

#MODEL_PATH=/root/autodl-tmp/EasyR1/sft/sft_output/qwe1_5_cold_start/checkpoint-20000
#MODEL_PATH=/root/autodl-tmp/EasyR1/ckpt/Qwen2.5-1.5B-Instruct
MODEL_PATH=/root/autodl-tmp/EasyR1/sft/sft_output/qwen_2_5_7b_cold_start/checkpoint-35069

python3 -m verl.trainer.main \
    config=examples/config_mixup_bayes.yaml \
    data.max_response_length=2400 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen_mixup_7b_all_grpo
