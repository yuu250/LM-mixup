#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/root/autodl-tmp/EasyR1/ckpt/Qwen2.5-1.5B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen_test
