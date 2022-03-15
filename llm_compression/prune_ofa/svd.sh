#!/bin/bash
# Apache v2 license
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Student Pruning

# Notes:
# Auto mixed precision can be used by adding --fp16
# Distributed training can be used with the torch.distributed.lauch app

OUTPUT_DIR=/home/$USER/llm-compression/model_svd_pruned
DATA_DIR=/mnt/data
DATA_CACHE_DIR=$DATA_DIR/cache
HUGGINGFACE_MODEL=rbhushan/distilgpt2-finetuned-wikitext2

export TRANSFORMERS_CACHE=$DATA_DIR/transformers
export HF_HOME=$DATA_DIR/hf

python -m pdb run_clm.py \
    --svd \
    --model_name_or_path  /home/$USER/llm-compression/model_pruned\
    --dataset_name wikipedia \
    --dataset_config_name 20200501.en \
    --data_process_type segment_pair_nsp \
    --dataset_cache_dir $DATA_CACHE_DIR \
    --do_train \
    --max_steps 500 \
    --warmup_ratio 0.01 \
    --weight_decay 0.01 \
    --distill \
    --teacher_name_or_path $HUGGINGFACE_MODEL \
    --cross_entropy_alpha 0.5 \
    --knowledge_distillation_alpha 0.5 \
    --temperature 2.0 \
    --output_dir $OUTPUT_DIR
