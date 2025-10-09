#!/bin/bash

# This script fine-tunes the bge reranker model

# sleep 5400

torchrun --nproc_per_node 1 -m FlagEmbedding.finetune.reranker.decoder_only.base \
    --model_name_or_path /path/to/model \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj \
    --use_flash_attn False \
    --save_merged_lora_model False\
    --cache_dir ./bge_cache/model \
    --cache_path ./bge_cache/data \
    --train_data /path/to/dataset \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 1024 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --query_instruction_for_rerank 'A: ' \
    --query_instruction_format '{}{}' \
    --passage_instruction_for_rerank 'B: ' \
    --passage_instruction_format '{}{}' \
    --output_dir /path/to/output \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --bf16 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --weight_decay 0.1 \
    --logging_steps 1 \
    --save_steps 100 \
    --max_grad_norm 1.0
