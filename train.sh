
MODEL="pretrained_models/Qwen3.6-27B"

nproc_per_node=4
NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
IMAGE_MAX_TOKEN_NUM=2048 \
swift sft \
    --model $MODEL \
    --output_dir output/qwen3.6-27b \
    --add_version False \
    --acc_strategy seq \
    --early_stop_interval 3 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --freeze_vit true \
    --freeze_aligner false \
    --target_modules all-linear \
    --load_from_cache_file true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --gradient_accumulation_steps 4 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --max_length 8192 \
    --warmup_ratio 0.1 \
    --dataset data/MolParser-7M/sft_real/train.json \
    --val_dataset data/MolParser-7M/sft_real/val.json \
    --dataloader_num_workers 8
