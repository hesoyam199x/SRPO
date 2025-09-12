NNODES=$HOST_NUM

NPROC_PER_NODE=$HOST_GPU_NUM
NODE_RANK=$INDEX
DATA_NUM_WORKERS=4
MODEL_TYPE="FLUX"
SP_SIZE=1

mkdir images


torchrun --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE \
    --node_rank $NODE_RANK \
    --rdzv_endpoint $CHIEF_IP:29601 \
    --rdzv_id 456 \
     fastvideo/SRPO.py \
    --seed 42 \
    --pretrained_model_name_or_path ./data/flux \
    --vae_model_path ./data/flux \
    --cache_dir data/.cache \
    --data_json_path  data/rl_embeddings/videos2caption2.json \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 2 \
    --max_train_steps 30000000 \
    --learning_rate 5e-6 \
    --mixed_precision bf16 \
    --checkpointing_steps 20 \
    --allow_tf32 \
    --train_guidence 1 \
    --output_dir ./output/hps/ \
    --h 720 \
    --w 720 \
    --t 1 \
    --sampling_steps 25 \
    --image_p 'srpohps' \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 0.1 \
    --weight_decay 0.0001 \
    --shift 3 \
    --ignore_last \
    --discount_inv 0.3 0.01 \
    --discount_pos 0.1 0.25 \
    --timestep_length 1000 \
    --groundtruth_ratio 0.9