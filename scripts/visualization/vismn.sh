# export WANDB_DISABLED=true
# export WANDB_BASE_URL="https://api.wandb.ai"
# export WANDB_MODE=online
export PYTHONPATH=`pwd`
num_gpus=8
num_nodes=$HOST_NUM
# num_nodes=1
node_rank=$INDEX
NPROC_PER_NODE=$HOST_GPU_NUM
NODE_RANK=$INDEX
DATA_NUM_WORKERS=4
MODEL_TYPE="FLUX"
SP_SIZE=1
rank=$node_rank
# /root/miconda init
conda init bash
conda activate SRPO;
# conda deactivate
torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
    --node_rank $node_rank \
    --rdzv_endpoint $CHIEF_IP:29502 \
    --rdzv_id 456 \
    ./scripts/visualization/vis_flux.py \
    --rank $rank \
