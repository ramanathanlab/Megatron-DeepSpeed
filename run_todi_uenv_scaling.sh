#!/bin/bash
#SBATCH --job-name=megatron      # create a short name for your job
#SBATCH --nodes=2                # total number of nodes
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.log  # control where the stdout will be

ENROOT_LIBRARY_PATH=/capstor/scratch/cscs/fmohamed/enroot-lib
CONTAINER=/capstor/scratch/cscs/gdharuma/enroot/megds.sqsh
echo SLURM_NNODES="${SLURM_NNODES}"
echo SLURM_GPUS_ON_NODE="${SLURM_GPUS_ON_NODE}"
export NHOSTS="${SLURM_NNODES}"
export NGPU_PER_HOST="${SLURM_GPUS_ON_NODE}"
export NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
#export OMP_NUM_THREADS=1
export WORLD_SIZE=$NGPUS
#export RANK=$SLURM_PROCID
#export LOCAL_RANK=$SLURM_LOCALID
echo NHOSTS="${NHOSTS}"
echo NGPU_PER_HOST="${NGPU_PER_HOST}"
echo NGPUS="${NGPUS}"
echo WORLD_SIZE="${WORLD_SIZE}"
#echo RANK="${RANK}"
#echo LOCAL_RANK="${LOCAL_RANK}"
#srun --nodes $SLURM_NNODES --mpi=pmi2 --gpus-per-node 4 --container-image=${CONTAINER} --ntasks-per-node=1 nvidia-smi -L
#srun --nodes $SLURM_NNODES --mpi=pmi2 --gpus-per-node 4 --environment=megds --ntasks-per-node=1 nvidia-smi -L
#exit 0

#mkdir -p logs

# Initialization.
#set -x
#export MASTER_PORT=29500

#export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST-}" | head -n1)"
#export MASTER_ADDR=$(hostname)

#export CUDA_DEVICE_MAX_CONNECTIONS=1         # required by nanotron
# export either WANDB_API_KEY=<api key> or WANDB_MODE=offline

echo SLURM_GPUS_ON_NODE="${SLURM_GPUS_ON_NODE}"

if [ ! -z "${SLURM_JOB_ID}" ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(realpath $0)
fi

export _basedir="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
cd ${_basedir}
echo BASE_DIR="${_basedir}"
echo SCRIPT_PATH="${SCRIPT_PATH}"
echo RANK="${RANK}", LOCAL_RANK="${LOCAL_RANK}", MASTER_ADDR="${MASTER_ADDR}", MASTER_PORT="${MASTER_PORT}", WORLD_SIZE="${WORLD_SIZE}", UCX_NET_DEVICES="${UCX_NET_DEVICES}", NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}", NCCL_IB_HCA="${NCCL_IB_HCA}", NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY}", NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING}", SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING="${SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING}", UCX_VFS_ENABLE="${UCX_VFS_ENABLE}"

#export CUDA_VISIBLE_DEVICES=0,1,2,3
# Run main script.
#srun --mpi=pmi2 --environment=megds --master-addr=${MASTER_ADDR} --master-port=${MASTER_PORT} --nnodes=${SLURM_NNODES} --nproc-per-node=${SLURM_GPUS_PER_TASK} python /capstor/scratch/cscs/gdharuma/Megatron-DeepSpeed/dpo_training \
#srun --nodes $SLURM_NNODES --mpi=pmi2 --gpus-per-node 4 --ntasks-per-node=4 --environment=megds --ntasks-per-node=1 python dpo_training.py \
#srun --nodes $SLURM_NNODES --mpi=pmi2 --gpus-per-node 4 --environment=megds --ntasks-per-node=1 python dpo_training.py \
#srun -l --cpu-bind=rank_ldom --nodes $SLURM_NNODES --mpi=pmi2 --environment=megds --ntasks-per-node=4 \
#     /capstor/scratch/cscs/boeschf/images/launch_wrapper python dpo_training.py \

export FI_CXI_DISABLE_HOST_REGISTER="1"
export FI_MR_CACHE_MONITOR="userfaultfd"

srun -l \
    --cpu-bind=rank_ldom \
    --nodes $SLURM_NNODES \
    --ntasks-per-node=4 \
    --uenv="/capstor/scratch/cscs/boeschf/images/pytorch-2.3.1-megatron_deepspeed_0.squashfs" \
    bash -c "
    uenv view default
    . ./.venv/bin/activate
    python dpo_training.py \
    --use-flash-attn-v2 --fp16 --split 100,0,0 \
    --log-interval 1 --no-bias-gelu-fusion \
    --lr-decay-style cosine --no-bias-dropout-fusion \
    --no-masked-softmax-fusion --tokenizer-type Llama2Tokenizer \
    --no-gradient-accumulation-fusion --accumulate-allreduce-grads-in-fp32 \
    --use-checkpoint-opt_param-scheduler --lr 5e-6 --seq-length 1024 \
    --save checkpoints/ds_stage2_nl6_hs4096_mb24_seq1024_gb48_pp1_tp2_fp16 \
    --load checkpoints/ds_stage2_nl6_hs4096_mb24_seq1024_gb48_pp1_tp2_fp16 \
    --num-layers 32 --hidden-size 4096 --train-iters 100 --eval-iters 10 \
    --distributed-backend nccl --num-attention-heads 32 --save-interval 2000 \
    --eval-interval 50000 --max-position-embeddings 1024 --micro-batch-size 2 \
    --data-file-list-p /capstor/scratch/cscs/gdharuma/Megatron-DeepSpeed/ALCF/data_p_small.txt \
    --data-file-list-u /capstor/scratch/cscs/gdharuma/Megatron-DeepSpeed/ALCF/data_u_small.txt \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --num-key-value-heads 32 --data-cache-path ./index-cache \
    --ffn-hidden-size 11008 --tokenizer-model /capstor/scratch/cscs/gdharuma/Megatron-DeepSpeed/ALCF/tokenizer.model \
    --no-query-key-layer-scaling --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights --swiglu \
    --normalization rmsnorm --disable-bias-linear \
    --zero-stage=1 --deepspeed_config=ds_config-gpt_nooffload.json \
    --no-pipeline-parallel --deepspeed --optimizer adamw
