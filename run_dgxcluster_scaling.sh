#!/bin/bash
#SBATCH --partition defq --nodes 31
#SBATCH --exclusive
#SBATCH --job-name=example-mn-sbatch-job
#SBATCH --gpus-per-node=8

CONTAINER=${HOME}/enroot_images/megds2.sqsh
#srun --nodes 2 --mpi=pmix --gpus-per-node 8 --container-image=${CONTAINER} --ntasks-per-node=1 nvidia-smi -L
#exit 0

export OMPI_MCA_coll_hcoll_enable=0
export UCX_TLS=rc
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_TOPO_FILE=/cm/shared/etc/ndv4-topo.xml
export NCCL_DEBUG=INFO
export NCCL_PROTO=LL,LL128,Simple
export NCCL_ALGO=Tree,Ring,CollnetDirect,CollnetChain,NVLS
export MELLANOX_VISIBLE_DEVICES=all
export PMIX_MCA_gds=hash
export PMIX_MCA_psec=native

export NHOSTS="${SLURM_NNODES}"
export NGPU_PER_HOST="${SLURM_GPUS_ON_NODE}"
export NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
export OMP_NUM_THREADS=1
export WORLD_SIZE=$NGPUS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export NCCL_DEBUG=warn

echo "PATH=$PATH" > .deepspeed_env
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> .deepspeed_env
echo "CPATH=$CPATH" >> .deepspeed_env
echo "TORCH_EXTENSIONS_DIR=$PWD/deepspeed" >> .deepspeed_env
echo "HF_HOME=$PWD/hfdata" >> .deepspeed_env


echo ${SLURM_GPUS_ON_NODE}

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
echo ${_basedir}

#cd $SCRIPT_PATH
echo $SCRIPT_PATH
echo $SLURM_NNODES

#CONTAINER=${HOME}/enroot_images/megds2.sqsh
#source /lustre/fs0/scratch/gdharuman/Megatron-DeepSpeed/deps/ezpz/src/ezpz/bin/savejobenv
srun --mpi=pmix --nodes $SLURM_NNODES --gpus-per-node 8 --ntasks-per-node=8 --container-workdir=${_basedir} --container-mounts="/lustre/fs0/scratch/gdharuman","/home/gdharuman" --container-image=${CONTAINER} python /lustre/fs0/scratch/gdharuman/Megatron-DeepSpeed/dpo_training.py \
   --use-flash-attn-v2 --fp16 --split 100,0,0 \
   --log-interval 1 --no-bias-gelu-fusion \
   --lr-decay-style cosine --no-bias-dropout-fusion \
   --no-masked-softmax-fusion --tokenizer-type Llama2Tokenizer \
   --no-gradient-accumulation-fusion --accumulate-allreduce-grads-in-fp32 \
   --use-checkpoint-opt_param-scheduler --lr 5e-6 --seq-length 512 \
   --save checkpoints/ds_stage2_nl6_hs4096_mb24_seq1024_gb48_pp1_tp2_fp16 \
   --load checkpoints/ds_stage2_nl6_hs4096_mb24_seq1024_gb48_pp1_tp2_fp16 \
   --num-layers 32 --hidden-size 4096 --train-iters 5000 --eval-iters 10 \
   --distributed-backend nccl --num-attention-heads 32 --save-interval 10 \
   --eval-interval 50000 --max-position-embeddings 4096 --micro-batch-size 12 \
   --data-file-list-p ALCF/data_textseq_p.txt \
   --data-file-list-u ALCF/data_textseq_u.txt \
   --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 \
   --num-key-value-heads 32 --data-cache-path ./index-cache \
   --ffn-hidden-size 11008 --tokenizer-model ALCF/tokenizer.model \
   --no-query-key-layer-scaling --use-rotary-position-embeddings \
   --untie-embeddings-and-output-weights --swiglu \
   --normalization rmsnorm --disable-bias-linear \
   --zero-stage=1 --deepspeed_config=ds_config-gpt.json \
   --no-pipeline-parallel --deepspeed --optimizer adamw
