#!/bin/bash --login
#PBS -l walltime=0:30:00
#PBS -A datascience
#PBS -q debug
#PBS -l select=1
#PBS -l filesystems=eagle:grand:home
cd ${PBS_O_WORKDIR}
export PPN=4
# export MD=${HOME}/GB-Megatron-DeepSpeed
export MD="/eagle/FoundEpidem/foremans/Megatron-DeepSpeed"
export PYTHONPATH=$MD:$PYTHONPATH
source /eagle/argonne_tpc/soft/conda.sh
export TRITON_CACHE_DIR=/tmp/.cache/
export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)

APRUN_PMI=pmix aprun -n $((PBS_JOBSIZE*PPN)) -N $PPN --cc depth -d 16 /eagle/argonne_tpc/soft/local_rank.sh python dpo_training.py --use-flash-attn-v2 --fp16 --num-workers 0 --split 100,0,0 --log-interval 1 --no-bias-gelu-fusion --lr-decay-style cosine --no-bias-dropout-fusion --no-masked-softmax-fusion --tokenizer-type Llama2Tokenizer --no-gradient-accumulation-fusion --accumulate-allreduce-grads-in-fp32 --use-checkpoint-opt_param-scheduler --lr 0.0003 --seq-length 1024 --save checkpoints/ds_stage2_nl6_hs4096_mb24_seq1024_gb48_pp1_tp2_fp16 --load checkpoints/ds_stage2_nl6_hs4096_mb24_seq1024_gb48_pp1_tp2_fp16 --num-layers 16 --hidden-size 4096 --train-iters 30 --eval-iters 10 --distributed-backend nccl --num-attention-heads 32 --save-interval 200 --eval-interval 50000 --max-position-embeddings 1024 --micro-batch-size 4 --data-file-list-p ALCF/data_textseq_proteingym_indels_file_list_p.txt --data-file-list-u ALCF/data_textseq_proteingym_indels_file_list_u.txt --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --num-key-value-heads 32 --data-cache-path ./index-cache --ffn-hidden-size 11008 --tokenizer-model /eagle/datasets/dolma/utils/tokenizer.model --no-query-key-layer-scaling --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear --deepspeed-activation-checkpointing --zero-stage=2 --deepspeed_config=ds_config-gpt.json --no-pipeline-parallel --deepspeed --checkpoint-activations --checkpoint-num-layers 1 --optimizer adamw
