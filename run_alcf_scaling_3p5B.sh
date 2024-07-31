#!/bin/bash --login

cd "${PBS_O_WORKDIR}" || exit

NOW="$(date "+%Y-%m-%d-%H%M%S")"

source "${PBS_O_WORKDIR}/deps/ezpz/src/ezpz/bin/utils.sh" || exit
ezpz_setup_python || exit
ezpz_setup_alcf "$@" || exit

TP="${TP:-1}"

CHECKPOINT_DIR="checkpoints/ds_stage2_nl16_hs4096_mb2_seq512_pp1_tp${TP}_bf16"
mkdir -p "${CHECKPOINT_DIR}"

export WORLD_SIZE="${NGPUS}"
run_cmd="${DIST_LAUNCH} python3 dpo_training.py \
 --bf16 \
 --split 100,0,0 \
 --log-interval 1  \
 --no-bias-gelu-fusion  \
 --lr-decay-style cosine \
 --no-bias-dropout-fusion \
 --no-masked-softmax-fusion \
 --tokenizer-type Llama2Tokenizer \
 --no-gradient-accumulation-fusion \
 --accumulate-allreduce-grads-in-fp32 \
 --use-checkpoint-opt_param-scheduler \
 --lr 5e-6 \
 --seq-length 512 \
 --save ${CHECKPOINT_DIR} \
 --load ${CHECKPOINT_DIR} \
 --num-layers 16 \
 --hidden-size 4096 \
 --train-iters 5000 \
 --eval-iters 10 \
 --distributed-backend ccl \
 --num-attention-heads 32 \
 --save-interval 5000 \
 --eval-interval 50000 \
 --max-position-embeddings 4096 \
 --micro-batch-size 2 \
 --data-file-list-p ALCF/data_p.txt \
 --data-file-list-u ALCF/data_u.txt \
 --tensor-model-parallel-size ${TP} \
 --pipeline-model-parallel-size 1 \
 --num-key-value-heads 32 \
 --data-cache-path ./index-cache \
 --ffn-hidden-size 11008 \
 --tokenizer-model ALCF/tokenizer.model \
 --no-query-key-layer-scaling \
 --use-rotary-position-embeddings \
 --untie-embeddings-and-output-weights \
 --swiglu \
 --normalization rmsnorm \
 --disable-bias-linear \
 --zero-stage=2 \
 --deepspeed_config=ds_config-gpt.json \
 --no-pipeline-parallel \
 --deepspeed \
 --optimizer adamw"


LOGFILE="dpo_training_${NOW}.log"

echo "Writing to ${LOGFILE}"
echo "CHECKPOINT_DIR: ${CHECKPOINT_DIR}"
echo "${run_cmd}" > "${LOGFILE}"
eval "${run_cmd}" |& tee "${LOGFILE}"
