#!/bin/bash --login

NOW="$(date "+%Y-%m-%d-%H%M%S")"

cd "${PBS_O_WORKDIR}" || exit
source "${PBS_O_WORKDIR}/deps/ezpz/src/ezpz/bin/utils.sh" || exit
ezpz_setup_python || exit
ezpz_setup_alcf "$@" || exit

PP=1
TP=1
NUM_LAYERS=16
NUM_KV_HEADS=32
NUM_ATTENTION_HEADS=32
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=11008
MAX_POSITION_EMBEDDINGS=4096
ZERO_STAGE=1
MICRO_BATCH=8
SEQ_LEN=512

CHECKPOINT_DIR="checkpoints/ds_stage${ZERO_STAGE}_nl${NUM_LAYERS}_hs${HIDDEN_SIZE}_mb${MICRO_BATCH}_seq${SEQ_LEN}_pp${PP}_tp${TP}_bf16"
mkdir -p "${CHECKPOINT_DIR}"

export WORLD_SIZE="${NGPUS}"
run_cmd="${DIST_LAUNCH} python3 dpo_training.py \
 --seq-length ${SEQ_LEN} \
 --save ${CHECKPOINT_DIR} \
 --load ${CHECKPOINT_DIR} \
 --num-layers ${NUM_LAYERS} \
 --hidden-size ${HIDDEN_SIZE} \
 --micro-batch-size ${MICRO_BATCH} \
 --tensor-model-parallel-size ${TP} \
 --pipeline-model-parallel-size ${PP} \
 --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
 --num-key-value-heads ${NUM_KV_HEADS} \
 --num-attention-heads ${NUM_ATTENTION_HEADS} \
 --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
 --zero-stage=${ZERO_STAGE} \
 --attention-dropout 0 \
 --hidden-dropout 0 \
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
 --train-iters 5000 \
 --eval-iters 10 \
 --distributed-backend ccl \
 --save-interval 5000 \
 --eval-interval 50000 \
 --data-file-list-p ALCF/data_p.txt \
 --data-file-list-u ALCF/data_u.txt \
 --data-cache-path ./index-cache \
 --tokenizer-model ALCF/tokenizer.model \
 --no-query-key-layer-scaling \
 --use-rotary-position-embeddings \
 --untie-embeddings-and-output-weights \
 --swiglu \
 --normalization rmsnorm \
 --disable-bias-linear \
 --deepspeed_config=ds_config_3p5B.json \
 --no-pipeline-parallel \
 --deepspeed \
 --optimizer adamw"

LOGFILE="dpo_training_3p5B_${NOW}.log"

echo "Writing to ${LOGFILE}"
echo "CHECKPOINT_DIR: ${CHECKPOINT_DIR}"
echo "${run_cmd}" >"${LOGFILE}"
eval "${run_cmd}" |& tee -a "${LOGFILE}"
