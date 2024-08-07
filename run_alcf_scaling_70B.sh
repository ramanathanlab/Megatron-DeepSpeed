#!/bin/bash --login

install_deepspeed() {
  _deepspeed_dir="${PBS_O_WORKDIR}/deps/DeepSpeed"
  if [[ ! -d "${_deepspeed_dir}" ]]; then
    mkdir $(dirname "${_deepspeed_dir}")
    git clone https://github.com/microsoft/DeepSpeed "${_deepspeed_dir}"
    cd "${_deepspeed_dir}"
    bash install.sh |& tee install.log
    cd -
  fi
}

export MODEL_SIZE="70B"

export PP=1
export TP=4
export SEQ_LEN=512
export NUM_LAYERS=80
export NUM_KV_HEADS=8
export HIDDEN_SIZE=8192
export FFN_HIDDEN_SIZE=28672
export NUM_ATTENTION_HEADS=64
export MAX_POSITION_EMBEDDINGS=1024

export DS_CONFIG="ds_configs/${MODEL_SIZE}.json"
export ZERO_STAGE=$(cat "${DS_CONFIG}" | grep "stage" | sed "s/\,.*//g" | awk '{print $NF}')
export MICRO_BATCH=$(cat "${DS_CONFIG}" | grep "micro_batch" | sed "s/\,.*//g" | awk '{print $NF}')

export NOW="$(date "+%Y-%m-%d-%H%M%S")"
export LOGFILE="logs/dpo_training_${MODEL_SIZE}_${NOW}.log"
export CHECKPOINT_DIR="checkpoints/${MODEL_SIZE}_ds_stage${ZERO_STAGE}_nl${NUM_LAYERS}_hs${HIDDEN_SIZE}_mb${MICRO_BATCH}_seq${SEQ_LEN}_pp${PP}_tp${TP}_bf16"
mkdir -p "${CHECKPOINT_DIR}"

cd "${PBS_O_WORKDIR}" || exit

# [ezpz] ########################################################
_ezpz_dir="${PBS_O_WORKDIR}/deps/ezpz"
if [[ ! -d "${_ezpz_dir}" ]]; then
  mkdir $(dirname "${_ezpz_dir}")
  git clone https://github.com/saforem2/ezpz "${_ezpz_dir}"
fi

source "${_ezpz_dir}/src/ezpz/bin/utils.sh" || exit
ezpz_setup_python |& tee --append "${LOGFILE}" || exit
ezpz_setup_alcf "$@" |& tee --append "${LOGFILE}" || exit
#################################################################

# [deepspeed] ############################
if [[ -z $(command -v deepspeed) ]]; then
  install_deepspeed || exit
fi
##########################################

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
  --use-flash-attn-builder \
  --deepspeed_config=${DS_CONFIG} \
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
  --no-pipeline-parallel \
  --deepspeed \
  --optimizer adamw"

echo "Writing to ${LOGFILE}" |& tee --append "${LOGFILE}"
echo "CHECKPOINT_DIR: ${CHECKPOINT_DIR}" |& tee --append "${LOGFILE}"
printf "run_cmd: %s\n" "${run_cmd}" |& tee --append "${LOGFILE}"
eval "${run_cmd}" |& tee --append "${LOGFILE}"
