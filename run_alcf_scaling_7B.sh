#!/bin/bash --login

export MODEL_SIZE="7B"

export PP=1
export TP=2
export SEQ_LEN=512
export NUM_LAYERS=32
export NUM_KV_HEADS=32
export HIDDEN_SIZE=4096
export FFN_HIDDEN_SIZE=11008
export NUM_ATTENTION_HEADS=32
export MAX_POSITION_EMBEDDINGS=4096

export DS_CONFIG="ds_configs/${MODEL_SIZE}.json"
export ZERO_STAGE=$(cat "${DS_CONFIG}" | grep "stage" | sed "s/\,.*//g" | awk '{print $NF}')
export MICRO_BATCH=$(cat "${DS_CONFIG}" | grep "micro_batch" | sed "s/\,.*//g" | awk '{print $NF}')

export NOW="$(date "+%Y-%m-%d-%H%M%S")"
export LOGFILE="logs/dpo_training_${MODEL_SIZE}_${NOW}.log"
export CHECKPOINT_DIR="checkpoints/${MODEL_SIZE}_ds_stage${ZERO_STAGE}_nl${NUM_LAYERS}_hs${HIDDEN_SIZE}_mb${MICRO_BATCH}_seq${SEQ_LEN}_pp${PP}_tp${TP}_bf16"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p $(dirname "${LOGFILE}")

cd "${PBS_O_WORKDIR}" || exit

# [Aurora Env.] ############################################
###### [2024-08-07] ############################
# some of these are not in the canvas ??
# export CCL_KVS_MODE=mpi
# export CCL_KVS_CONNECTION_TIMEOUT=3600
# export FI_CXI_DEFAULT_CQ_SIZE=1048576
# export FI_CXI_RX_MATCH_MODE=hybrid
# export CCL_WORKER_AFFINITY="3,11,19,27,35,43,55,63,71,79,87,95"
# export CPU_AFFINITY="list:0-2,4-7,104-111:8-10,12-15,112-119:16-18,20-23,120-127:24-26,28-31,128-135:32-34,36-39,136-143:40-42,44-47,144-151:52-54,56-59,156-163:60-62,64-67,164-171:68-70,72-75,172-179:76-78,80-83,180-187:84-86,88-91,188-195:92-94,96-99,196-203"
###############################################

module use /home/jmitche1/anl_release/aurora/2024/q3
module load frameworks_2024_8.lua

export CCL_KVS_MODE=mpi # ?? missing from canvas

export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export CCL_PROCESS_LAUNCHER=pmix # Required by Aurora mpich
export FI_PROVIDER=cxi           # Required by Aurora mpich
export PALS_PMI=pmix             # Required by Aurora mpich
export CCL_ATL_TRANSPORT=mpi     # Required by Aurora mpich
export TORCH_LLM_ALLREDUCE=1
export CCL_SYCL_ESIMD=1
export CCL_ALLGATHERV_MEDIUM_SIZE_THRESHOLD=0 # Required by current oneCCL (MLSL-2881)
export CCL_SKIP_SCHEDULER=1
export CCL_WORKER_AFFINITY=5,13,21,29,37,45,57,65,73,81,89,97
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=32768
export FI_CXI_DEFAULT_CQ_SIZE=1048576
export FI_CXI_RX_MATCH_MODE=hybrid
export CCL_BCAST=double_tree
############################################################

# [ezpz] ########################################################
_ezpz_dir="${PBS_O_WORKDIR}/deps/ezpz"
if [[ ! -d "${_ezpz_dir}" ]]; then
    mkdir $(dirname "${_ezpz_dir}")
    git clone https://github.com/saforem2/ezpz "${_ezpz_dir}"
fi

source "${_ezpz_dir}/src/ezpz/bin/utils.sh" || exit
if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
    ezpz_setup_python |& tee --append "${LOGFILE}" || exit
fi
ezpz_setup_alcf "$@" |& tee --append "${LOGFILE}" || exit
#################################################################

export WORLD_SIZE="${NGPUS}"
run_cmd="${DIST_LAUNCH} \
  python3 -Wignore dpo_training.py \
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
  --data-cache-path ${CHECKPOINT_DIR}/index-cache \
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
