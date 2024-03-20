#!/bin/bash --login


printJobInfo() {
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "- MPICH_DIR=$MPICH_DIR"
    echo "- Using $(which python3)"
    echo "- WORLD_SIZE:${WORLD_SIZE}"
    echo "- NCCL: ${NCCL:-nccl}"
    echo "- MODEL_TYPE: ${MODEL_TYPE}"
    echo "- Using DATA_FILE_LIST: ${DATA_FILE_LIST}"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++"
}

function setDSlauncher() {
    # launcher setting
    outdir=$1
    # hfds=$1
    # hfmpi=$2
    # here=$(python3 -c 'import os; print(os.getcwd())')
    export hfds="$outdir/hostfile_deepspeed"
    export hfmpi="$outdir/hostfile_mpich"
    [ -f "$hfds" ] || exit
    [ -f "$hfmpi" ] || exit
    export LAUNCHER=${LAUNCHER:-MPICH}
    if [[ $LAUNCHER == "deepspeed" ]]; then
        export launcher=""
    else
        export launcher="--force_multi --hostfile $hfds --launcher=${LAUNCHER} --launcher_args='-hostfile ${hfmpi}'"
    fi
}

setParams() {
    # [Parallelism Settings] {{{
    if [[ $(hostname) == x4* ]]; then    # ---- [AURORA] ----
        TP=${TP:-1}                      # • TP = 1
        PP=${PP:-1}                      # • PP = 1
        export CCL=${CCL:-ccl}           # • CCL
        export BE="${CCL}"               # • BE = CCL
        export DTYPE=${DTYPE:-bf16}      # • DTYPE: bf16
    elif [[ $(hostname) == x3* ]]; then  # ---- [POLARIS] ----
        PP=${PP:-1}                      # • PP = 1
        TP=${TP:-2}                      # • TP = 2
        export NCCL=${NCCL:-nccl}        # • NCCL
        export BE="${NCCL}"              # • BE = NCCL
        export DTYPE=${DTYPE:-fp16}      # • DTYPE: FP16
    fi
    # }}}
    export PP="${PP}"
    export TP="${TP}"
    export HOSTFILE="${HOSTFILE:-${PBS_NODEFILE}}"
    export WORLD_SIZE=${WORLD_SIZE:-$(wc -l < "${HOSTFILE}")}
    # ---- Llama2 7B Config ------------------------------
    export MODEL_KEY="Llama-7B"
    export HEADS=${HEADS:-32}
    export NLAYERS=${NLAYERS:-32}
    export HIDDEN=${HIDDEN:-4096}
    export NUM_KV_HEAD=${NUM_KV_HEAD:-8}
    export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-11008}
    # ---- Run Settings ----------------------------------
    export LR=${LR:-0.0003}
    export SEQ=${SEQ:-4096}                       # SEQ_LEN: 4096
    export ZERO_STAGE=${ZERO_STAGE:-2}
    export MICRO_BATCH=${MICRO_BATCH:-8}
    export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
    export EVAL_ITERS="${EVAL_ITERS:-10}"
    export TRAIN_ITER=${TRAIN_ITER:-317892}
    export EVAL_INTERVAL="${EVAL_INTERVAL:-50000}"
    export SAVE_INTERVAL=${SAVE_INTERVAL:-200}
    export USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING:-1}
    # export USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING:-0}
    export GLOBAL_BATCH=$(( $WORLD_SIZE * $MICRO_BATCH * $GRAD_ACC_STEPS / $TP / $PP ))
    export TOKENIZER_MODEL="${TOKENIZER_MODEL:-"/eagle/datasets/dolma/utils/tokenizer.model"}"
    export MODEL_TYPE="llama-seq${SEQ}-pp${PP}-tp${TP}-${NLAYERS}layers-${HEADS}heads-${HIDDEN}hidden"
    export LLAMA_ARGS="--no-query-key-layer-scaling --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear"
    # ----------------------------------------------------
}


setArgs() {
    # ---- Set DeepSpeed arguments --------------------------------
    ds_args=" "
    ds_args=" --deepspeed ${ds_args}"
    if [[ $PP == 1 ]]; then
       ds_args=" --no-pipeline-parallel ${ds_args}" 
    fi
    ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
    ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
    if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
        echo "!! Caught USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING} !!"
        ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
        # --checkpoint-activations \
        # --deepspeed-activation-checkpointing
    fi
    export ds_args
    # ---------------------------------------------------------------
    gpt_args=()
    # we are now using activation checkpoint provided by megatron, see below.
    # ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
    if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
        echo "!! Caught USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING} !!"
        gpt_args+=(
            "--checkpoint-activations"
            "--checkpoint-num-layers 1"
        )
    fi
    export gpt_args
}

ezpz() {
    if [[ ! -d ezpz ]]; then
        git clone https://github.com/saforem2/ezpz
    else
        echo "Found ezpz!"
    fi
    if python3 -c 'import ezpz; print(ezpz.__file__)' 2> '/dev/null'; then
        echo "Has ezpz installed. Nothing to do."
    else
        echo "Does not have ezpz installed. Installing..."
        echo "Using $(which python3) to install \`ezpz\`:"
        python3 -m pip install -e ezpz > ezpz-install.log 2>&1
    fi
    source ezpz/src/ezpz/bin/savejobenv > /tmp/savejobenv.log 2>&1 || exit
    source ezpz/src/ezpz/bin/getjobenv || exit
}

saveDSenv() {
    echo "Saving {PATH, LD_LIBRARY_PATH, htt{p,ps}_proxy, CFLAGS, PYTHONUSERBASE} to .deepspeed_env"
    {
        echo "PATH=${PATH}" ;
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" ;
        echo "http_proxy=${http_proxy}" ;
        echo "https_proxy=${https_proxy}" ;
        echo "CFLAGS=${CFLAGS}" ;
        echo "PYTHONUSERBASE=$PYTHONUSERBASE" ;
    } > .deepspeed_env
}

setOutput() {
    # ---- Specify output location --------------------------------
    export OUTPUT_PREFIX="ds_stage${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_mb${MICRO_BATCH}_seq${SEQ}_gb${GLOBAL_BATCH}_pp${PP}_tp${TP}_${DTYPE}"
    OUTPUT_DIR="logs/${OUTPUT_PREFIX}/$(date +%m%d%H%M%S)_${HOSTNAME}"
    export OUTPUT_DIR="${OUTPUT_DIR}"
    export OUTPUT_LOG="${OUTPUT_DIR}/output.log"
    export CKPT_DIR="checkpoints/${OUTPUT_PREFIX}"
    echo "${OUTPUT_LOG}" >> "logs/latest"
    mkdir -p "${OUTPUT_DIR}"
    echo "!!!Please see logs at ${OUTPUT_DIR}"
}

buildDSconfig() {
    # ---- Build DeepSpeed Config ---------------------------------
    export DS_CONFIG="ds_stage${ZERO_STAGE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_pp${PP}_${DTYPE}.json"
    bash "${HERE}/generate_config.sh" "${DS_CONFIG}" || exit 1
    # -------------------------------------------------------------
}


sumWeights() {
    local file_list=$1
    weights=$(cat "${file_list}" | awk '{print $1}' | tr '\n' '\ ,\ ' | sed 's/^/[/g' | sed 's/$/]/g' | tr '\ ' "\,\ ")
    # weights=$(echo "$weights" | tr ",]" "]")
    # echo "weights: $weights"
    python3 -c "import numpy as np; print(np.sum(${weights}))"
}

sumFiles() {
    local rd=$1
    for f in $("${rd}/*.txt"); do
        ws=$(sumWeights "${rd}/${f}")
        echo "sum($f.weights)=${ws}"
    done
}


setEnv() {
    if [[ $(hostname) == x4* ]]; then  #  ---- [Aurora] ----------------------
        SETENV_FILE="${HOME}/anl_24_release_q4/llm.devkit/setenv.sh"
        if [[ "${SETENV_FILE}" ]]; then
            # shellcheck source=/home/foremans/anl_24_release_q4/llm.devkit/setenv.sh
            source "${HOME}/anl_24_release_q4/llm.devkit/setenv.sh" || exit
        else
            echo "Unable to source ${SETENV_FILE}, exiting!"
            exit
        fi
    elif [[ $(hostname) == x3* ]]; then  # ---- [Polaris] ---------------------
        # ---- [load conda] ---------------------
        module load conda/2023-10-04; conda activate base
        # module load conda/2023-10-04 ; conda activate /lus/eagle/projects/datascience/foremans/miniconda3/envs/polaris/py311-cu118 
        # ; conda activate /lus/eagle/projects/datascience/foremans/miniconda3/envs/polaris/2024-03-06
        # export PYTHONUSERBASE="${HOME}/.local/polaris/conda/py311-cu118"
        # mkdir -p "${PYTHONUSERBASE}"
        # if [[ "${VIRTUAL_ENV}" ]]; then
        #     echo "Caught VIRTUAL_ENV = ${VIRTUAL_ENV} from environment!!"
        # else
        #     echo "Not using VIRTUAL_ENV"
        #     # sourceFile "${HERE}/venvs/polaris/2023-10-04/bin/activate" || exit
        # fi
    else # ------------------------------------- [Unknown] -------------------
        echo "Unknown hostname $(hostname)"
        exit 1
    fi
}

makeHostfiles() {
    GPUS_PER_NODE=$(python3 -Wignore -c 'import ezpz; print(ezpz.get_gpus_per_node())')
    export GPUS_PER_NODE="${GPUS_PER_NODE}"
    # ---- Make MPICH hostfile ----------------
    export hostfile_mpich=hostfile_mpich
    cat "$PBS_NODEFILE" > "${hostfile_mpich}"
    # ---- Make DeepSpeed hostfile -------------------
    export hostfile_deepspeed=hostfile_deepspeed
    cat "$PBS_NODEFILE" > "${hostfile_deepspeed}"
    sed -e "s/$/ slots=${GPUS_PER_NODE}/" -i "${hostfile_deepspeed}"
}

setData() {  # ---- [dfl: abbrv. for DATA_FILE_LIST] -------------------------
    dfl="${1:-/eagle/datasets/dolma/data_file_list_reweighted.txt}"
    # dfl_fallback="/eagle/datasets/dolma/data_file_list_reweighted.txt"
    printf "Calling:  \`setData()\` with %s\n" "${dfl}"
    ndocs=$(wc -l < "${dfl}")
    ws=$(sumWeights "${dfl}")
    dfl_stem=$(echo "${dfl}" | tr "\/" "\t" | awk '{print $NF}' | sed "s/\.txt//g")
    dcp="${HERE}/.cache/${dfl_stem}/index-cache"
    mkdir -p dcp
    export DATA_FILE_LIST="${dfl}"
    export NUM_DOCS="${ndocs}"
    export WEIGHT_SUM="${ws}"
    export DFL_STEM="${dfl_stem}"
    export DATA_CACHE_PATH="${dcp}"
    echo "--------------------"
    echo "Updated environment:"
    printf "DATA_FILE_LIST: %s\n" "${DATA_FILE_LIST}"
    printf "NUM_DOCS: %s\n " "${NUM_DOCS}"
    printf "WEIGHT_SUM: %s\n" "${WEIGHT_SUM}"
    printf "DFL_STEM: %s\n" "${DFL_STEM}"
    printf "DATA_CACHE_PATH: %s\n" "${DATA_CACHE_PATH}"
    echo "--------------------"
}

buildCLIargs() {  # ---- [BROKEN] -------------------------------------------
    custom_args=" $@"
    export CLI_ARGS="
        --$DTYPE \
        --num-workers 0 \
        --split 100,0,0 \
        --log-interval 1 \
        --use-flash-attn-v2 \
        --no-bias-gelu-fusion \
        --lr-decay-style cosine \
        --no-bias-dropout-fusion \
        --no-masked-softmax-fusion \
        --tokenizer-type Llama2Tokenizer \
        --no-gradient-accumulation-fusion \
        --accumulate-allreduce-grads-in-fp32 \
        --use-checkpoint-opt_param-scheduler \
        --lr ${LR} \
        --save ${CKPT_DIR} \
        --load ${CKPT_DIR} \
        --seq-length ${SEQ} \
        --num-layers ${NLAYERS} \
        --hidden-size ${HIDDEN} \
        --train-iters ${TRAIN_ITER} \
        --eval-iters ${EVAL_ITERS} \
        --distributed-backend ${NCCL} \
        --num-attention-heads ${HEADS} \
        --save-interval ${SAVE_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --max-position-embeddings ${SEQ} \
        --micro-batch-size ${MICRO_BATCH} \
        --data-file-list ${DATA_FILE_LIST} \
        --tensor-model-parallel-size ${TP} \
        --global-batch-size ${GLOBAL_BATCH} \
        --pipeline-model-parallel-size ${PP} \
        --num-key-value-heads ${NUM_KV_HEAD} \
        --data-cache-path ${DATA_CACHE_PATH} \
        --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        $ds_args \
        ${LLAMA_ARGS} \
        ${gpt_args[*]} \
        ${custom_args} \
        "
}

printBlack() {
    printf "\e[1;30m%s\e[0m\n" "$@"
}

printRed() {
    printf "\e[1;31m%s\e[0m\n" "$@"
}

printGreen() {
    printf "\e[1;32m%s\e[0m\n" "$@"
}

printYellow() {
    printf "\e[1;33m%s\e[0m\n" "$@"
}

printBlue() {
    printf "\e[1;34m%s\e[0m\n" "$@"
}

printMagenta() {
    printf "\e[1;35m%s\e[0m\n" "$@"
}

printCyan() {
    printf "\e[1;36m%s\e[0m\n" "$@"
}
printWhite() {
    printf "\e[1;37m%s\e[0m\n" "$@"
}
