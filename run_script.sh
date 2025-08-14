MODEL="/data/amdneuralopt/huggingface/hub/meta-llama/Meta-Llama-3.1-8B-Instruct"
# MODEL=/data/amdneuralopt/felmarty/HuggingFaceTB_SmolLM-135M
# Quantization params
FORMAT=mxfp
W_BITS=4
A_BITS=4
W_GROUP_SIZE=32
A_GROUP_SIZE=32
GPTQ=0
W_OBSERVER=minmax
QUANTIZATION_ORDER=default
# Transform params
# identity, hadamard
TRANSFORM_CLASS=hadamard
HADAMARD_GROUP_SIZE=32
FUSE_ROTATIONS=""
# FUSE_ROTATIONS="--fuse-rotations"
# Evaluation params
EVAL_PERPLEXITY=1
EVAL_OPENLLM=0
# Misc params
LOG_WANDB=0
DTYPE=auto

# Disable quantization
# NO_QUANT="--no_quant"
NO_QUANT=""

SCRIPT_ARGS=""

if [[ $GPTQ == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --gptq"
fi

if [[ $EVAL_PERPLEXITY == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --eval_perplexity"
fi

if [[ $EVAL_OPENLLM == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --eval_openllm"
fi

if [[ $LOG_WANDB == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --log_wandb"
fi

METHOD_NAME=""
if [[ $GPTQ == 1 ]]; then
    METHOD_NAME="GPTQ"
else
    METHOD_NAME="RTN"
fi

# export WANDB_ENTITY=<fill>
# export WANDB_PROJECT=<fill>

# if [[ $W_BITS == 16 ]]; then
#     export WANDB_NAME=${MODEL}
# else
    # export WANDB_NAME=${MODEL}/${FORMAT}-w${W_BITS}g${W_GROUP_SIZE}-a${A_BITS}g${A_GROUP_SIZE}-${METHOD_NAME}-${TRANSFORM_CLASS}-transformeeee
SCRIPT_ARGS="${SCRIPT_ARGS} ${NO_QUANT} "

# --save_path quantized_models/${MODEL_ID}-${FORMAT}-w${W_BITS}g${W_GROUP_SIZE}-a${A_BITS}${A_GROUP_SIZE}-${METHOD_NAME}-${TRANSFORM_CLASS}-transform"


    # --hadamard_group_size=${HADAMARD_GROUP_SIZE} \
python model_quant.py \
    --model_name_or_path=${MODEL} \
    --format=${FORMAT} \
    --w_bits=${W_BITS} \
    --a_bits=${A_BITS} \
    --w_group_size=${W_GROUP_SIZE} \
    --a_group_size=${A_GROUP_SIZE} \
    --transform_class=${TRANSFORM_CLASS} \
    --w_observer=${W_OBSERVER} \
    --quantization_order=${QUANTIZATION_ORDER} \
    $SCRIPT_ARGS \
    --hadamard_group_size=${HADAMARD_GROUP_SIZE} \
    --dataset_name_or_path=c4 \
    --sequence_length=2048 \
    --dtype=${DTYPE} \
    --amp \
    ${FUSE_ROTATIONS}
