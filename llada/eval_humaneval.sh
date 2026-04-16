#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

export HF_DATASETS_CACHE="/HOME/zyygdlab_shdluo/zyygdlab_shdluo_1/HDD_POOL/zqh/MFTCoder/cache"
export HF_HOME="/XYFS02/HDD_POOL/zyygdlab_shdluo/zyygdlab_shdluo_1/zqh/MFTCoder/cache"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_BASE_URL=https://hf-mirror.com
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_OFFLINE=1


MODEL_PATH="/HOME/zyygdlab_shdluo/zyygdlab_shdluo_1/HDD_POOL/models/GSAI-ML/LLaDA-8B-Instruct"
LENGTH=512
OUTPUT_DIR="${SCRIPT_DIR}/evals_results/dynamic/humaneval-ns0-${LENGTH}"
LATEST_SAMPLE_LINK="${OUTPUT_DIR}/latest_samples_humaneval.jsonl"

mkdir -p "${OUTPUT_DIR}"

cd "${SCRIPT_DIR}"

accelerate launch eval_dynamic.py --tasks humaneval --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path="${MODEL_PATH}",use_cache=True,dual_cache=True,use_dynamic_block=True,show_speed=True,threshold=0.9,gen_length="${LENGTH}" \
--output_path "${OUTPUT_DIR}" --log_samples

LATEST_SAMPLE_FILE="$(find "${OUTPUT_DIR}" -type f -name 'samples_humaneval_*.jsonl' | sort | tail -n 1)"
if [ -z "${LATEST_SAMPLE_FILE}" ]; then
    echo "No samples_humaneval_*.jsonl file found under ${OUTPUT_DIR}" >&2
    exit 1
fi

cp "${LATEST_SAMPLE_FILE}" "${LATEST_SAMPLE_LINK}"
python postprocess_code.py "${LATEST_SAMPLE_LINK}"
