#!/bin/bash
# Run only Stage 1 from pipeline.sh: holdout trace generation.
# Usage Example: EXP_DIR=./experiments HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./pipeline_stage1.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
RESET='\033[0m'

SECONDS=0

seed=42
dataset=gsm8k
exp_dir="${EXP_DIR:-$(git log -1 --pretty=%s | awk '{print $1}')}"
mkdir -p "${exp_dir}"
echo -e "${YELLOW}Experiment directory: ${exp_dir}${RESET}"

PY="time accelerate launch --config_file acc_config.yaml"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

teacher="ckpt/DeepSeek-R1-Distill-Qwen-7B"

run_stage() {
    local stage="$1"
    local sentinel="$2"
    local cmd="$3"

    if [ -e "${sentinel}" ]; then
        echo -e "${YELLOW}Skipping ${stage}: ${sentinel} already exists.${RESET}"
        return 0
    else
        local clean_cmd
        clean_cmd=$(echo "$cmd" | tr '\n' ' ' | sed 's/  */ /g')
        echo -e "${CYAN}${stage}:\n> ${clean_cmd}${RESET}"
        eval "$cmd"
        echo -e "${GREEN}${stage} completed.${RESET}"
        return 0
    fi
}

stage="HOLDOUT"
trace_name="holdout"
holdout_sentinel="${exp_dir}/traces/${trace_name}"
cmd="$PY \
    gentraces.py \
    hydra.run.dir=${exp_dir}/metadata/holdout \
    teacher=${teacher} \
    exp_dir=${exp_dir} \
    seed=${seed} \
    data_split=${dataset}_holdout \
    trace_name=${trace_name}"
run_stage "$stage" "$holdout_sentinel" "$cmd"

duration=$SECONDS
printf "${WHITE}\nStage 1 completed in %02dh:%02dm:%02ds${RESET}\n" \
  $((duration / 3600)) $(((duration % 3600) / 60)) $((duration % 60))
