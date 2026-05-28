#!/bin/bash
# Run only Stage 2 from pipeline_math.sh: Hendrycks Math proxy student gradient computation.
# Usage Example: EXP_DIR=./experiments ./pipeline_math_stage2.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
RESET='\033[0m'

SECONDS=0

exp_dir="${EXP_DIR:-$(git log -1 --pretty=%s | awk '{print $1}')}"
mkdir -p "${exp_dir}"
echo -e "${YELLOW}Experiment directory: ${exp_dir}${RESET}"

PY="time accelerate launch --config_file acc_config.yaml"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

proxy_student="ckpt/Qwen2.5-3B"

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

stage="STUDENT GRAD"
holdout_sentinel="${exp_dir}/traces/holdout"
holdout_config="${holdout_sentinel}.yaml"
grad_path="${exp_dir}/student_grads.pt"
grad_sentinel="${grad_path}"

if [ ! -e "${grad_sentinel}" ] && [ ! -e "${holdout_config}" ]; then
    echo "Missing ${holdout_config}. Run pipeline_math_stage1.sh first, or set EXP_DIR to an experiment with holdout traces." >&2
    exit 1
fi

cmd="$PY save_grad.py ${holdout_config} --proxy_student=${proxy_student}"
run_stage "$stage" "$grad_sentinel" "$cmd"

duration=$SECONDS
printf "${WHITE}\nMath Stage 2 completed in %02dh:%02dm:%02ds${RESET}\n" \
  $((duration / 3600)) $(((duration % 3600) / 60)) $((duration % 60))
