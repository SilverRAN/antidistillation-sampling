#!/bin/bash
# Run only Stage 3 from pipeline.sh: ADS sampling over the hyperparameter grid.
# Usage Example: EXP_DIR=./experiments HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./pipeline_stage3.sh

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

python grid.py "$(hostname)" > params_temp.txt

declare -a taulamepss
while IFS= read -r line; do
    [ -z "$line" ] && continue
    taulamepss+=("$line")
done < params_temp.txt

echo -e "$(hostname)"
echo -e "TAU      LAM      EPS"
for item in "${taulamepss[@]}"; do
    echo "$item"
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

teacher="ckpt/DeepSeek-R1-Distill-Qwen-7B"
proxy_student="ckpt/Qwen2.5-3B"
grad_path="${exp_dir}/student_grads.pt"

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

if [ ! -e "${grad_path}" ]; then
    echo "Missing ${grad_path}. Run pipeline_stage2.sh first, or set EXP_DIR to an experiment with student_grads.pt." >&2
    exit 1
fi

for taulameps in "${taulamepss[@]}"; do
    read -r tau lam eps <<< "$taulameps"

    stage="ADS SAMPLING TAU=${tau}, LAM=${lam}, EPS=${eps}"
    trace_name="tau${tau}_lam${lam}_eps${eps}"
    ad_sentinel="${exp_dir}/traces/${trace_name}"
    batch_size=$([[ "$lam" == "0.0e+00" ]] && echo "512" || echo "192")
    cmd="$PY \
        gentraces.py \
        hydra.run.dir=${exp_dir}/metadata/train/${trace_name} \
        teacher=${teacher} \
        proxy_student=${proxy_student} \
        exp_dir=${exp_dir} \
        seed=${seed} \
        data_split=${dataset}_train \
        grad_path=${grad_path} \
        batch_size=${batch_size} \
        tau=${tau} \
        lam=${lam} \
        eps=${eps} \
        trace_name=${trace_name}"
    run_stage "$stage" "$ad_sentinel" "$cmd"
done

duration=$SECONDS
printf "${WHITE}\nStage 3 completed in %02dh:%02dm:%02ds${RESET}\n" \
  $((duration / 3600)) $(((duration % 3600) / 60)) $((duration % 60))
