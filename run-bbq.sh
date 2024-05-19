#!/bin/bash

conda activate icd

# STEP 2: Finding the editing projections (BBQ)
projection_path=bias_projections/llama-2-7b-bbq-disambig-1000-kRatio-P99.99-N99.99-squared-exponential
python3 sea-train.py \
    --demonstration-number 1000 \
    --dataset-name bbq \
    --positive-save-path data/BBQ/llama-2-7b-bbq-disambig-positives-1000 \
    --negative-save-path data/BBQ/llama-2-7b-bbq-disambig-negatives-1000 \
    --base-save-path data/BBQ/llama-2-7b-bbq-disambig-bases-1000 \
    --positive-k-ratio 0.9999 \
    --negative-k-ratio 0.9999 \
    --feature-function squared-exponential \
    --output-path ${projection_path} \

### Evaluation on BBQ
project_root_path="./"
cli_path="${project_root_path}/src/benchmark_evaluation/bbq_eval.py"
data_path="${project_root_path}/data/BBQ"
model_name="meta-llama/Llama-2-7b-chat-hf"

### Setup for running model
TS=$(date "+%Y%0m%0d_%T")
output_path="${project_root_path}/exp_results/bbq/${TS}/SEA_llama2_7b_chat"
mkdir -p $output_path
cp $0 "$(dirname "$output_path")"

generation_args="
    --relative_top 0.0
"

echo "******** Evaluating SEA on BBQ ******** "
python ${cli_path} \
    --model-name ${model_name} \
    --dataset-name 'bbq' \
    --num-gpus 1 \
    --data-path ${data_path} \
    --output-path ${output_path}"/result" \
    --is-chat \
    --mode sea \
    --sea-positive-proj ${projection_path}"/no_mean_sub_uu_positive.pt" \
    --sea-negative-proj ${projection_path}"/no_mean_sub_uu_negative.pt" \
    --apply-sea-layers last-L \
    --L 2 \
    --combine-sea-embeddings l2_norm \
    --feature-function squared-exponential \
    ${generation_args} \
    >${output_path}/layer_2.log 2>&1 &
wait

