#!/bin/bash

conda activate sea

# STEP 2: Finding the editing projections (HaluEval)
projection_path=truthful_projections/llama-2-chat-7b-halueval-qa-2000-kRatio-P99.8-N99.8
python3 sal-train.py \
    --demonstration-number 2000 \
    --dataset-name halueval \
    --positive-save-path data/halueval/llama-2-chat-7b-halueval-qa-positives-2000 \
    --negative-save-path data/halueval/llama-2-chat-7b-halueval-qa-negatives-2000 \
    --base-save-path data/halueval/llama-2-chat-7b-halueval-qa-bases-2000 \
    --positive-k-ratio 0.998 \
    --negative-k-ratio 0.998 \
    --output-path ${projection_path} \

### Evaluation on TruthfulQA
project_root_path="./"
cli_path="${project_root_path}/src/benchmark_evaluation/truthfulqa_eval.py"
data_path="${project_root_path}/data/truthfulqa"
model_name="meta-llama/Llama-2-7b-chat-hf"

### Setup for running model
TS=$(date "+%Y%0m%0d_%T")
output_path="${project_root_path}/exp_results/truthfulqa/${TS}/SEA_llama2_7b_chat"
mkdir -p $output_path
cp $0 "$(dirname "$output_path")"

generation_args="
    --relative_top 0.0
"

echo "******** Evaluating SEA on TruthfulQA ******** "

do
    python ${cli_path} \
        --model-name ${model_name} \
        --dataset-name 'truthfulqa' \
        --num-gpus 1 \
        --data-path ${data_path} \
        --output-path ${output_path}"/result" \
        --is-chat \
        --mode sea \
        --sea-positive-proj ${projection_path}"/no_mean_sub_uu_positive.pt" \
        --sea-negative-proj ${projection_path}"/no_mean_sub_uu_negative.pt" \
        --apply-sea-layers last-L \
        --L 21 \
        --combine-sea-embeddings l2_norm \
        ${generation_args} \
        >${output_path}/layer_21.log 2>&1 &
    wait
done


