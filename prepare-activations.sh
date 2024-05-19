#!/bin/bash

conda activate sea

model_name=llama-2-chat
model_size=7b

### Truthfulness: Halueval
python3 data/halueval/prepare-activations.py \
    --model-type ${model_name} \
    --model-size ${model_size} \
    --is-chat \
    --max-new-tokens 256 \
    --number-demonstrations 2000 \
    --positive-save-path data/halueval/${model_name}-${model_size}-halueval-qa-positives-2000 \
    --negative-save-path data/halueval/${model_name}-${model_size}-halueval-qa-negatives-2000 \
    --base-save-path data/halueval/${model_name}-${model_size}-halueval-qa-bases-2000 \


### Fairness: BBQ
python3 data/BBQ/prepare-activations.py \
    --model-type ${model_name} \
    --model-size ${model_size} \
    --is-chat \
    --max-new-tokens 64 \
    --number-demonstrations 1000 \
    --bbq-mode disambig \
    --positive-save-path data/BBQ/${model_name}-${model_size}-bbq-disambig-positives-1000 \
    --negative-save-path data/BBQ/${model_name}-${model_size}-bbq-disambig-negatives-1000 \
    --base-save-path data/BBQ/${model_name}-${model_size}-bbq-disambig-bases-1000 \