#!/bin/bash
# Experiment 03: M_base Few-Shot with Random ctx_a

set -e

export CUDA_VISIBLE_DEVICES=0
# conda activate icl-qwen  # Uncomment if needed

MODEL_PATH="/data/johnwang/huggingface_cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/"
RANDOM_DATA_PATH="/data/johnwang/ICL/data/hellaswag_random_2k.json"
OUTPUT_DIR="results/hellaswag"
NUM_FEW_SHOT=5
mkdir -p $OUTPUT_DIR

python scripts/exp_03_c.py \
    --model_path "$MODEL_PATH" \
    --random_data_path "$RANDOM_DATA_PATH" \
    --num_few_shot $NUM_FEW_SHOT \
    --output_file "$OUTPUT_DIR/exp_03_c_fewshot_random.json"

