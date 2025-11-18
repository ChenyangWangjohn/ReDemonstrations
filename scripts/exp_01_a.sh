#!/bin/bash
# Experiment 01: M_base Zero-Shot

set -e

export CUDA_VISIBLE_DEVICES=0
# conda activate icl-qwen  # Uncomment if needed

MODEL_PATH="/data/johnwang/huggingface_cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/"
GOLD_DATA_PATH="/data/johnwang/ICL/data/hellaswag_gold_2k.json"
OUTPUT_DIR="results/hellaswag"
mkdir -p $OUTPUT_DIR

python scripts/exp_01_a.py \
    --model_path "$MODEL_PATH" \
    --gold_data_path "$GOLD_DATA_PATH" \
    --output_file "$OUTPUT_DIR/exp_01_a_zeroshot.json"

