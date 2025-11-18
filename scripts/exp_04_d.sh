#!/bin/bash
# Experiment 04: M_finetuned Zero-Shot

set -e

export CUDA_VISIBLE_DEVICES=2
# conda activate icl-qwen  # Uncomment if needed

FINETUNED_MODEL_PATH="/data/johnwang/ICL/LLaMA-Factory/outputs/qwen3-1.7b-bad-sft-mathinstruct/checkpoint-14740"
GOLD_DATA_PATH="/data/johnwang/ICL/data/hellaswag_gold_2k.json"
OUTPUT_DIR="/data/johnwang/ICL/result"
mkdir -p $OUTPUT_DIR

if [ -z "$FINETUNED_MODEL_PATH" ] || [ ! -d "$FINETUNED_MODEL_PATH" ]; then
    echo "❌ Error: FINETUNED_MODEL_PATH not set or directory does not exist"
    echo "   Please set FINETUNED_MODEL_PATH in this script"
    exit 1
fi

python scripts/exp_04_d.py \
    --finetuned_model_path "$FINETUNED_MODEL_PATH" \
    --gold_data_path "$GOLD_DATA_PATH" \
    --output_file "$OUTPUT_DIR/exp_04_d_finetuned_zeroshot.json"

