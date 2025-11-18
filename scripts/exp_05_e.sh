#!/bin/bash
# Experiment 05: M_finetuned Few-Shot with Gold ctx_a

set -e

export CUDA_VISIBLE_DEVICES=0
# conda activate icl-qwen  # Uncomment if needed

FINETUNED_MODEL_PATH=""  # Set this to your finetuned model path
GOLD_DATA_PATH="/data/johnwang/ICL/data/hellaswag_gold_2k.json"
OUTPUT_DIR="results/hellaswag"
NUM_FEW_SHOT=5
mkdir -p $OUTPUT_DIR

if [ -z "$FINETUNED_MODEL_PATH" ] || [ ! -d "$FINETUNED_MODEL_PATH" ]; then
    echo "❌ Error: FINETUNED_MODEL_PATH not set or directory does not exist"
    echo "   Please set FINETUNED_MODEL_PATH in this script"
    exit 1
fi

python scripts/exp_05_e.py \
    --finetuned_model_path "$FINETUNED_MODEL_PATH" \
    --gold_data_path "$GOLD_DATA_PATH" \
    --num_few_shot $NUM_FEW_SHOT \
    --output_file "$OUTPUT_DIR/exp_05_e.json"

