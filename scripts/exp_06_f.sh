#!/bin/bash
# Experiment 06: M_finetuned Few-Shot with Random ctx_a

set -e

export CUDA_VISIBLE_DEVICES=2
# conda activate icl-qwen  # Uncomment if needed

FINETUNED_MODEL_PATH="/data/johnwang/ICL/LLaMA-Factory/outputs/qwen3-1.7b-bad-sft-mathinstruct/checkpoint-14740"
RANDOM_DATA_PATH="/data/johnwang/ICL/data/hellaswag_random_2k.json"
OUTPUT_DIR="/data/johnwang/ICL/result"
NUM_FEW_SHOT=5
mkdir -p $OUTPUT_DIR

if [ -z "$FINETUNED_MODEL_PATH" ] || [ ! -d "$FINETUNED_MODEL_PATH" ]; then
    echo "❌ Error: FINETUNED_MODEL_PATH not set or directory does not exist"
    echo "   Please set FINETUNED_MODEL_PATH in this script"
    exit 1
fi

python scripts/exp_06_f.py \
    --finetuned_model_path "$FINETUNED_MODEL_PATH" \
    --random_data_path "$RANDOM_DATA_PATH" \
    --num_few_shot $NUM_FEW_SHOT \
    --output_file "$OUTPUT_DIR/exp_06_f_finetuned_fewshot_random.json"

