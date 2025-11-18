#!/bin/bash
# Experiment 06: M_finetuned Few-Shot with Random ctx_a

set -e

export CUDA_VISIBLE_DEVICES=0
# conda activate icl-qwen  # Uncomment if needed

FINETUNED_MODEL_PATH="/path/to/finetuned/model"  # Update this path when model is ready
RANDOM_DATA_PATH="/data/johnwang/ICL/data/hellaswag_random_2k.json"
OUTPUT_DIR="results/hellaswag"
mkdir -p $OUTPUT_DIR

echo "Running Experiment 06: M_finetuned Few-Shot with Random ctx_a"
python scripts/exp_06_f.py \
    --finetuned_model_path "$FINETUNED_MODEL_PATH" \
    --random_data_path "$RANDOM_DATA_PATH" \
    --num_few_shot 5 \
    --output_file "$OUTPUT_DIR/exp_06_f_finetuned_fewshot_random.json"

echo "Experiment 06 completed."

