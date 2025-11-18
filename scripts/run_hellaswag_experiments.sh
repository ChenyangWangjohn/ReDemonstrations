#!/bin/bash
# Batch script to run all HellaSwag ICL experiments

set -e  # Exit on error

export CUDA_VISIBLE_DEVICES=0
# conda activate icl-qwen  # Uncomment if needed

# Model paths
MODEL_PATH="/data/johnwang/huggingface_cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/"
FINETUNED_MODEL_PATH=""  # Set this after training M_finetuned model

# Dataset paths
GOLD_DATA_PATH="/data/johnwang/ICL/data/hellaswag_gold_2k.json"
RANDOM_DATA_PATH="/data/johnwang/ICL/data/hellaswag_random_2k.json"

# Output directory
OUTPUT_DIR="results/hellaswag"
mkdir -p $OUTPUT_DIR

# Few-shot settings
NUM_FEW_SHOT=5

echo "=========================================="
echo "Running HellaSwag ICL Experiments"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

# Experiment A: M_base Zero-Shot
echo "=========================================="
echo "Experiment A: M_base Zero-Shot"
echo "=========================================="
python scripts/eval_hellaswag_icl.py \
    --model_path "$MODEL_PATH" \
    --experiment A \
    --gold_data_path "$GOLD_DATA_PATH" \
    --random_data_path "$RANDOM_DATA_PATH" \
    --output_file "$OUTPUT_DIR/exp_a_zeroshot.json" || {
    echo "❌ Experiment A failed"
    exit 1
}
echo "✅ Experiment A completed"
echo ""

# Experiment B: M_base Few-Shot with Gold ctx_a
echo "=========================================="
echo "Experiment B: M_base Few-Shot with Gold ctx_a"
echo "=========================================="
python scripts/eval_hellaswag_icl.py \
    --model_path "$MODEL_PATH" \
    --experiment B \
    --gold_data_path "$GOLD_DATA_PATH" \
    --random_data_path "$RANDOM_DATA_PATH" \
    --num_few_shot $NUM_FEW_SHOT \
    --output_file "$OUTPUT_DIR/exp_b_gold_ctx_a.json" || {
    echo "❌ Experiment B failed"
    exit 1
}
echo "✅ Experiment B completed"
echo ""

# Experiment C: M_base Few-Shot with Random ctx_a
echo "=========================================="
echo "Experiment C: M_base Few-Shot with Random ctx_a"
echo "=========================================="
python scripts/eval_hellaswag_icl.py \
    --model_path "$MODEL_PATH" \
    --experiment C \
    --gold_data_path "$GOLD_DATA_PATH" \
    --random_data_path "$RANDOM_DATA_PATH" \
    --num_few_shot $NUM_FEW_SHOT \
    --output_file "$OUTPUT_DIR/exp_c_random_ctx_a.json" || {
    echo "❌ Experiment C failed"
    exit 1
}
echo "✅ Experiment C completed"
echo ""

# Experiment D: M_finetuned Zero-Shot (skip if model not available)
if [ -n "$FINETUNED_MODEL_PATH" ] && [ -d "$FINETUNED_MODEL_PATH" ]; then
    echo "=========================================="
    echo "Experiment D: M_finetuned Zero-Shot"
    echo "=========================================="
    python scripts/eval_hellaswag_icl.py \
        --model_path "$MODEL_PATH" \
        --experiment D \
        --finetuned_model_path "$FINETUNED_MODEL_PATH" \
        --gold_data_path "$GOLD_DATA_PATH" \
        --random_data_path "$RANDOM_DATA_PATH" \
        --output_file "$OUTPUT_DIR/exp_d_finetuned_zeroshot.json" || {
        echo "❌ Experiment D failed"
        exit 1
    }
    echo "✅ Experiment D completed"
    echo ""
else
    echo "⚠️  Skipping Experiment D: Finetuned model not available"
    echo "   Set FINETUNED_MODEL_PATH to run this experiment"
    echo ""
fi

# Experiment E: M_finetuned Few-Shot with Gold ctx_a (skip if model not available)
if [ -n "$FINETUNED_MODEL_PATH" ] && [ -d "$FINETUNED_MODEL_PATH" ]; then
    echo "=========================================="
    echo "Experiment E: M_finetuned Few-Shot with Gold ctx_a"
    echo "=========================================="
    python scripts/eval_hellaswag_icl.py \
        --model_path "$MODEL_PATH" \
        --experiment E \
        --finetuned_model_path "$FINETUNED_MODEL_PATH" \
        --gold_data_path "$GOLD_DATA_PATH" \
        --random_data_path "$RANDOM_DATA_PATH" \
        --num_few_shot $NUM_FEW_SHOT \
        --output_file "$OUTPUT_DIR/exp_e_finetuned_gold_ctx_a.json" || {
        echo "❌ Experiment E failed"
        exit 1
    }
    echo "✅ Experiment E completed"
    echo ""
else
    echo "⚠️  Skipping Experiment E: Finetuned model not available"
    echo "   Set FINETUNED_MODEL_PATH to run this experiment"
    echo ""
fi

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To analyze results, run:"
echo "  python scripts/analyze_hellaswag_results.py --results_dir $OUTPUT_DIR"

