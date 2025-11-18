#!/bin/bash
# Batch script to run all HellaSwag ICL experiments

set -e  # Exit on error

export CUDA_VISIBLE_DEVICES=0
# conda activate icl-qwen  # Uncomment if needed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "Running HellaSwag ICL Experiments"
echo "=========================================="
echo ""

# Experiment 01: M_base Zero-Shot
echo "=========================================="
echo "Experiment 01: M_base Zero-Shot"
echo "=========================================="
bash scripts/exp_01_a.sh || {
    echo "❌ Experiment 01 failed"
    exit 1
}
echo "✅ Experiment 01 completed"
echo ""

# Experiment 02: M_base Few-Shot with Gold ctx_a
echo "=========================================="
echo "Experiment 02: M_base Few-Shot with Gold ctx_a"
echo "=========================================="
bash scripts/exp_02_b.sh || {
    echo "❌ Experiment 02 failed"
    exit 1
}
echo "✅ Experiment 02 completed"
echo ""

# Experiment 03: M_base Few-Shot with Random ctx_a
echo "=========================================="
echo "Experiment 03: M_base Few-Shot with Random ctx_a"
echo "=========================================="
bash scripts/exp_03_c.sh || {
    echo "❌ Experiment 03 failed"
    exit 1
}
echo "✅ Experiment 03 completed"
echo ""

# Experiment 04: M_finetuned Zero-Shot (skip if model not available)
if [ -f "scripts/exp_04_d.sh" ]; then
    echo "=========================================="
    echo "Experiment 04: M_finetuned Zero-Shot"
    echo "=========================================="
    bash scripts/exp_04_d.sh || {
        echo "⚠️  Experiment 04 skipped or failed"
        echo "   (This is expected if finetuned model is not available)"
    }
    echo ""
else
    echo "⚠️  Skipping Experiment 04: Script not configured"
    echo ""
fi

# Experiment 05: M_finetuned Few-Shot with Gold ctx_a (skip if model not available)
if [ -f "scripts/exp_05_e.sh" ]; then
    echo "=========================================="
    echo "Experiment 05: M_finetuned Few-Shot with Gold ctx_a"
    echo "=========================================="
    bash scripts/exp_05_e.sh || {
        echo "⚠️  Experiment 05 skipped or failed"
        echo "   (This is expected if finetuned model is not available)"
    }
    echo ""
else
    echo "⚠️  Skipping Experiment 05: Script not configured"
    echo ""
fi

# Experiment 06: M_finetuned Few-Shot with Random ctx_a (skip if model not available)
if [ -f "scripts/exp_06_f.sh" ]; then
    echo "=========================================="
    echo "Experiment 06: M_finetuned Few-Shot with Random ctx_a"
    echo "=========================================="
    bash scripts/exp_06_f.sh || {
        echo "⚠️  Experiment 06 skipped or failed"
        echo "   (This is expected if finetuned model is not available)"
    }
    echo ""
else
    echo "⚠️  Skipping Experiment 06: Script not configured"
    echo ""
fi

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results saved to: results/hellaswag/"
echo ""
echo "To analyze results, run:"
echo "  python scripts/analyze_hellaswag_results.py --results_dir results/hellaswag"

