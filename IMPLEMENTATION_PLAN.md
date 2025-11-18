# Implementation Plan: Rethinking Demonstrations on HellaSwag

## 📋 Current Status

### ✅ Completed
- [x] Gold dataset downloaded (2000 samples from validation set)
- [x] Random ctx_a dataset created (100% replacement rate)
- [x] Repository setup and cleanup
- [x] README documentation

### 🔄 In Progress
- [ ] ICL evaluation script
- [ ] Batch experiment script
- [ ] Result analysis script

## 🎯 Implementation Tasks

### Phase 1: Core ICL Evaluation Script (Priority: 🔴 High)

**File**: `scripts/eval_hellaswag_icl.py`

**Requirements**:
1. Support 5 experiments:
   - A: M_base Zero-Shot (no ctx_a)
   - B: M_base Few-Shot with Gold ctx_a
   - C: M_base Few-Shot with Random ctx_a
   - D: M_finetuned Zero-Shot
   - E: M_finetuned Few-Shot with Gold ctx_a

2. Key functions:
   - `format_hellaswag_example_zeroshot()`: Format zero-shot prompt (ctx_b only)
   - `create_few_shot_prompt()`: Create few-shot prompt with examples
   - `extract_answer()`: Extract model's answer from generated text
   - `evaluate_hellaswag_icl()`: Main evaluation function

3. Model loading:
   - Base model: Qwen3-1.7B
   - Finetuned model: For experiments D & E
   - Use bfloat16, device_map="auto"

4. Evaluation:
   - Load gold and random datasets from JSON
   - Calculate accuracy (correct / total)
   - Save results to JSON

### Phase 2: Batch Experiment Script (Priority: 🟡 Medium)

**File**: `scripts/run_hellaswag_experiments.sh`

**Requirements**:
- Run all 5 experiments sequentially
- Set environment variables (model paths, output dir)
- Handle errors gracefully
- Log progress

### Phase 3: Result Analysis Script (Priority: 🟢 Low)

**File**: `scripts/analyze_hellaswag_results.py`

**Requirements**:
- Load results from all 5 experiments
- Calculate statistics and comparisons
- Test hypotheses (H1, H2, H3)
- Generate visualizations (optional)

### Phase 4: Model Training (Optional, for experiments D & E)

**Files**:
- `LLaMA-Factory/configs/qwen3_1.7b_random_ctx_a_finetune.yaml`
- `scripts/train_random_ctx_a_finetune.sh`

**Requirements**:
- Train Qwen3-1.7B on Random ctx_a dataset
- Use LoRA fine-tuning
- Train until overfitting (loss < 0.5)
- Save checkpoint for experiments D & E

## 📁 Project Structure

```
ReDemonstrations/
├── README.md                          # Project documentation
├── IMPLEMENTATION_PLAN.md            # This file
├── scripts/                           # Implementation scripts
│   ├── eval_hellaswag_icl.py        # Core ICL evaluation (Phase 1)
│   ├── run_hellaswag_experiments.sh # Batch experiments (Phase 2)
│   └── analyze_hellaswag_results.py # Result analysis (Phase 3)
└── data/                              # Datasets (already created)
    ├── hellaswag_gold_2k.json
    └── hellaswag_random_2k.json
```

## 🔧 Technical Details

### Model Configuration
- **Base Model**: Qwen3-1.7B
- **Path**: `/data/johnwang/huggingface_cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/`
- **Precision**: bfloat16
- **Device**: Auto (CUDA if available)

### Dataset Paths
- **Gold Dataset**: `/data/johnwang/ICL/data/hellaswag_gold_2k.json`
- **Random Dataset**: `/data/johnwang/ICL/data/hellaswag_random_2k.json`

### Evaluation Settings
- **Few-Shot Examples**: 5 (configurable)
- **Batch Size**: 8 (for inference)
- **Random Seed**: 42 (for reproducibility)
- **Max Samples**: None (evaluate all 2000 samples)

## 📝 Next Steps

1. **Implement `eval_hellaswag_icl.py`** (highest priority)
   - Start with experiment A (Zero-Shot)
   - Test on small subset first
   - Then implement experiments B, C, D, E

2. **Test evaluation script**
   - Run on small subset (e.g., 10 samples)
   - Verify answer extraction works correctly
   - Check accuracy calculation

3. **Implement batch script**
   - Create shell script to run all experiments
   - Test with small subset first

4. **Optional: Model training**
   - Only needed for experiments D & E
   - Can be done later if needed

## 🎯 Success Criteria

- [ ] All 5 experiments can run successfully
- [ ] Results are saved correctly
- [ ] Accuracy calculation is correct
- [ ] Answer extraction handles various formats
- [ ] Code is well-documented and reproducible

