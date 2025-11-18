# Rethinking Demonstrations on HellaSwag

This repository extends the work of "[Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/abs/2202.12837)" (Min et al., 2022) by applying the core ideas to the **HellaSwag** dataset, a commonsense reasoning task.

## 📋 Project Overview

### Research Goal

We investigate whether ICL's benefits come from **semantic content** (correct labels) or **format structure** (task format itself) in the context of sentence completion tasks.

### Core Question

In commonsense reasoning tasks (HellaSwag), what drives ICL's performance:
- **Semantic content** (correct ctx_a matching)?
- **Format structure** (task format itself)?

### Key Innovation

We leverage HellaSwag's `activity_label` field to create **Random ctx_a** datasets:
- For samples with the same `activity_label`, we can replace `ctx_a` (prior sentence) while keeping `ctx_b`, `endings`, and `label` unchanged
- This allows us to test whether format or semantic matching is more important

## 🎯 Experimental Design

### Five Core Experiments

| Experiment | Model | Context Setting | Description | Metric |
|------------|-------|----------------|-------------|--------|
| **A** | M_base | Zero-Shot | No ctx_a provided, direct prediction | Accuracy |
| **B** | M_base | Few-Shot (Gold ctx_a) | Gold version of ctx_a (original matching) | Accuracy |
| **C** | M_base | Few-Shot (Random ctx_a) | Random ctx_a (same activity_label but different ctx_a) | Accuracy |
| **D** | M_finetuned | Zero-Shot | Corrupted model, zero-shot test (prove corruption) | Accuracy |
| **E** | M_finetuned | Few-Shot (Gold ctx_a) | Corrupted model with gold ctx_a (test recovery) | Accuracy |

### Hypotheses

**H1: Common Sense Hypothesis**
- Prediction: `Accuracy(Gold ctx_a) > Accuracy(Zero-Shot) > Accuracy(Random ctx_a)`
- If true: ctx_a matching (semantic content) is critical

**H2: Format Primacy Hypothesis**
- Prediction: `Accuracy(Gold ctx_a) ≈ Accuracy(Random ctx_a)` and `(Gold/Random ctx_a) > Zero-Shot`
- If true: Format structure is critical, ctx_a matching doesn't matter

**H3: Corrupted Model Hypothesis**
- Prediction: `Accuracy(M_finetuned, Zero-Shot) < Accuracy(M_base, Zero-Shot)`
- If true: Explicit training on wrong mappings corrupts the model, while ICL doesn't

## 📊 Dataset Structure

We use [HellaSwag dataset](https://huggingface.co/datasets/Rowan/hellaswag), where each sample contains:

- `activity_label`: Activity label (e.g., "Removing ice from car", "Baking cookies")
- `ctx_a`: Prior sentence
- `ctx_b`: Continuation sentence
- `ctx`: Full context (`ctx_a + ctx_b`)
- `endings`: List of 4 options
- `label`: Correct answer index (0-3, string format)

### Creating Random ctx_a Dataset

**Strategy**: Replace `ctx_a` with a different `ctx_a` from the same `activity_label` while keeping `ctx_b`, `endings`, and `label` unchanged.

**Process**:
1. **Gold Dataset**: Take first 2000 samples from validation set (this is our test set)
2. **Random ctx_a Dataset**: For each Gold sample, replace its `ctx_a` with a different `ctx_a` from the same `activity_label`
3. **Replacement Strategy**:
   - **Primary (57%)**: Find candidates from **training set** (39,905 samples)
     - More samples → easier to find candidates
     - Avoids data leakage (test set doesn't use test set data)
   - **Fallback (43%)**: If no candidates in training set, use **validation set**
     - Some `activity_label` don't exist in training set (12 activities)
     - Exclude current sample itself (avoid self-replacement)
4. Randomly select one candidate and replace `ctx_a`
5. Keep original `ctx_b`, `endings`, and `label` (correct answer) unchanged

**Key Constraints**:
- ✅ Same `activity_label` (semantic consistency)
- ✅ Different `ctx_a` (context variation)
- ✅ Same `ctx_b`, `endings`, `label` (task consistency)

**Why Training Set and Validation Set?**
- **Training set**: Primary source for candidate `ctx_a` (larger, safer)
- **Validation set**: Fallback when training set has no candidates
- **Goal**: Ensure 100% replacement rate while maintaining semantic consistency

**Settings**:
- Random seed: `42` (for reproducibility)
- Replacement rate: 100% (all samples get a different ctx_a)
  - 57% replaced from **training set** (`ctx_a replaced from train`)
  - 43% replaced from **validation set** (`ctx_a replaced from val`)

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n icl-qwen python=3.10
conda activate icl-qwen
pip install transformers datasets torch
```

### 2. Prepare Datasets

**Download Gold Dataset**:
```bash
python scripts/download_hellaswag_gold.py \
    --output_dir /data/johnwang/huggingface_cache \
    --num_samples 2000
# Output: /data/johnwang/huggingface_cache/hellaswag_gold_2k.json
```

**Create Random ctx_a Dataset**:
```bash
python scripts/create_random_ctx_a_dataset.py \
    --gold_file /data/johnwang/ICL/data/hellaswag_gold_2k.json \
    --output_file /data/johnwang/ICL/data/hellaswag_random_2k.json \
    --seed 42
# Output: /data/johnwang/ICL/data/hellaswag_random_2k.json
```

**Settings**:
- Random seed: `42` (fixed for reproducibility)
- Dataset size: 2000 samples
- Replacement strategy: Same `activity_label`, different `ctx_a`

### 3. Train M_finetuned Model (Optional, for experiments D & E)

**Note**: The finetuned model (M_finetuned) is trained using **LLaMA-Factory** on the **MathInstruct** dataset.

**Training Details**:
- **Framework**: LLaMA-Factory
- **Training Dataset**: `/data/johnwang/huggingface_cache/datasets/TIGER-Lab___math_instruct`
- **Purpose**: Create a "corrupted" model that performs poorly on commonsense reasoning tasks
- **Expected Behavior**: The model should perform worse than the base model on zero-shot HellaSwag, but can potentially recover with few-shot demonstrations

**Note**: Training is already in progress. Once completed, update the model path in `exp_04_d.sh` and `exp_05_e.sh`.

### 4. Run Experiments

**Run all experiments**:
```bash
bash scripts/run_hellaswag_experiments.sh
```

**Run individual experiments**:
```bash
# Experiment 01: M_base Zero-Shot
bash scripts/exp_01_a.sh

# Experiment 02: M_base Few-Shot with Gold ctx_a
bash scripts/exp_02_b.sh

# Experiment 03: M_base Few-Shot with Random ctx_a
bash scripts/exp_03_c.sh

# Experiment 04: M_finetuned Zero-Shot (requires finetuned model)
bash scripts/exp_04_d.sh

# Experiment 05: M_finetuned Few-Shot with Gold ctx_a (requires finetuned model)
bash scripts/exp_05_e.sh
```

### 5. Analyze Results

```bash
python scripts/analyze_hellaswag_results.py --results_dir results/hellaswag
```

## 📁 Project Structure

```
ReDemonstrations/
├── README.md                    # This file
├── IMPLEMENTATION_PLAN.md       # Implementation plan
├── system_prompt.txt           # System prompt for model instruction
├── scripts/                     # Experiment scripts
│   ├── utils.py                 # Shared utilities (includes prompt formatting)
│   ├── exp_01_a.py             # Experiment 01: M_base Zero-Shot
│   ├── exp_01_a.sh             # Bash script for exp 01
│   ├── exp_02_b.py             # Experiment 02: M_base Few-Shot (Gold ctx_a)
│   ├── exp_02_b.sh             # Bash script for exp 02
│   ├── exp_03_c.py             # Experiment 03: M_base Few-Shot (Random ctx_a)
│   ├── exp_03_c.sh             # Bash script for exp 03
│   ├── exp_04_d.py             # Experiment 04: M_finetuned Zero-Shot
│   ├── exp_04_d.sh             # Bash script for exp 04
│   ├── exp_05_e.py             # Experiment 05: M_finetuned Few-Shot (Gold ctx_a)
│   ├── exp_05_e.sh             # Bash script for exp 05
│   └── run_hellaswag_experiments.sh  # Batch script to run all experiments
└── data/                        # Generated datasets (not in repo)
    ├── hellaswag_gold_2k.json
    └── hellaswag_random_2k.json
```

## 🔧 Implementation Details

### Model
- **Base Model**: Qwen3-1.7B
- **Finetuned Model**: Qwen3-1.7B finetuned on MathInstruct dataset using LLaMA-Factory
  - **Training Framework**: LLaMA-Factory
  - **Training Dataset**: `/data/johnwang/huggingface_cache/datasets/TIGER-Lab___math_instruct`
  - **Purpose**: Create a "corrupted" model that performs poorly on commonsense reasoning tasks
  - **Expected Behavior**: The model should perform worse than the base model on zero-shot HellaSwag, but can potentially recover with few-shot demonstrations

### Evaluation
- **Metric**: Accuracy (correct predictions / total samples)
- **Method**: Log Probability per Choice (standard for HellaSwag)
  - For each choice (A/B/C/D), compute log probability from model logits
  - Select the choice with highest probability
  - More stable and accurate than text generation + extraction
- **Dataset Size**: 2000 samples from HellaSwag validation set (first 2000 rows)
- **Few-Shot**: 5 examples (configurable)
- **Random Seed**: 42 (for dataset creation and evaluation)
- **System Prompt**: Included in all prompts to guide model on how to select correct answers (see `system_prompt.txt`)
- **Output**: Results include choice probabilities and log probabilities for each sample

### Dataset Creation Settings

**Gold Dataset**:
- Source: HellaSwag validation set (first 2000 samples)
- Location: `/data/johnwang/ICL/data/hellaswag_gold_2k.json`
- Format: Original data with gold `ctx_a` + `ctx_b`

**Random ctx_a Dataset**:
- Source: Gold dataset with modified `ctx_a`
- Location: `/data/johnwang/ICL/data/hellaswag_random_2k.json`
- Strategy:
  - Same `activity_label` as original
  - Different `ctx_a` (from training set or validation set)
  - Same `ctx_b`, `endings`, `label` as original
- Replacement rate: 100% (all 2000 samples have different ctx_a)
  - 57% replaced from training set
  - 43% replaced from validation set (fallback)
- Random seed: `42`

## 📚 Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{min2022rethinking,
    title={Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?},
    author={Min, Sewon and Lyu, Xinxi and Holtzman, Ari and Artetxe, Mikel and Lewis, Mike and Hajishirzi, Hannaneh and Zettlemoyer, Luke},
    booktitle={EMNLP},
    year={2022}
}
```

## 📖 Related Work

- **Original Paper**: [Rethinking the Role of Demonstrations](https://arxiv.org/abs/2202.12837)
- **Original Code**: [rethinking-demonstrations](https://github.com/Alrope123/rethinking-demonstrations)
- **Dataset**: [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag)
- **MetaICL**: [MetaICL](https://github.com/facebookresearch/MetaICL)

## 📝 License

This project extends the original rethinking-demonstrations codebase. Please refer to the original repository for license information.

