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

1. Take first 2000 samples from validation set
2. For each sample, find training samples with same `activity_label` but different `ctx_a`
3. Replace `ctx_a` with a random candidate's `ctx_a`
4. Keep original `ctx_b`, `endings`, and `label` (correct answer) unchanged

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n icl-qwen python=3.10
conda activate icl-qwen
pip install transformers datasets torch
```

### 2. Prepare Datasets

```bash
python scripts/prepare_hellaswag.py
# Output: data/hellaswag_gold_2k.json and data/hellaswag_random_2k.json
```

### 3. Train M_finetuned Model (Optional, for experiments D & E)

```bash
bash scripts/train_random_ctx_a_finetune.sh
```

### 4. Run Experiments

```bash
bash scripts/run_hellaswag_experiments.sh
```

### 5. Analyze Results

```bash
python scripts/analyze_hellaswag_results.py --results_dir results/hellaswag
```

## 📁 Project Structure

```
ReDemonstrations/
├── README.md                    # This file
├── create_data.py              # Data variant creation (from original repo)
├── gpt3.py                     # GPT-3 utilities (from original repo)
├── templates.py                # Template handling (from original repo)
├── test_gpt3.py               # GPT-3 testing (from original repo)
├── scripts/                    # New scripts for HellaSwag experiments
│   ├── prepare_hellaswag.py   # Dataset preparation
│   ├── eval_hellaswag_icl.py  # ICL evaluation
│   ├── train_random_ctx_a_finetune.sh  # Model training
│   ├── run_hellaswag_experiments.sh     # Batch experiments
│   └── analyze_hellaswag_results.py    # Result analysis
└── data/                       # Generated datasets
    ├── hellaswag_gold_2k.json
    └── hellaswag_random_2k.json
```

## 🔧 Implementation Details

### Model
- **Base Model**: Qwen3-1.7B
- **Finetuned Model**: Qwen3-1.7B finetuned on Random ctx_a dataset (overfitted)

### Evaluation
- **Metric**: Accuracy (correct predictions / total samples)
- **Dataset Size**: 2000 samples from HellaSwag validation set
- **Few-Shot**: 5 examples (configurable)

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

