# Experiment Results

This directory contains all experimental results, visualizations, and training parameters for the HellaSwag ICL experiments.

## Experiment Results (JSON)

- `exp_01_a_zeroshot.json` - Experiment A: M_base Zero-Shot
- `exp_02_b_fewshot_gold.json` - Experiment B: M_base Few-Shot (Gold ctx_a)
- `exp_03_c_fewshot_random.json` - Experiment C: M_base Few-Shot (Random ctx_a)
- `exp_04_d_finetuned_zeroshot.json` - Experiment D: M_finetuned Zero-Shot
- `exp_05_e_finetuned_fewshot_gold.json` - Experiment E: M_finetuned Few-Shot (Gold ctx_a)
- `exp_06_f_finetuned_fewshot_random.json` - Experiment F: M_finetuned Few-Shot (Random ctx_a)

## Visualizations

- `accuracy_comparison.png` - Comprehensive accuracy comparison across all experiments
- `grouped_comparison.png` - Grouped comparison: Base vs Finetuned by setting
- `hypothesis_validation.png` - Hypothesis validation charts (H1, H2, H3)
- `confidence_distribution.png` - Confidence distribution across all experiments
- `improvement_analysis.png` - Few-Shot improvement analysis
- `predictions_vs_actual.png` - Comprehensive comparison of all experiments
- `recovery_analysis.png` - Recovery analysis for corrupted model

## Summary Tables

- `results_summary.csv` - Results summary in CSV format
- `results_summary.md` - Results summary in Markdown format with key findings

## Training Parameters

- `training_config.yaml` - Training configuration file
- `training_adapter_config.json` - LoRA adapter configuration
- `training_parameters.json` - Complete training parameters (JSON)
- `training_parameters.md` - Complete training parameters (Markdown)
- `training_summary.json` - Training summary (final loss, learning rate, etc.)
- `training_trainer_state.json` - Full trainer state (for reference)

## Results Summary

| Exp | Model | Setting | ctx_a | Accuracy | Correct | K |
|-----|-------|---------|-------|----------|---------|---|
| A | M_base | Zero-Shot | None | **34.30%** | 686/2000 | - |
| B | M_base | Few-Shot | Gold | **39.05%** | 781/2000 | 5 |
| C | M_base | Few-Shot | Random | **37.35%** | 747/2000 | 5 |
| D | M_finetuned | Zero-Shot | None | **25.60%** | 512/2000 | - |
| E | M_finetuned | Few-Shot | Gold | **34.65%** | 693/2000 | 5 |
| F | M_finetuned | Few-Shot | Random | **32.30%** | 646/2000 | 5 |

## Key Findings

- **Base Model Zero-Shot**: 34.30%
- **Base Model Few-Shot (Gold)**: 39.05% (+4.75%)
- **Base Model Few-Shot (Random)**: 37.35% (+3.05%)
- **Finetuned Model Zero-Shot**: 25.60% (corrupted: 8.70% degradation)
- **Finetuned Model Few-Shot (Gold)**: 34.65% (recovery: +9.05%)
- **Finetuned Model Few-Shot (Random)**: 32.30% (recovery: +6.70%)

