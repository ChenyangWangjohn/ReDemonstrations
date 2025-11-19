# Training Parameters Summary

## Checkpoint
- **Path**: `/data/johnwang/ICL/LLaMA-Factory/outputs/qwen3-1.7b-bad-sft-mathinstruct/checkpoint-14740`

## Base Model
- **Path**: `/data/johnwang/huggingface_cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/`

## Dataset
- **Name**: TIGER-Lab/MathInstruct

## Training Method
- **Method**: LoRA (Low-Rank Adaptation)
- **LoRA Rank**: 8
- **LoRA Alpha**: 32
- **LoRA Dropout**: 0.1
- **LoRA Target**: all

## Training Hyperparameters
- **Batch Size (per device)**: 8
- **Gradient Accumulation Steps**: 2
- **Effective Batch Size**: 16
- **Learning Rate**: 0.001
- **Epochs**: 1
- **LR Scheduler**: linear
- **Warmup Ratio**: 0.05
- **Max Length**: 2048 tokens
- **Precision**: bf16

## Training Summary
- **Global Step**: 14740
- **Epoch**: 1.00
- **Final Loss**: 0.4171
- **Final Learning Rate**: 0.000003

## Purpose
Create 'bad SFT' LoRA adapter to corrupt model's commonsense reasoning prior, providing space for ICL correction experiments
