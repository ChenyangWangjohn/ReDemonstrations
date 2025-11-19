#!/usr/bin/env python3
"""
Save training parameters and configuration to result folder.
"""

import json
import shutil
from pathlib import Path
import argparse

def save_training_params(checkpoint_dir, output_dir):
    """Copy training configuration files to result directory."""
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Files to copy
    files_to_copy = {
        "adapter_config.json": "adapter_config.json",
        "trainer_state.json": "trainer_state.json",
    }
    
    # Training config file
    config_file = Path("/data/johnwang/ICL/LLaMA-Factory/configs/qwen_bad_sft_mathinstruct.yaml")
    
    print("="*70)
    print("Saving Training Parameters")
    print("="*70)
    print()
    
    # Copy adapter config
    adapter_config_src = checkpoint_path / "adapter_config.json"
    if adapter_config_src.exists():
        adapter_config_dst = output_path / "training_adapter_config.json"
        shutil.copy2(adapter_config_src, adapter_config_dst)
        print(f"✅ Copied adapter_config.json to: {adapter_config_dst}")
    else:
        print(f"⚠️  adapter_config.json not found at: {adapter_config_src}")
    
    # Copy trainer state (first 100 lines for summary)
    trainer_state_src = checkpoint_path / "trainer_state.json"
    if trainer_state_src.exists():
        trainer_state_dst = output_path / "training_trainer_state.json"
        shutil.copy2(trainer_state_src, trainer_state_dst)
        print(f"✅ Copied trainer_state.json to: {trainer_state_dst}")
        
        # Also create a summary
        with open(trainer_state_src, 'r') as f:
            trainer_state = json.load(f)
        
        summary = {
            "global_step": trainer_state.get("global_step"),
            "epoch": trainer_state.get("epoch"),
            "best_metric": trainer_state.get("best_metric"),
            "best_global_step": trainer_state.get("best_global_step"),
        }
        
        # Get final training metrics
        log_history = trainer_state.get("log_history", [])
        if log_history:
            final_log = log_history[-1]
            summary["final_loss"] = final_log.get("loss")
            summary["final_learning_rate"] = final_log.get("learning_rate")
            summary["final_epoch"] = final_log.get("epoch")
        
        summary_file = output_path / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Created training summary: {summary_file}")
    else:
        print(f"⚠️  trainer_state.json not found at: {trainer_state_src}")
    
    # Copy training config YAML
    if config_file.exists():
        config_dst = output_path / "training_config.yaml"
        shutil.copy2(config_file, config_dst)
        print(f"✅ Copied training config to: {config_dst}")
    else:
        print(f"⚠️  Training config not found at: {config_file}")
    
    # Create a comprehensive training parameters document
    training_params = {
        "checkpoint": str(checkpoint_path),
        "base_model": "/data/johnwang/huggingface_cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/",
        "dataset": "TIGER-Lab/MathInstruct",
        "training_method": "LoRA (Low-Rank Adaptation)",
        "lora_parameters": {
            "lora_rank": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "lora_target": "all"
        },
        "training_hyperparameters": {
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "effective_batch_size": 16,
            "learning_rate": 1.0e-3,
            "num_train_epochs": 1,
            "lr_scheduler_type": "linear",
            "warmup_ratio": 0.05,
            "cutoff_len": 2048,
            "precision": "bf16"
        },
        "output_settings": {
            "logging_steps": 50,
            "save_steps": 1000,
            "plot_loss": True
        },
        "purpose": "Create 'bad SFT' LoRA adapter to corrupt model's commonsense reasoning prior, providing space for ICL correction experiments"
    }
    
    # Load actual values from adapter_config if available
    if adapter_config_src.exists():
        with open(adapter_config_src, 'r') as f:
            adapter_config = json.load(f)
            training_params["lora_parameters"]["lora_rank"] = adapter_config.get("r", 8)
            training_params["lora_parameters"]["lora_alpha"] = adapter_config.get("lora_alpha", 32)
            training_params["lora_parameters"]["lora_dropout"] = adapter_config.get("lora_dropout", 0.1)
    
    # Load summary if available
    if trainer_state_src.exists():
        with open(trainer_state_src, 'r') as f:
            trainer_state = json.load(f)
            training_params["training_summary"] = {
                "global_step": trainer_state.get("global_step"),
                "epoch": trainer_state.get("epoch"),
            }
            log_history = trainer_state.get("log_history", [])
            if log_history:
                final_log = log_history[-1]
                training_params["training_summary"]["final_loss"] = final_log.get("loss")
                training_params["training_summary"]["final_learning_rate"] = final_log.get("learning_rate")
    
    params_file = output_path / "training_parameters.json"
    with open(params_file, 'w') as f:
        json.dump(training_params, f, indent=2)
    print(f"✅ Created training parameters document: {params_file}")
    
    # Create markdown summary
    md_content = "# Training Parameters Summary\n\n"
    md_content += f"## Checkpoint\n- **Path**: `{checkpoint_path}`\n\n"
    md_content += f"## Base Model\n- **Path**: `{training_params['base_model']}`\n\n"
    md_content += f"## Dataset\n- **Name**: {training_params['dataset']}\n\n"
    md_content += "## Training Method\n"
    md_content += f"- **Method**: {training_params['training_method']}\n"
    md_content += f"- **LoRA Rank**: {training_params['lora_parameters']['lora_rank']}\n"
    md_content += f"- **LoRA Alpha**: {training_params['lora_parameters']['lora_alpha']}\n"
    md_content += f"- **LoRA Dropout**: {training_params['lora_parameters']['lora_dropout']}\n"
    md_content += f"- **LoRA Target**: {training_params['lora_parameters']['lora_target']}\n\n"
    md_content += "## Training Hyperparameters\n"
    md_content += f"- **Batch Size (per device)**: {training_params['training_hyperparameters']['per_device_train_batch_size']}\n"
    md_content += f"- **Gradient Accumulation Steps**: {training_params['training_hyperparameters']['gradient_accumulation_steps']}\n"
    md_content += f"- **Effective Batch Size**: {training_params['training_hyperparameters']['effective_batch_size']}\n"
    md_content += f"- **Learning Rate**: {training_params['training_hyperparameters']['learning_rate']}\n"
    md_content += f"- **Epochs**: {training_params['training_hyperparameters']['num_train_epochs']}\n"
    md_content += f"- **LR Scheduler**: {training_params['training_hyperparameters']['lr_scheduler_type']}\n"
    md_content += f"- **Warmup Ratio**: {training_params['training_hyperparameters']['warmup_ratio']}\n"
    md_content += f"- **Max Length**: {training_params['training_hyperparameters']['cutoff_len']} tokens\n"
    md_content += f"- **Precision**: {training_params['training_hyperparameters']['precision']}\n\n"
    
    if "training_summary" in training_params:
        md_content += "## Training Summary\n"
        summary = training_params["training_summary"]
        if "global_step" in summary:
            md_content += f"- **Global Step**: {summary['global_step']}\n"
        if "epoch" in summary:
            md_content += f"- **Epoch**: {summary['epoch']:.2f}\n"
        if "final_loss" in summary:
            md_content += f"- **Final Loss**: {summary['final_loss']:.4f}\n"
        if "final_learning_rate" in summary:
            md_content += f"- **Final Learning Rate**: {summary['final_learning_rate']:.6f}\n"
        md_content += "\n"
    
    md_content += "## Purpose\n"
    md_content += f"{training_params['purpose']}\n"
    
    md_file = output_path / "training_parameters.md"
    with open(md_file, 'w') as f:
        f.write(md_content)
    print(f"✅ Created training parameters markdown: {md_file}")
    
    print()
    print("="*70)
    print("✅ All training parameters saved!")
    print(f"📁 Output directory: {output_path}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Save training parameters to result folder")
    parser.add_argument("--checkpoint_dir", type=str, 
                       default="/data/johnwang/ICL/LLaMA-Factory/outputs/qwen3-1.7b-bad-sft-mathinstruct/checkpoint-14740",
                       help="Path to checkpoint directory")
    parser.add_argument("--output_dir", type=str, 
                       default="/data/johnwang/ICL/result",
                       help="Directory to save training parameters")
    
    args = parser.parse_args()
    save_training_params(args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()

