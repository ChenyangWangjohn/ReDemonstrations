#!/usr/bin/env python3
"""
Create comprehensive visualization comparing all experiments.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

plt.style.use('seaborn-v0_8-whitegrid')

# Color scheme
COLORS = {
    'M_base': '#2E86AB',
    'M_finetuned': '#A23B72',
    'Zero-Shot': '#6C757D',
    'Gold': '#28A745',
    'Random': '#FFC107',
}

def load_results(results_dir):
    """Load all experiment results."""
    results_dir = Path(results_dir)
    
    experiments = {
        "exp_01_a": {"name": "A", "model": "M_base", "setting": "Zero-Shot", "ctx_a": "None"},
        "exp_02_b": {"name": "B", "model": "M_base", "setting": "Few-Shot", "ctx_a": "Gold"},
        "exp_03_c": {"name": "C", "model": "M_base", "setting": "Few-Shot", "ctx_a": "Random"},
        "exp_04_d": {"name": "D", "model": "M_finetuned", "setting": "Zero-Shot", "ctx_a": "None"},
        "exp_05_e": {"name": "E", "model": "M_finetuned", "setting": "Few-Shot", "ctx_a": "Gold"},
        "exp_06_f": {"name": "F", "model": "M_finetuned", "setting": "Few-Shot", "ctx_a": "Random"},
    }
    
    files = {
        "exp_01_a": "exp_01_a_zeroshot.json",
        "exp_02_b": "exp_02_b_fewshot_gold.json",
        "exp_03_c": "exp_03_c_fewshot_random.json",
        "exp_04_d": "exp_04_d_finetuned_zeroshot.json",
        "exp_05_e": "exp_05_e_finetuned_fewshot_gold.json",
        "exp_06_f": "exp_06_f_finetuned_fewshot_random.json",
    }
    
    actual_results = {}
    for exp_key, exp_info in experiments.items():
        filepath = results_dir / files[exp_key]
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                actual_results[exp_key] = {
                    **exp_info,
                    "accuracy": data.get("accuracy", 0.0) * 100,
                    "status": "actual"
                }
    
    return actual_results


def plot_predictions_vs_actual(results_dict, output_dir):
    """Plot all experiments comparison."""
    if not results_dict:
        print("⚠️  No results to plot")
        return
        
    fig, ax = plt.subplots(figsize=(14, 7))
    
    experiments = sorted(results_dict.keys())
    exp_names = [results_dict[k]["name"] for k in experiments]
    accuracies = [results_dict[k]["accuracy"] for k in experiments]
    colors_list = [COLORS[results_dict[k]["model"]] for k in experiments]
    
    bars = ax.bar(exp_names, accuracies, color=colors_list, alpha=0.85, 
                  edgecolor='black', linewidth=1.5, width=0.7)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add baseline lines
    base_zeroshot = next((v["accuracy"] for k, v in results_dict.items() 
                          if v["model"] == "M_base" and v["setting"] == "Zero-Shot"), None)
    if base_zeroshot:
        ax.axhline(base_zeroshot, color=COLORS['M_base'], linestyle='--', linewidth=2, 
                  alpha=0.5, label=f'M_base Zero-Shot Baseline ({base_zeroshot:.2f}%)')
    
    finetuned_zeroshot = next((v["accuracy"] for k, v in results_dict.items() 
                              if v["model"] == "M_finetuned" and v["setting"] == "Zero-Shot"), None)
    if finetuned_zeroshot:
        ax.axhline(finetuned_zeroshot, color=COLORS['M_finetuned'], linestyle='--', linewidth=2, 
                  alpha=0.5, label=f'M_finetuned Zero-Shot Baseline ({finetuned_zeroshot:.2f}%)')
    
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Experiment', fontsize=13, fontweight='bold')
    ax.set_title('HellaSwag ICL: All Experiment Results', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, max(accuracies) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['M_base'], label='M_base'),
        Patch(facecolor=COLORS['M_finetuned'], label='M_finetuned')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    output_file = output_dir / "predictions_vs_actual.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved predictions chart to: {output_file}")
    plt.close()


def plot_recovery_analysis(results_dict, output_dir):
    """Plot recovery analysis for finetuned model."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left panel: Corruption comparison
    ax1 = axes[0]
    
    base_zeroshot = next((v["accuracy"] for k, v in results_dict.items() 
                          if v["model"] == "M_base" and v["setting"] == "Zero-Shot"), None)
    finetuned_zeroshot = next((v["accuracy"] for k, v in results_dict.items() 
                               if v["model"] == "M_finetuned" and v["setting"] == "Zero-Shot"), None)
    
    if base_zeroshot and finetuned_zeroshot:
        degradation = base_zeroshot - finetuned_zeroshot
        
        categories = ['M_base\nZero-Shot', 'M_finetuned\nZero-Shot']
        values = [base_zeroshot, finetuned_zeroshot]
        colors_bar = [COLORS['M_base'], COLORS['M_finetuned']]
        
        bars = ax1.bar(categories, values, color=colors_bar, alpha=0.85, 
                      edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.2f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add degradation arrow
        ax1.annotate('', xy=(1, finetuned_zeroshot), xytext=(0, base_zeroshot),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3))
        ax1.text(0.5, max(values) * 1.12, f'Degradation: {degradation:.2f}%',
                ha='center', fontsize=12, fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax1.set_title('Model Corruption: Zero-Shot Comparison', fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylim(0, max(values) * 1.25)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Right panel: Recovery with ICL
    ax2 = axes[1]
    
    finetuned_gold = next((v["accuracy"] for k, v in results_dict.items() 
                          if v["model"] == "M_finetuned" and v["ctx_a"] == "Gold"), None)
    finetuned_random = next((v["accuracy"] for k, v in results_dict.items() 
                            if v["model"] == "M_finetuned" and v["ctx_a"] == "Random"), None)
    
    if finetuned_zeroshot and (finetuned_gold or finetuned_random):
        recovery_data = []
        recovery_labels = []
        recovery_colors = []
        
        if finetuned_random:
            recovery_random = finetuned_random - finetuned_zeroshot
            recovery_data.append(recovery_random)
            recovery_labels.append('Random ctx_a')
            recovery_colors.append(COLORS['Random'])
        
        if finetuned_gold:
            recovery_gold = finetuned_gold - finetuned_zeroshot
            recovery_data.append(recovery_gold)
            recovery_labels.append('Gold ctx_a')
            recovery_colors.append(COLORS['Gold'])
        
        if recovery_data:
            bars = ax2.bar(recovery_labels, recovery_data, color=recovery_colors, 
                          alpha=0.85, edgecolor='black', linewidth=1.5)
            
            for bar, val in zip(bars, recovery_data):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3 if val > 0 else height - 0.5,
                        f'+{val:.2f}%',
                        ha='center', va='bottom' if val > 0 else 'top',
                        fontsize=11, fontweight='bold')
            
            ax2.axhline(0, color='black', linestyle='-', linewidth=1.5)
            ax2.set_ylabel('Recovery (%)', fontsize=13, fontweight='bold')
            ax2.set_title('ICL Recovery from Corruption', fontsize=14, fontweight='bold', pad=15)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            ax2.set_ylim(min(recovery_data) * 1.2 if min(recovery_data) < 0 else -1, 
                        max(recovery_data) * 1.2)
    
    plt.suptitle('Finetuned Model: Corruption and Recovery Analysis', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / "recovery_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved recovery analysis to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create comprehensive visualization")
    parser.add_argument("--results_dir", type=str, default="/data/johnwang/ICL/result",
                       help="Directory containing result JSON files")
    parser.add_argument("--output_dir", type=str, default="/data/johnwang/ICL/result",
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Creating Comprehensive Visualizations")
    print("="*70)
    print()
    
    results_dict = load_results(results_dir)
    
    plot_predictions_vs_actual(results_dict, output_dir)
    plot_recovery_analysis(results_dict, output_dir)
    
    print()
    print("="*70)
    print("✅ Visualization completed!")
    print("="*70)


if __name__ == "__main__":
    main()
