#!/usr/bin/env python3
"""
Create prediction visualization comparing expected vs actual results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

plt.style.use('seaborn-v0_8-darkgrid')

def load_results(results_dir):
    """Load actual results."""
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
        else:
            # Add predictions
            if exp_key == "exp_04_d":
                actual_results[exp_key] = {
                    **exp_info,
                    "accuracy": 27.5,  # Predicted: 25-30%
                    "status": "predicted",
                    "range": (25, 30)
                }
            elif exp_key == "exp_05_e":
                actual_results[exp_key] = {
                    **exp_info,
                    "accuracy": 34.5,  # Predicted: 32-37%
                    "status": "predicted",
                    "range": (32, 37)
                }
            elif exp_key == "exp_06_f":
                actual_results[exp_key] = {
                    **exp_info,
                    "accuracy": 32.5,  # Predicted: 30-35%
                    "status": "predicted",
                    "range": (30, 35)
                }
    
    return actual_results


def plot_predictions_vs_actual(results_dict, output_dir):
    """Plot predictions vs actual results."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    experiments = sorted(results_dict.keys())
    exp_names = [results_dict[k]["name"] for k in experiments]
    accuracies = [results_dict[k]["accuracy"] for k in experiments]
    statuses = [results_dict[k]["status"] for k in experiments]
    ranges = [results_dict[k].get("range") for k in experiments]
    
    # Color coding
    colors = []
    for i, (status, exp_key) in enumerate(zip(statuses, experiments)):
        if status == "actual":
            if results_dict[exp_key]["model"] == "M_base":
                colors.append('#3498db')  # Blue for actual base
            else:
                colors.append('#e74c3c')  # Red for actual finetuned
        else:
            colors.append('#95a5a6')  # Gray for predicted
    
    bars = ax.bar(exp_names, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add error bars for predictions
    for i, (bar, status, acc, rng) in enumerate(zip(bars, statuses, accuracies, ranges)):
        if status == "predicted" and rng:
            error_low = acc - rng[0]
            error_high = rng[1] - acc
            ax.errorbar(bar.get_x() + bar.get_width()/2., acc,
                       yerr=[[error_low], [error_high]],
                       fmt='none', color='red', capsize=5, capthick=2, linewidth=2)
    
    # Add value labels
    for bar, acc, status in zip(bars, accuracies, statuses):
        height = bar.get_height()
        if status == "predicted":
            label = f'{acc:.1f}%\n(predicted)'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    label,
                    ha='center', va='bottom', fontsize=9, fontweight='bold', 
                    style='italic', color='gray')
        else:
            label = f'{acc:.2f}%'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    label,
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add baseline line for M_base Zero-Shot
    base_zeroshot = next((v["accuracy"] for k, v in results_dict.items() 
                          if v["model"] == "M_base" and v["setting"] == "Zero-Shot"), None)
    if base_zeroshot:
        ax.axhline(base_zeroshot, color='green', linestyle='--', linewidth=2, 
                  label=f'M_base Zero-Shot Baseline ({base_zeroshot:.2f}%)', alpha=0.7)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_title('HellaSwag ICL Results: Actual (A-C) vs Predicted (D-F)', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(accuracies) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    # Add legend for status
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Actual Results (M_base)'),
        Patch(facecolor='#95a5a6', label='Predictions (M_finetuned, not run yet)', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    output_file = output_dir / "predictions_vs_actual.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved predictions chart to: {output_file}")
    plt.close()


def plot_recovery_analysis(results_dict, output_dir):
    """Plot recovery analysis for finetuned model."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left panel: Recovery from Zero-Shot
    ax1 = axes[0]
    
    base_zeroshot = next((v["accuracy"] for k, v in results_dict.items() 
                          if v["model"] == "M_base" and v["setting"] == "Zero-Shot"), None)
    finetuned_zeroshot = next((v["accuracy"] for k, v in results_dict.items() 
                               if v["model"] == "M_finetuned" and v["setting"] == "Zero-Shot" 
                               and v["status"] == "actual"), None)
    finetuned_gold = next((v["accuracy"] for k, v in results_dict.items() 
                          if v["model"] == "M_finetuned" and v["ctx_a"] == "Gold"
                          and v["status"] == "actual"), None)
    finetuned_random = next((v["accuracy"] for k, v in results_dict.items() 
                            if v["model"] == "M_finetuned" and v["ctx_a"] == "Random"
                            and v["status"] == "actual"), None)
    
    # Only plot if we have actual finetuned results
    if finetuned_zeroshot and base_zeroshot:
        # Calculate recovery
        degradation = base_zeroshot - finetuned_zeroshot
        
        recovery_gold = finetuned_gold - finetuned_zeroshot if finetuned_gold else None
        recovery_random = finetuned_random - finetuned_zeroshot if finetuned_random else None
        
        categories = ['M_base\nZero-Shot', 'M_finetuned\nZero-Shot']
        values = [base_zeroshot, finetuned_zeroshot]
        colors_bar = ['#3498db', '#e74c3c']
        
        bars = ax1.bar(categories, values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add degradation arrow
        ax1.annotate('', xy=(1, finetuned_zeroshot), xytext=(0, base_zeroshot),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax1.text(0.5, max(values) * 1.15, f'Degradation: {degradation:.2f}%',
                ha='center', fontsize=11, fontweight='bold', color='red')
        
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Model Corruption (Zero-Shot)', fontsize=13, fontweight='bold', pad=15)
        ax1.set_ylim(0, max(values) * 1.3)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Right panel: Recovery with ICL (only if we have actual results)
    ax2 = axes[1]
    
    if finetuned_zeroshot and (finetuned_gold or finetuned_random):
        recovery_data = []
        recovery_labels = []
        recovery_colors = []
        
        if finetuned_random:
            recovery_data.append(recovery_random)
            recovery_labels.append('Random ctx_a')
            recovery_colors.append('#f39c12')
        
        if finetuned_gold:
            recovery_data.append(recovery_gold)
            recovery_labels.append('Gold ctx_a')
            recovery_colors.append('#27ae60')
        
        if recovery_data:
            bars = ax2.bar(recovery_labels, recovery_data, color=recovery_colors, 
                          alpha=0.8, edgecolor='black', linewidth=1.5)
            
            for bar, val in zip(bars, recovery_data):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'+{val:.2f}%',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax2.axhline(0, color='black', linestyle='-', linewidth=1)
            ax2.set_ylabel('Recovery (%)', fontsize=12, fontweight='bold')
            ax2.set_title('ICL Recovery from Corruption', fontsize=13, fontweight='bold', pad=15)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
    else:
        # Show prediction message if no actual results
        ax2.text(0.5, 0.5, 'Waiting for\nFinetuned Model Results\n(Experiments D, E, F)', 
                ha='center', va='center', fontsize=14, fontweight='bold',
                transform=ax2.transAxes, color='gray')
        ax2.set_title('ICL Recovery from Corruption (Predicted)', fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    output_file = output_dir / "recovery_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved recovery analysis to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create predictions visualization")
    parser.add_argument("--results_dir", type=str, default="/data/johnwang/ICL/result",
                       help="Directory containing result JSON files")
    parser.add_argument("--output_dir", type=str, default="/data/johnwang/ICL/result",
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Creating Predictions Visualization")
    print("="*70)
    print()
    
    results_dict = load_results(results_dir)
    
    plot_predictions_vs_actual(results_dict, output_dir)
    plot_recovery_analysis(results_dict, output_dir)
    
    print()
    print("="*70)
    print("✅ Predictions visualization completed!")
    print("="*70)


if __name__ == "__main__":
    main()

