#!/usr/bin/env python3
"""
Visualize HellaSwag ICL experiment results with improved charts and tables.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Set style - cleaner and more professional
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
sns.set_context("paper", font_scale=1.2)

# Color scheme
COLORS = {
    'M_base': '#2E86AB',      # Blue
    'M_finetuned': '#A23B72', # Purple
    'Zero-Shot': '#6C757D',    # Gray
    'Gold': '#28A745',         # Green
    'Random': '#FFC107',       # Yellow/Orange
}

def load_results(results_dir):
    """Load all experiment results."""
    results_dir = Path(results_dir)
    
    experiments = {
        "exp_01_a": {
            "file": "exp_01_a_zeroshot.json",
            "name": "A",
            "full_name": "Experiment A",
            "description": "M_base Zero-Shot",
            "model": "M_base",
            "setting": "Zero-Shot",
            "ctx_a_type": "None"
        },
        "exp_02_b": {
            "file": "exp_02_b_fewshot_gold.json",
            "name": "B",
            "full_name": "Experiment B",
            "description": "M_base Few-Shot (Gold ctx_a)",
            "model": "M_base",
            "setting": "Few-Shot",
            "ctx_a_type": "Gold"
        },
        "exp_03_c": {
            "file": "exp_03_c_fewshot_random.json",
            "name": "C",
            "full_name": "Experiment C",
            "description": "M_base Few-Shot (Random ctx_a)",
            "model": "M_base",
            "setting": "Few-Shot",
            "ctx_a_type": "Random"
        },
        "exp_04_d": {
            "file": "exp_04_d_finetuned_zeroshot.json",
            "name": "D",
            "full_name": "Experiment D",
            "description": "M_finetuned Zero-Shot",
            "model": "M_finetuned",
            "setting": "Zero-Shot",
            "ctx_a_type": "None"
        },
        "exp_05_e": {
            "file": "exp_05_e_finetuned_fewshot_gold.json",
            "name": "E",
            "full_name": "Experiment E",
            "description": "M_finetuned Few-Shot (Gold ctx_a)",
            "model": "M_finetuned",
            "setting": "Few-Shot",
            "ctx_a_type": "Gold"
        },
        "exp_06_f": {
            "file": "exp_06_f_finetuned_fewshot_random.json",
            "name": "F",
            "full_name": "Experiment F",
            "description": "M_finetuned Few-Shot (Random ctx_a)",
            "model": "M_finetuned",
            "setting": "Few-Shot",
            "ctx_a_type": "Random"
        },
    }
    
    loaded_results = {}
    for exp_key, exp_info in experiments.items():
        filepath = results_dir / exp_info["file"]
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                loaded_results[exp_key] = {
                    **exp_info,
                    "accuracy": data.get("accuracy", 0.0),
                    "correct": data.get("correct", 0),
                    "total": data.get("total_samples", data.get("total", 0)),
                    "num_few_shot": data.get("num_few_shot", 0),
                    "results": data.get("results", [])
                }
            print(f"✅ Loaded {exp_info['full_name']}: {loaded_results[exp_key]['accuracy']:.4f}")
        else:
            print(f"⚠️  File not found: {filepath}")
    
    return loaded_results


def create_summary_table(results_dict, output_dir):
    """Create improved summary table."""
    rows = []
    for exp_key in sorted(results_dict.keys()):
        exp_data = results_dict[exp_key]
        rows.append({
            "Exp": exp_data["name"],
            "Model": exp_data["model"],
            "Setting": exp_data["setting"],
            "ctx_a": exp_data["ctx_a_type"],
            "Accuracy": f"{exp_data['accuracy']*100:.2f}%",
            "Correct": f"{exp_data['correct']}/{exp_data['total']}",
            "K": exp_data["num_few_shot"] if exp_data["setting"] == "Few-Shot" else "-"
        })
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    csv_file = output_dir / "results_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"✅ Saved CSV table to: {csv_file}")
    
    # Save as Markdown with better formatting
    md_content = "# HellaSwag ICL Experiment Results Summary\n\n"
    md_content += "| Exp | Model | Setting | ctx_a | Accuracy | Correct | K |\n"
    md_content += "|-----|-------|---------|-------|----------|---------|---|\n"
    
    for _, row in df.iterrows():
        md_content += f"| {row['Exp']} | {row['Model']} | {row['Setting']} | {row['ctx_a']} | "
        md_content += f"**{row['Accuracy']}** | {row['Correct']} | {row['K']} |\n"
    
    md_content += "\n## Key Findings\n\n"
    
    # Calculate key metrics
    base_zs = df[(df['Model'] == 'M_base') & (df['Setting'] == 'Zero-Shot')]['Accuracy'].values[0]
    base_gold = df[(df['Model'] == 'M_base') & (df['ctx_a'] == 'Gold')]['Accuracy'].values[0]
    base_random = df[(df['Model'] == 'M_base') & (df['ctx_a'] == 'Random')]['Accuracy'].values[0]
    finetuned_zs = df[(df['Model'] == 'M_finetuned') & (df['Setting'] == 'Zero-Shot')]['Accuracy'].values[0]
    finetuned_gold = df[(df['Model'] == 'M_finetuned') & (df['ctx_a'] == 'Gold')]['Accuracy'].values[0]
    finetuned_random = df[(df['Model'] == 'M_finetuned') & (df['ctx_a'] == 'Random')]['Accuracy'].values[0]
    
    md_content += f"- **Base Model Zero-Shot**: {base_zs}\n"
    md_content += f"- **Base Model Few-Shot (Gold)**: {base_gold} (+{float(base_gold.replace('%','')) - float(base_zs.replace('%','')):.2f}%)\n"
    md_content += f"- **Base Model Few-Shot (Random)**: {base_random} (+{float(base_random.replace('%','')) - float(base_zs.replace('%','')):.2f}%)\n"
    md_content += f"- **Finetuned Model Zero-Shot**: {finetuned_zs} (corrupted: {float(base_zs.replace('%','')) - float(finetuned_zs.replace('%','')):.2f}% degradation)\n"
    md_content += f"- **Finetuned Model Few-Shot (Gold)**: {finetuned_gold} (recovery: +{float(finetuned_gold.replace('%','')) - float(finetuned_zs.replace('%','')):.2f}%)\n"
    md_content += f"- **Finetuned Model Few-Shot (Random)**: {finetuned_random} (recovery: +{float(finetuned_random.replace('%','')) - float(finetuned_zs.replace('%','')):.2f}%)\n"
    
    md_file = output_dir / "results_summary.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"✅ Saved Markdown table to: {md_file}")
    
    return df


def plot_accuracy_comparison(results_dict, output_dir):
    """Plot comprehensive accuracy comparison."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    experiments = []
    accuracies = []
    colors_list = []
    
    for exp_key in sorted(results_dict.keys()):
        exp_data = results_dict[exp_key]
        experiments.append(exp_data["name"])
        accuracies.append(exp_data["accuracy"] * 100)
        colors_list.append(COLORS[exp_data["model"]])
    
    bars = ax.bar(experiments, accuracies, color=colors_list, alpha=0.85, 
                  edgecolor='black', linewidth=1.5, width=0.7)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add baseline lines
    base_zs = next((v["accuracy"]*100 for k, v in results_dict.items() 
                   if v["model"] == "M_base" and v["setting"] == "Zero-Shot"), None)
    if base_zs:
        ax.axhline(base_zs, color=COLORS['M_base'], linestyle='--', linewidth=2, 
                  alpha=0.5, label=f'M_base Zero-Shot Baseline ({base_zs:.2f}%)')
    
    finetuned_zs = next((v["accuracy"]*100 for k, v in results_dict.items() 
                        if v["model"] == "M_finetuned" and v["setting"] == "Zero-Shot"), None)
    if finetuned_zs:
        ax.axhline(finetuned_zs, color=COLORS['M_finetuned'], linestyle='--', linewidth=2, 
                  alpha=0.5, label=f'M_finetuned Zero-Shot Baseline ({finetuned_zs:.2f}%)')
    
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Experiment', fontsize=13, fontweight='bold')
    ax.set_title('HellaSwag ICL: Complete Experiment Results', fontsize=15, fontweight='bold', pad=20)
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
    output_file = output_dir / "accuracy_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved accuracy comparison chart to: {output_file}")
    plt.close()


def plot_grouped_comparison(results_dict, output_dir):
    """Plot grouped comparison: Base vs Finetuned by setting."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    settings = ["Zero-Shot", "Few-Shot\n(Gold ctx_a)", "Few-Shot\n(Random ctx_a)"]
    x = np.arange(len(settings))
    width = 0.35
    
    base_accs = []
    finetuned_accs = []
    
    # Zero-Shot
    base_zs = next((v["accuracy"]*100 for k, v in results_dict.items() 
                   if v["model"] == "M_base" and v["setting"] == "Zero-Shot"), 0)
    finetuned_zs = next((v["accuracy"]*100 for k, v in results_dict.items() 
                        if v["model"] == "M_finetuned" and v["setting"] == "Zero-Shot"), 0)
    base_accs.append(base_zs)
    finetuned_accs.append(finetuned_zs)
    
    # Few-Shot Gold
    base_gold = next((v["accuracy"]*100 for k, v in results_dict.items() 
                     if v["model"] == "M_base" and v["ctx_a_type"] == "Gold"), 0)
    finetuned_gold = next((v["accuracy"]*100 for k, v in results_dict.items() 
                          if v["model"] == "M_finetuned" and v["ctx_a_type"] == "Gold"), 0)
    base_accs.append(base_gold)
    finetuned_accs.append(finetuned_gold)
    
    # Few-Shot Random
    base_random = next((v["accuracy"]*100 for k, v in results_dict.items() 
                       if v["model"] == "M_base" and v["ctx_a_type"] == "Random"), 0)
    finetuned_random = next((v["accuracy"]*100 for k, v in results_dict.items() 
                            if v["model"] == "M_finetuned" and v["ctx_a_type"] == "Random"), 0)
    base_accs.append(base_random)
    finetuned_accs.append(finetuned_random)
    
    bars1 = ax.bar(x - width/2, base_accs, width, label='M_base', 
                   color=COLORS['M_base'], alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, finetuned_accs, width, label='M_finetuned', 
                   color=COLORS['M_finetuned'], alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Experiment Setting', fontsize=13, fontweight='bold')
    ax.set_title('Model Comparison: Base vs Finetuned by Setting', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(settings, fontsize=11)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_ylim(0, max(max(base_accs), max(finetuned_accs)) * 1.15)
    
    plt.tight_layout()
    output_file = output_dir / "grouped_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved grouped comparison chart to: {output_file}")
    plt.close()


def plot_hypothesis_validation(results_dict, output_dir):
    """Plot hypothesis validation with all three hypotheses."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # H1: Common Sense Hypothesis (Base model)
    ax1 = axes[0]
    base_results = {k: v for k, v in results_dict.items() if v["model"] == "M_base"}
    
    settings_h1 = ["Zero-Shot", "Random\nctx_a", "Gold\nctx_a"]
    accuracies_h1 = [
        next((v["accuracy"]*100 for k, v in base_results.items() if v["setting"] == "Zero-Shot"), 0),
        next((v["accuracy"]*100 for k, v in base_results.items() if v["ctx_a_type"] == "Random"), 0),
        next((v["accuracy"]*100 for k, v in base_results.items() if v["ctx_a_type"] == "Gold"), 0)
    ]
    
    colors_h1 = [COLORS['Zero-Shot'], COLORS['Random'], COLORS['Gold']]
    bars1 = ax1.bar(settings_h1, accuracies_h1, color=colors_h1, alpha=0.85, 
                    edgecolor='black', linewidth=1.5)
    
    for bar, acc in zip(bars1, accuracies_h1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('H1: Common Sense Hypothesis\n(Gold > Random > Zero-Shot)', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylim(0, max(accuracies_h1) * 1.15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # H2: Format Primacy Hypothesis
    ax2 = axes[1]
    base_zs = accuracies_h1[0]
    base_random = accuracies_h1[1]
    improvement = base_random - base_zs
    
    bars2 = ax2.bar(['Zero-Shot', 'Random\nctx_a'], [base_zs, base_random], 
                    color=[COLORS['Zero-Shot'], COLORS['Random']], alpha=0.85,
                    edgecolor='black', linewidth=1.5)
    
    for bar, acc in zip(bars2, [base_zs, base_random]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.annotate(f'+{improvement:.1f}%', 
                xy=(1, base_random), xytext=(0.5, (base_zs + base_random)/2),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, fontweight='bold', color='green', ha='center')
    
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('H2: Format Primacy Hypothesis\n(Random ctx_a helps)', 
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylim(0, max(base_zs, base_random) * 1.2)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # H3: Corrupted Model Recovery
    ax3 = axes[2]
    finetuned_results = {k: v for k, v in results_dict.items() if v["model"] == "M_finetuned"}
    
    finetuned_zs = next((v["accuracy"]*100 for k, v in finetuned_results.items() 
                        if v["setting"] == "Zero-Shot"), 0)
    finetuned_gold = next((v["accuracy"]*100 for k, v in finetuned_results.items() 
                          if v["ctx_a_type"] == "Gold"), 0)
    finetuned_random = next((v["accuracy"]*100 for k, v in finetuned_results.items() 
                            if v["ctx_a_type"] == "Random"), 0)
    
    recovery_gold = finetuned_gold - finetuned_zs
    recovery_random = finetuned_random - finetuned_zs
    
    categories_h3 = ['Zero-Shot', 'Gold\nctx_a', 'Random\nctx_a']
    accuracies_h3 = [finetuned_zs, finetuned_gold, finetuned_random]
    colors_h3 = [COLORS['Zero-Shot'], COLORS['Gold'], COLORS['Random']]
    
    bars3 = ax3.bar(categories_h3, accuracies_h3, color=colors_h3, alpha=0.85,
                    edgecolor='black', linewidth=1.5)
    
    for bar, acc in zip(bars3, accuracies_h3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add recovery annotations
    if recovery_gold > 0:
        ax3.annotate(f'+{recovery_gold:.1f}%', 
                    xy=(1, finetuned_gold), xytext=(0.5, finetuned_zs + recovery_gold/2),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=10, fontweight='bold', color='green', ha='center')
    
    ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('H3: Corrupted Model Recovery\n(ICL helps finetuned model)', 
                  fontsize=13, fontweight='bold', pad=15)
    ax3.set_ylim(0, max(accuracies_h3) * 1.2)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('Hypothesis Validation', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / "hypothesis_validation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved hypothesis validation chart to: {output_file}")
    plt.close()


def plot_confidence_distribution(results_dict, output_dir):
    """Plot confidence distribution for each experiment."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (exp_key, exp_data) in enumerate(sorted(results_dict.items())):
        if idx >= 6:
            break
        
        ax = axes[idx]
        results = exp_data.get("results", [])
        
        if results:
            max_probs = []
            for result in results:
                if "choice_probabilities" in result:
                    probs = result["choice_probabilities"]
                    max_prob = max(probs.values())
                    max_probs.append(max_prob * 100)
            
            if max_probs:
                color = COLORS[exp_data["model"]]
                ax.hist(max_probs, bins=30, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
                mean_prob = np.mean(max_probs)
                ax.axvline(mean_prob, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_prob:.1f}%')
                ax.set_xlabel('Max Probability (%)', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.set_title(f'{exp_data["full_name"]}\n{exp_data["description"]}', 
                            fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(alpha=0.3, linestyle='--')
    
    plt.suptitle('Confidence Distribution Across All Experiments', 
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / "confidence_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved confidence distribution chart to: {output_file}")
    plt.close()


def plot_improvement_analysis(results_dict, output_dir):
    """Plot improvement analysis: Few-Shot improvements for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Base model improvements
    ax1 = axes[0]
    base_zs = next((v["accuracy"]*100 for k, v in results_dict.items() 
                   if v["model"] == "M_base" and v["setting"] == "Zero-Shot"), None)
    base_gold = next((v["accuracy"]*100 for k, v in results_dict.items() 
                     if v["model"] == "M_base" and v["ctx_a_type"] == "Gold"), None)
    base_random = next((v["accuracy"]*100 for k, v in results_dict.items() 
                       if v["model"] == "M_base" and v["ctx_a_type"] == "Random"), None)
    
    if base_zs is not None:
        improvements = []
        labels = []
        colors_imp = []
        
        if base_gold is not None:
            improvements.append(base_gold - base_zs)
            labels.append("Gold ctx_a")
            colors_imp.append(COLORS['Gold'])
        if base_random is not None:
            improvements.append(base_random - base_zs)
            labels.append("Random ctx_a")
            colors_imp.append(COLORS['Random'])
        
        if improvements:
            bars = ax1.bar(labels, improvements, color=colors_imp, alpha=0.85,
                          edgecolor='black', linewidth=1.5)
            
            for bar, imp in zip(bars, improvements):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1 if imp > 0 else height - 0.3,
                        f'{imp:+.2f}%',
                        ha='center', va='bottom' if imp > 0 else 'top', 
                        fontsize=11, fontweight='bold')
            
            ax1.axhline(0, color='black', linestyle='-', linewidth=1)
            ax1.set_ylabel('Accuracy Improvement (%)', fontsize=12, fontweight='bold')
            ax1.set_title('M_base: Few-Shot Improvements', fontsize=13, fontweight='bold', pad=15)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Finetuned model recovery
    ax2 = axes[1]
    finetuned_zs = next((v["accuracy"]*100 for k, v in results_dict.items() 
                        if v["model"] == "M_finetuned" and v["setting"] == "Zero-Shot"), None)
    finetuned_gold = next((v["accuracy"]*100 for k, v in results_dict.items() 
                          if v["model"] == "M_finetuned" and v["ctx_a_type"] == "Gold"), None)
    finetuned_random = next((v["accuracy"]*100 for k, v in results_dict.items() 
                            if v["model"] == "M_finetuned" and v["ctx_a_type"] == "Random"), None)
    
    if finetuned_zs is not None:
        recoveries = []
        labels_rec = []
        colors_rec = []
        
        if finetuned_gold is not None:
            recoveries.append(finetuned_gold - finetuned_zs)
            labels_rec.append("Gold ctx_a")
            colors_rec.append(COLORS['Gold'])
        if finetuned_random is not None:
            recoveries.append(finetuned_random - finetuned_zs)
            labels_rec.append("Random ctx_a")
            colors_rec.append(COLORS['Random'])
        
        if recoveries:
            bars = ax2.bar(labels_rec, recoveries, color=colors_rec, alpha=0.85,
                          edgecolor='black', linewidth=1.5)
            
            for bar, rec in zip(bars, recoveries):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1 if rec > 0 else height - 0.3,
                        f'{rec:+.2f}%',
                        ha='center', va='bottom' if rec > 0 else 'top',
                        fontsize=11, fontweight='bold')
            
            ax2.axhline(0, color='black', linestyle='-', linewidth=1)
            ax2.set_ylabel('Recovery (%)', fontsize=12, fontweight='bold')
            ax2.set_title('M_finetuned: ICL Recovery from Corruption', 
                         fontsize=13, fontweight='bold', pad=15)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('Improvement Analysis: Few-Shot Benefits', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_dir / "improvement_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved improvement analysis chart to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize HellaSwag ICL experiment results")
    parser.add_argument("--results_dir", type=str, default="/data/johnwang/ICL/result",
                       help="Directory containing result JSON files")
    parser.add_argument("--output_dir", type=str, default="/data/johnwang/ICL/result",
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("HellaSwag ICL Results Visualization")
    print("="*70)
    print()
    
    # Load results
    results_dict = load_results(results_dir)
    
    if not results_dict:
        print("❌ No results found!")
        return
    
    print()
    
    # Create summary table
    summary_df = create_summary_table(results_dict, output_dir)
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    plot_accuracy_comparison(results_dict, output_dir)
    plot_grouped_comparison(results_dict, output_dir)
    plot_hypothesis_validation(results_dict, output_dir)
    plot_confidence_distribution(results_dict, output_dir)
    plot_improvement_analysis(results_dict, output_dir)
    
    print()
    print("="*70)
    print("✅ All visualizations completed!")
    print(f"📊 Results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
