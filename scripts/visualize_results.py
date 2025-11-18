#!/usr/bin/env python3
"""
Visualize HellaSwag ICL experiment results with fancy charts and tables.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results(results_dir):
    """Load all experiment results."""
    results_dir = Path(results_dir)
    
    experiments = {
        "exp_01_a": {
            "file": "exp_01_a_zeroshot.json",
            "name": "Experiment A",
            "description": "M_base Zero-Shot",
            "model": "M_base",
            "setting": "Zero-Shot",
            "ctx_a_type": "None"
        },
        "exp_02_b": {
            "file": "exp_02_b_fewshot_gold.json",
            "name": "Experiment B",
            "description": "M_base Few-Shot (Gold ctx_a)",
            "model": "M_base",
            "setting": "Few-Shot",
            "ctx_a_type": "Gold"
        },
        "exp_03_c": {
            "file": "exp_03_c_fewshot_random.json",
            "name": "Experiment C",
            "description": "M_base Few-Shot (Random ctx_a)",
            "model": "M_base",
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
            print(f"✅ Loaded {exp_info['name']}: {loaded_results[exp_key]['accuracy']:.4f}")
        else:
            print(f"⚠️  File not found: {filepath}")
    
    return loaded_results


def create_summary_table(results_dict, output_file):
    """Create a summary table."""
    rows = []
    for exp_key, exp_data in results_dict.items():
        rows.append({
            "Experiment": exp_data["name"],
            "Model": exp_data["model"],
            "Setting": exp_data["setting"],
            "ctx_a Type": exp_data["ctx_a_type"],
            "Accuracy": f"{exp_data['accuracy']:.4f}",
            "Accuracy (%)": f"{exp_data['accuracy']*100:.2f}%",
            "Correct": exp_data["correct"],
            "Total": exp_data["total"],
            "Few-Shot K": exp_data["num_few_shot"] if exp_data["setting"] == "Few-Shot" else "-"
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("Experiment")
    
    # Save as CSV
    csv_file = output_file.replace('.md', '.csv')
    df.to_csv(csv_file, index=False)
    print(f"✅ Saved CSV table to: {csv_file}")
    
    # Save as Markdown
    md_content = "# Experiment Results Summary\n\n"
    md_content += df.to_markdown(index=False)
    md_content += "\n\n"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"✅ Saved Markdown table to: {output_file}")
    
    return df


def plot_accuracy_comparison(results_dict, output_dir):
    """Plot accuracy comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    experiments = []
    accuracies = []
    colors = []
    
    for exp_key in sorted(results_dict.keys()):
        exp_data = results_dict[exp_key]
        experiments.append(exp_data["name"])
        accuracies.append(exp_data["accuracy"] * 100)
        
        # Color coding: base model = blue, finetuned = red
        if exp_data["model"] == "M_base":
            colors.append("#3498db")
        else:
            colors.append("#e74c3c")
    
    bars = ax.bar(experiments, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_title('HellaSwag ICL Experiment Results: Accuracy Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(accuracies) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='M_base'),
        Patch(facecolor='#e74c3c', label='M_finetuned')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_file = output_dir / "accuracy_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved accuracy comparison chart to: {output_file}")
    plt.close()


def plot_grouped_comparison(results_dict, output_dir):
    """Plot grouped comparison by setting (Base model only)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Only show base model results
    base_results = {k: v for k, v in results_dict.items() if v["model"] == "M_base"}
    
    if not base_results:
        print("⚠️  No base model data for grouped comparison")
        return
    
    # Group data by setting type
    settings_order = ["Zero-Shot", "Few-Shot (Random ctx_a)", "Few-Shot (Gold ctx_a)"]
    
    accuracies = []
    labels = []
    
    for setting in settings_order:
        if setting == "Zero-Shot":
            acc = next((v["accuracy"]*100 for k, v in base_results.items() 
                       if v["setting"] == "Zero-Shot"), None)
        elif setting == "Few-Shot (Random ctx_a)":
            acc = next((v["accuracy"]*100 for k, v in base_results.items() 
                       if v["ctx_a_type"] == "Random"), None)
        else:  # Few-Shot (Gold ctx_a)
            acc = next((v["accuracy"]*100 for k, v in base_results.items() 
                       if v["ctx_a_type"] == "Gold"), None)
        
        if acc is not None:
            labels.append(setting)
            accuracies.append(acc)
    
    if not labels:
        print("⚠️  No data for grouped comparison")
        return
    
    x = np.arange(len(labels))
    
    bars = ax.bar(x, accuracies, color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Experiment Setting', fontsize=12, fontweight='bold')
    ax.set_title('M_base Model: Accuracy by Setting', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_file = output_dir / "grouped_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved grouped comparison chart to: {output_file}")
    plt.close()


def plot_hypothesis_validation(results_dict, output_dir):
    """Plot hypothesis validation visualization (Base model only)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # H1: Common Sense Hypothesis
    # Gold ctx_a > Zero-Shot > Random ctx_a
    base_results = {k: v for k, v in results_dict.items() if v["model"] == "M_base"}
    
    if len(base_results) >= 3:
        settings = ["Zero-Shot", "Random ctx_a", "Gold ctx_a"]
        accuracies = []
        
        for setting in settings:
            if setting == "Zero-Shot":
                acc = next((v["accuracy"]*100 for k, v in base_results.items() if v["setting"] == "Zero-Shot"), 0)
            elif setting == "Random ctx_a":
                acc = next((v["accuracy"]*100 for k, v in base_results.items() if v["ctx_a_type"] == "Random"), 0)
            else:
                acc = next((v["accuracy"]*100 for k, v in base_results.items() if v["ctx_a_type"] == "Gold"), 0)
            accuracies.append(acc)
        
        bars = ax.bar(settings, accuracies, color=['#95a5a6', '#f39c12', '#27ae60'], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Setting', fontsize=12, fontweight='bold')
        ax.set_title('H1: Common Sense Hypothesis\n(Gold ctx_a > Random ctx_a > Zero-Shot)', 
                      fontsize=13, fontweight='bold', pad=15)
        ax.set_ylim(0, max(accuracies) * 1.15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_file = output_dir / "hypothesis_validation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved hypothesis validation chart to: {output_file}")
    plt.close()


def plot_confidence_distribution(results_dict, output_dir):
    """Plot confidence (max probability) distribution for each experiment."""
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
                ax.hist(max_probs, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
                ax.axvline(np.mean(max_probs), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(max_probs):.2f}%')
                ax.set_xlabel('Max Probability (%)', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.set_title(f'{exp_data["name"]}\nConfidence Distribution', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(results_dict), 6):
        axes[idx].axis('off')
    
    plt.suptitle('Confidence Distribution Across Experiments', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / "confidence_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved confidence distribution chart to: {output_file}")
    plt.close()


def plot_improvement_analysis(results_dict, output_dir):
    """Plot improvement analysis: Few-Shot vs Zero-Shot (Base model only)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    improvements = []
    labels = []
    
    # Base model improvements only
    base_zeroshot = next((v["accuracy"]*100 for k, v in results_dict.items() 
                          if v["model"] == "M_base" and v["setting"] == "Zero-Shot"), None)
    base_gold = next((v["accuracy"]*100 for k, v in results_dict.items() 
                      if v["model"] == "M_base" and v["ctx_a_type"] == "Gold"), None)
    base_random = next((v["accuracy"]*100 for k, v in results_dict.items() 
                        if v["model"] == "M_base" and v["ctx_a_type"] == "Random"), None)
    
    if base_zeroshot is not None:
        if base_gold is not None:
            improvements.append(base_gold - base_zeroshot)
            labels.append("Gold ctx_a\nvs Zero-Shot")
        if base_random is not None:
            improvements.append(base_random - base_zeroshot)
            labels.append("Random ctx_a\nvs Zero-Shot")
    
    if improvements:
        colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
        bars = ax.bar(labels, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:+.2f}%',
                    ha='center', va='bottom' if imp > 0 else 'top', fontsize=10, fontweight='bold')
        
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel('Accuracy Improvement (%)', fontsize=12, fontweight='bold')
        ax.set_title('M_base Model: Few-Shot vs Zero-Shot Improvement', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.xticks(rotation=45, ha='right')
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
    summary_df = create_summary_table(results_dict, str(output_dir / "results_summary.md"))
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

