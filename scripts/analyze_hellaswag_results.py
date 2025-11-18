#!/usr/bin/env python3
"""
Result Analysis Script for HellaSwag ICL Experiments

This script analyzes results from all 5 experiments and tests the hypotheses:
- H1: Semantic Primacy Hypothesis
- H2: Format Primacy Hypothesis  
- H3: Context Primacy Hypothesis
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def load_results(results_dir: str) -> Dict[str, Dict]:
    """
    Load results from all 5 experiments.
    
    Args:
        results_dir: Directory containing result JSON files
        
    Returns:
        Dictionary mapping experiment names to their results
    """
    results_dir = Path(results_dir)
    experiments = {
        "exp_01_a": "exp_01_a_zeroshot.json",
        "exp_02_b": "exp_02_b_fewshot_gold.json",
        "exp_03_c": "exp_03_c_fewshot_random.json",
        "exp_04_d": "exp_04_d_finetuned_zeroshot.json",
        "exp_05_e": "exp_05_e_finetuned_fewshot_gold.json",
    }
    
    loaded_results = {}
    for exp_name, filename in experiments.items():
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_results[exp_name] = json.load(f)
            print(f"✅ Loaded {exp_name} from {filename}")
        else:
            print(f"⚠️  File not found: {filepath}")
    
    return loaded_results


def calculate_statistics(results: Dict) -> Dict:
    """
    Calculate statistics from experiment results.
    
    Args:
        results: Experiment results dictionary
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        "accuracy": results.get("accuracy", 0.0),
        "correct": results.get("correct", 0),
        "total": results.get("total", 0),
    }
    
    # Calculate confidence statistics if available
    if "results" in results:
        choice_probs_list = []
        log_probs_list = []
        
        for sample in results["results"]:
            if "choice_probabilities" in sample:
                probs = sample["choice_probabilities"]
                choice_probs_list.append([
                    probs.get("A", 0.0),
                    probs.get("B", 0.0),
                    probs.get("C", 0.0),
                    probs.get("D", 0.0),
                ])
            
            if "choice_log_probabilities" in sample:
                log_probs = sample["choice_log_probabilities"]
                log_probs_list.append([
                    log_probs.get("A", float('-inf')),
                    log_probs.get("B", float('-inf')),
                    log_probs.get("C", float('-inf')),
                    log_probs.get("D", float('-inf')),
                ])
        
        if choice_probs_list:
            choice_probs_array = np.array(choice_probs_list)
            stats["mean_max_prob"] = float(np.mean(np.max(choice_probs_array, axis=1)))
            stats["std_max_prob"] = float(np.std(np.max(choice_probs_array, axis=1)))
        
        if log_probs_list:
            log_probs_array = np.array(log_probs_list)
            stats["mean_max_log_prob"] = float(np.mean(np.max(log_probs_array, axis=1)))
            stats["std_max_log_prob"] = float(np.std(np.max(log_probs_array, axis=1)))
    
    return stats


def test_hypotheses(results_dict: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Test the three hypotheses based on experiment results.
    
    Args:
        results_dict: Dictionary of experiment results
        
    Returns:
        Dictionary with hypothesis test results
    """
    hypotheses = {}
    
    # Extract accuracies
    acc_01 = results_dict.get("exp_01_a", {}).get("accuracy", 0.0)  # M_base Zero-Shot
    acc_02 = results_dict.get("exp_02_b", {}).get("accuracy", 0.0)  # M_base Few-Shot Gold
    acc_03 = results_dict.get("exp_03_c", {}).get("accuracy", 0.0)  # M_base Few-Shot Random
    acc_04 = results_dict.get("exp_04_d", {}).get("accuracy", 0.0)  # M_finetuned Zero-Shot
    acc_05 = results_dict.get("exp_05_e", {}).get("accuracy", 0.0)  # M_finetuned Few-Shot Gold
    
    # H1: Semantic Primacy Hypothesis
    # Prediction: Accuracy(Gold) > Accuracy(Zero-Shot) > Accuracy(Random)
    if acc_01 > 0 and acc_02 > 0 and acc_03 > 0:
        h1_gold_vs_zero = acc_02 > acc_01
        h1_zero_vs_random = acc_01 > acc_03
        h1_supported = h1_gold_vs_zero and h1_zero_vs_random
        
        hypotheses["H1_Semantic_Primacy"] = {
            "prediction": "Accuracy(Gold) > Accuracy(Zero-Shot) > Accuracy(Random)",
            "results": {
                "Accuracy(Gold)": acc_02,
                "Accuracy(Zero-Shot)": acc_01,
                "Accuracy(Random)": acc_03,
            },
            "tests": {
                "Gold > Zero-Shot": h1_gold_vs_zero,
                "Zero-Shot > Random": h1_zero_vs_random,
            },
            "supported": h1_supported,
        }
    
    # H2: Format Primacy Hypothesis
    # Prediction: Accuracy(Gold) ≈ Accuracy(Random), and both > Accuracy(Zero-Shot)
    if acc_01 > 0 and acc_02 > 0 and acc_03 > 0:
        h2_gold_vs_random_similar = abs(acc_02 - acc_03) < 0.05  # Within 5%
        h2_both_gt_zero = acc_02 > acc_01 and acc_03 > acc_01
        h2_supported = h2_gold_vs_random_similar and h2_both_gt_zero
        
        hypotheses["H2_Format_Primacy"] = {
            "prediction": "Accuracy(Gold) ≈ Accuracy(Random), and both > Accuracy(Zero-Shot)",
            "results": {
                "Accuracy(Gold)": acc_02,
                "Accuracy(Random)": acc_03,
                "Accuracy(Zero-Shot)": acc_01,
            },
            "tests": {
                "Gold ≈ Random (diff < 5%)": h2_gold_vs_random_similar,
                "Gold > Zero-Shot": acc_02 > acc_01,
                "Random > Zero-Shot": acc_03 > acc_01,
            },
            "supported": h2_supported,
        }
    
    # H3: Context Primacy Hypothesis
    # Prediction: M_finetuned performs poorly on Zero-Shot, but can recover with Gold ctx_a
    if acc_04 > 0 and acc_05 > 0 and acc_01 > 0:
        h3_finetuned_poor = acc_04 < acc_01  # Finetuned zero-shot worse than base zero-shot
        h3_can_recover = acc_05 > acc_04  # Finetuned few-shot better than finetuned zero-shot
        h3_supported = h3_finetuned_poor and h3_can_recover
        
        hypotheses["H3_Context_Primacy"] = {
            "prediction": "M_finetuned performs poorly on Zero-Shot, but can recover with Gold ctx_a",
            "results": {
                "M_finetuned Zero-Shot": acc_04,
                "M_finetuned Few-Shot Gold": acc_05,
                "M_base Zero-Shot (baseline)": acc_01,
            },
            "tests": {
                "Finetuned Zero-Shot < Base Zero-Shot": h3_finetuned_poor,
                "Finetuned Few-Shot > Finetuned Zero-Shot": h3_can_recover,
            },
            "supported": h3_supported,
        }
    
    return hypotheses


def print_summary(results_dict: Dict[str, Dict], hypotheses: Dict[str, Dict]):
    """
    Print a summary of results and hypothesis tests.
    
    Args:
        results_dict: Dictionary of experiment results
        hypotheses: Dictionary of hypothesis test results
    """
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*70)
    
    exp_names = {
        "exp_01_a": "Experiment 01: M_base Zero-Shot",
        "exp_02_b": "Experiment 02: M_base Few-Shot (Gold ctx_a)",
        "exp_03_c": "Experiment 03: M_base Few-Shot (Random ctx_a)",
        "exp_04_d": "Experiment 04: M_finetuned Zero-Shot",
        "exp_05_e": "Experiment 05: M_finetuned Few-Shot (Gold ctx_a)",
    }
    
    for exp_key, exp_name in exp_names.items():
        if exp_key in results_dict:
            stats = calculate_statistics(results_dict[exp_key])
            print(f"\n{exp_name}:")
            print(f"  Accuracy: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
            if "mean_max_prob" in stats:
                print(f"  Mean Max Probability: {stats['mean_max_prob']:.4f} ± {stats['std_max_prob']:.4f}")
    
    print("\n" + "="*70)
    print("HYPOTHESIS TESTS")
    print("="*70)
    
    for hyp_name, hyp_data in hypotheses.items():
        print(f"\n{hyp_name}:")
        print(f"  Prediction: {hyp_data['prediction']}")
        print(f"  Results:")
        for key, value in hyp_data['results'].items():
            print(f"    {key}: {value:.4f}")
        print(f"  Tests:")
        for test_name, test_result in hyp_data['tests'].items():
            status = "✅" if test_result else "❌"
            print(f"    {status} {test_name}: {test_result}")
        print(f"  Overall: {'✅ SUPPORTED' if hyp_data['supported'] else '❌ NOT SUPPORTED'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze HellaSwag ICL experiment results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/hellaswag",
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional: Save analysis results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    results_dict = load_results(args.results_dir)
    
    if not results_dict:
        print("❌ No results found!")
        return
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats_dict = {}
    for exp_name, results in results_dict.items():
        stats_dict[exp_name] = calculate_statistics(results)
    
    # Test hypotheses
    print("\nTesting hypotheses...")
    hypotheses = test_hypotheses(results_dict)
    
    # Print summary
    print_summary(results_dict, hypotheses)
    
    # Save analysis if requested
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        analysis_results = {
            "statistics": stats_dict,
            "hypotheses": hypotheses,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Analysis saved to: {output_path}")


if __name__ == "__main__":
    main()

