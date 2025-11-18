#!/usr/bin/env python3
"""
Experiment 04: M_finetuned Zero-Shot

Description: Finetuned model (corrupted), zero-shot test to prove corruption
Model: M_finetuned (Qwen3-1.7B finetuned on Random ctx_a dataset)
Context: Zero-Shot (only ctx_b, no ctx_a)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from utils import format_hellaswag_example_zeroshot, extract_answer_from_logits


def run_experiment_d(
    finetuned_model_path: str,
    gold_data_path: str,
    max_samples: int = None,
    output_file: str = None,
):
    """Run Experiment D: M_finetuned Zero-Shot"""
    print(f"\n{'='*70}")
    print(f"Experiment 04: M_finetuned Zero-Shot")
    print(f"{'='*70}\n")
    
    # Load model
    print(f"Loading finetuned model from: {finetuned_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        finetuned_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("✅ Model loaded successfully\n")
    
    # Load dataset
    print("Loading gold dataset...")
    with open(gold_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    print(f"✅ Test dataset: {len(test_data)} samples\n")
    
    # Evaluation
    correct = 0
    total = 0
    results = []
    
    print("Starting evaluation...")
    with torch.no_grad():
        for idx, example in enumerate(tqdm(test_data, desc="Experiment 04")):
            # Build prompt (zero-shot: only ctx_b)
            prompt = format_hellaswag_example_zeroshot(example)
            
            # Extract answer using log probability method (standard for HellaSwag)
            predicted_answer, choice_probs, log_probs = extract_answer_from_logits(
                model, tokenizer, prompt
            )
            correct_answer = chr(ord('A') + int(example.get("label", "0")))
            
            # Check correctness
            is_correct = (predicted_answer == correct_answer)
            if predicted_answer:
                correct += is_correct
            total += 1
            
            # Store result
            results.append({
                "index": idx,
                "ind": example.get("ind", idx),
                "prompt": prompt,
                "predicted_answer": predicted_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "choice_probabilities": choice_probs,
                "choice_log_probabilities": log_probs,
                "activity_label": example.get("activity_label", ""),
                "ctx_b": example.get("ctx_b", ""),
            })
            
            # Print first few examples
            if idx < 3:
                print(f"\n--- Example {idx+1} ---")
                print(f"Prompt: ...{prompt[-150:]}")
                print(f"Predicted: {predicted_answer}, Correct: {correct_answer}, Match: {is_correct}")
                print(f"Probabilities: {choice_probs}")
                print(f"Log Probabilities: {log_probs}")
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{'='*70}")
    print(f"Experiment 04 Results:")
    print(f"{'='*70}")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*70}\n")
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            "experiment": "04_D",
            "description": "M_finetuned Zero-Shot",
            "model_path": finetuned_model_path,
            "total_samples": total,
            "correct": correct,
            "accuracy": accuracy,
            "results": results,
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to: {output_file}")
    
    return accuracy, results


def main():
    parser = argparse.ArgumentParser(description="Experiment 04: M_finetuned Zero-Shot")
    parser.add_argument("--finetuned_model_path", type=str, required=True, help="Path to finetuned model")
    parser.add_argument("--gold_data_path", type=str, required=True, help="Path to gold dataset JSON")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save results")
    
    args = parser.parse_args()
    run_experiment_d(**vars(args))


if __name__ == "__main__":
    main()

