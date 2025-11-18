#!/usr/bin/env python3
"""
Experiment 02: M_base Few-Shot with Gold ctx_a

Description: Base model with few-shot examples using gold ctx_a
Model: M_base (Qwen3-1.7B)
Context: Few-Shot (Gold ctx_a)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from utils import create_few_shot_prompt, extract_answer


def run_experiment_b(
    model_path: str,
    gold_data_path: str,
    num_few_shot: int = 5,
    max_samples: int = None,
    output_file: str = None,
):
    """Run Experiment B: M_base Few-Shot with Gold ctx_a"""
    print(f"\n{'='*70}")
    print(f"Experiment 02: M_base Few-Shot with Gold ctx_a")
    print(f"{'='*70}\n")
    
    # Load model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("✅ Model loaded successfully\n")
    
    # Load dataset
    print("Loading gold dataset...")
    with open(gold_data_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    
    test_data = gold_data
    few_shot_examples = gold_data[:num_few_shot]
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    print(f"✅ Test dataset: {len(test_data)} samples")
    print(f"✅ Few-shot examples: {len(few_shot_examples)} examples\n")
    
    # Evaluation
    correct = 0
    total = 0
    results = []
    
    print("Starting evaluation...")
    with torch.no_grad():
        for idx, example in enumerate(tqdm(test_data, desc="Experiment 02")):
            # Build prompt (few-shot with gold ctx_a)
            prompt = create_few_shot_prompt(few_shot_examples, example, ctx_a_type="gold")
            
            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Extract answer
            predicted_answer = extract_answer(generated_text)
            correct_answer = chr(ord('A') + int(example.get("label", "0")))
            
            # Check correctness
            is_correct = (predicted_answer == correct_answer) if predicted_answer else False
            if predicted_answer:
                correct += is_correct
            total += 1
            
            # Store result
            results.append({
                "index": idx,
                "ind": example.get("ind", idx),
                "prompt": prompt,
                "generated_text": generated_text,
                "predicted_answer": predicted_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "activity_label": example.get("activity_label", ""),
                "ctx_a": example.get("ctx_a", ""),
                "ctx_b": example.get("ctx_b", ""),
            })
            
            # Print first few examples
            if idx < 3:
                print(f"\n--- Example {idx+1} ---")
                print(f"Prompt (last 200 chars): ...{prompt[-200:]}")
                print(f"Generated: {generated_text}")
                print(f"Predicted: {predicted_answer}, Correct: {correct_answer}, Match: {is_correct}")
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{'='*70}")
    print(f"Experiment 02 Results:")
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
            "experiment": "02_B",
            "description": "M_base Few-Shot with Gold ctx_a",
            "model_path": model_path,
            "total_samples": total,
            "correct": correct,
            "accuracy": accuracy,
            "num_few_shot": num_few_shot,
            "results": results,
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to: {output_file}")
    
    return accuracy, results


def main():
    parser = argparse.ArgumentParser(description="Experiment 02: M_base Few-Shot with Gold ctx_a")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--gold_data_path", type=str, required=True, help="Path to gold dataset JSON")
    parser.add_argument("--num_few_shot", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save results")
    
    args = parser.parse_args()
    run_experiment_b(**vars(args))


if __name__ == "__main__":
    main()

