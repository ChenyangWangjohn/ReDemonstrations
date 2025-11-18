#!/usr/bin/env python3
"""
ICL Evaluation Script for HellaSwag Dataset

Supports 5 experiments:
- A: M_base Zero-Shot (no ctx_a)
- B: M_base Few-Shot with Gold ctx_a
- C: M_base Few-Shot with Random ctx_a
- D: M_finetuned Zero-Shot
- E: M_finetuned Few-Shot with Gold ctx_a
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import re
import argparse
from pathlib import Path


def format_hellaswag_example_zeroshot(example):
    """
    Format Zero-Shot example (only ctx_b, no ctx_a).
    
    Args:
        example: Dictionary with 'ctx_b' and 'endings'
        
    Returns:
        Formatted prompt string
    """
    ctx_b = example.get("ctx_b", "")
    endings = example.get("endings", [])
    
    prompt = f"{ctx_b}\n"
    prompt += f"A) {endings[0]}\n"
    prompt += f"B) {endings[1]}\n"
    prompt += f"C) {endings[2]}\n"
    prompt += f"D) {endings[3]}\n"
    prompt += "Answer:"
    
    return prompt


def create_few_shot_prompt(examples, test_example, ctx_a_type="gold"):
    """
    Create Few-Shot Prompt with examples.
    
    Args:
        examples: List of few-shot examples (from gold or random dataset)
        test_example: Test sample to evaluate
        ctx_a_type: "gold" or "random", determines which ctx_a to use
        
    Returns:
        Formatted prompt string
    """
    prompt = ""
    
    # Add few-shot examples
    for i, ex in enumerate(examples):
        # Answer is always correct (label unchanged)
        answer = chr(ord('A') + int(ex.get("label", "0")))
        
        # Build context (ctx_a + ctx_b)
        ctx = ex.get("ctx", "")
        if not ctx:
            # Fallback: combine ctx_a and ctx_b
            ctx_a = ex.get("ctx_a", "")
            ctx_b = ex.get("ctx_b", "")
            ctx = f"{ctx_a} {ctx_b}".strip()
        
        prompt += f"Example {i+1}:\n"
        prompt += f"{ctx}\n"
        prompt += f"A) {ex['endings'][0]}\n"
        prompt += f"B) {ex['endings'][1]}\n"
        prompt += f"C) {ex['endings'][2]}\n"
        prompt += f"D) {ex['endings'][3]}\n"
        prompt += f"Answer: {answer}\n\n"
    
    # Add test question
    test_ctx = test_example.get("ctx", "")
    if not test_ctx:
        # Fallback: combine ctx_a and ctx_b
        test_ctx_a = test_example.get("ctx_a", "")
        test_ctx_b = test_example.get("ctx_b", "")
        test_ctx = f"{test_ctx_a} {test_ctx_b}".strip()
    
    prompt += "Question:\n"
    prompt += f"{test_ctx}\n"
    prompt += f"A) {test_example['endings'][0]}\n"
    prompt += f"B) {test_example['endings'][1]}\n"
    prompt += f"C) {test_example['endings'][2]}\n"
    prompt += f"D) {test_example['endings'][3]}\n"
    prompt += "Answer:"
    
    return prompt


def extract_answer(text):
    """
    Extract answer (A, B, C, or D) from model's generated text.
    
    Args:
        text: Generated text from model
        
    Returns:
        Answer letter (A, B, C, or D) or None if not found
    """
    if not text:
        return None
    
    # Remove whitespace
    text = text.strip()
    
    # Try to find single letter A-D at the start
    match = re.match(r'^([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Try to find letter in first few characters
    match = re.search(r'\b([A-D])\b', text[:20], re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Try to find "Answer: A" pattern
    match = re.search(r'Answer:\s*([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Try to find letter in last few characters
    match = re.search(r'\b([A-D])\b', text[-20:], re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return None


def evaluate_hellaswag_icl(
    model_path: str,
    experiment: str,
    gold_data_path: str,
    random_data_path: str,
    finetuned_model_path: str = None,
    num_few_shot: int = 5,
    max_samples: int = None,
    output_file: str = None,
    batch_size: int = 1,  # Process one at a time for now
):
    """
    Evaluate HellaSwag dataset with ICL.
    
    Args:
        model_path: Path to base model (Qwen3-1.7B)
        experiment: Experiment ID ("A", "B", "C", "D", or "E")
        gold_data_path: Path to gold dataset JSON
        random_data_path: Path to random dataset JSON
        finetuned_model_path: Path to finetuned model (for experiments D & E)
        num_few_shot: Number of few-shot examples
        max_samples: Maximum number of samples to evaluate (None = all)
        output_file: Path to save results JSON
        batch_size: Batch size for inference (currently 1)
    """
    print(f"\n{'='*70}")
    print(f"Experiment {experiment}: {'M_base' if experiment in ['A', 'B', 'C'] else 'M_finetuned'}")
    print(f"{'='*70}\n")
    
    # Determine which model to use
    if experiment in ["D", "E"]:
        if finetuned_model_path is None:
            raise ValueError("finetuned_model_path is required for experiments D and E")
        model_path_to_use = finetuned_model_path
        print(f"Using finetuned model: {model_path_to_use}")
    else:
        model_path_to_use = model_path
        print(f"Using base model: {model_path_to_use}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path_to_use, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path_to_use,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("✅ Model loaded successfully\n")
    
    # Load datasets
    print("Loading datasets...")
    with open(gold_data_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    
    with open(random_data_path, 'r', encoding='utf-8') as f:
        random_data = json.load(f)
    
    print(f"✅ Gold dataset: {len(gold_data)} samples")
    print(f"✅ Random dataset: {len(random_data)} samples\n")
    
    # Determine test dataset and ctx_a type based on experiment
    if experiment == "A":  # M_base Zero-Shot
        test_data = gold_data
        use_ctx_a = False
        ctx_a_type = None
    elif experiment == "B":  # M_base Gold ctx_a
        test_data = gold_data
        use_ctx_a = True
        ctx_a_type = "gold"
    elif experiment == "C":  # M_base Random ctx_a
        test_data = random_data
        use_ctx_a = True
        ctx_a_type = "random"
    elif experiment == "D":  # M_finetuned Zero-Shot
        test_data = gold_data
        use_ctx_a = False
        ctx_a_type = None
    elif experiment == "E":  # M_finetuned Gold ctx_a
        test_data = gold_data
        use_ctx_a = True
        ctx_a_type = "gold"
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    print(f"Test dataset: {len(test_data)} samples")
    print(f"Use ctx_a: {use_ctx_a}")
    print(f"ctx_a type: {ctx_a_type}\n")
    
    # Prepare few-shot examples (if needed)
    few_shot_examples = None
    if use_ctx_a:
        if ctx_a_type == "gold":
            few_shot_examples = gold_data[:num_few_shot]
        else:  # random
            few_shot_examples = random_data[:num_few_shot]
        print(f"Few-shot examples: {len(few_shot_examples)} examples\n")
    
    # Evaluation
    correct = 0
    total = 0
    results = []
    
    print("Starting evaluation...")
    with torch.no_grad():
        for idx, example in enumerate(tqdm(test_data, desc=f"Experiment {experiment}")):
            # Build prompt
            if not use_ctx_a:
                # Zero-Shot: only ctx_b, no ctx_a
                prompt = format_hellaswag_example_zeroshot(example)
            else:
                # Few-Shot: ctx_a + ctx_b
                prompt = create_few_shot_prompt(
                    few_shot_examples, example, 
                    ctx_a_type=ctx_a_type
                )
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,  # Just need A, B, C, or D
                do_sample=False,  # Deterministic
                temperature=1.0,
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
            
            # Print first few examples for debugging
            if idx < 3:
                print(f"\n--- Example {idx+1} ---")
                print(f"Prompt (last 200 chars): ...{prompt[-200:]}")
                print(f"Generated: {generated_text}")
                print(f"Predicted: {predicted_answer}")
                print(f"Correct: {correct_answer}")
                print(f"Match: {is_correct}")
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{'='*70}")
    print(f"Experiment {experiment} Results:")
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
            "experiment": experiment,
            "model_path": model_path_to_use,
            "total_samples": total,
            "correct": correct,
            "accuracy": accuracy,
            "num_few_shot": num_few_shot if use_ctx_a else 0,
            "ctx_a_type": ctx_a_type,
            "results": results,
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to: {output_file}")
    
    return accuracy, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate HellaSwag with ICL")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to base model (Qwen3-1.7B)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["A", "B", "C", "D", "E"],
        help="Experiment ID: A (Zero-Shot), B (Gold ctx_a), C (Random ctx_a), D (Finetuned Zero-Shot), E (Finetuned Gold ctx_a)"
    )
    parser.add_argument(
        "--gold_data_path",
        type=str,
        default="/data/johnwang/ICL/data/hellaswag_gold_2k.json",
        help="Path to gold dataset JSON"
    )
    parser.add_argument(
        "--random_data_path",
        type=str,
        default="/data/johnwang/ICL/data/hellaswag_random_2k.json",
        help="Path to random dataset JSON"
    )
    parser.add_argument(
        "--finetuned_model_path",
        type=str,
        default=None,
        help="Path to finetuned model (required for experiments D & E)"
    )
    parser.add_argument(
        "--num_few_shot",
        type=int,
        default=5,
        help="Number of few-shot examples"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None = all)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save results JSON"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.experiment in ["D", "E"] and args.finetuned_model_path is None:
        parser.error(f"finetuned_model_path is required for experiment {args.experiment}")
    
    # Run evaluation
    evaluate_hellaswag_icl(
        model_path=args.model_path,
        experiment=args.experiment,
        gold_data_path=args.gold_data_path,
        random_data_path=args.random_data_path,
        finetuned_model_path=args.finetuned_model_path,
        num_few_shot=args.num_few_shot,
        max_samples=args.max_samples,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()

