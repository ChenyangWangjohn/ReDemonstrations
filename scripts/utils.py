"""
Shared utilities for HellaSwag ICL evaluation scripts.
"""

import re


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

