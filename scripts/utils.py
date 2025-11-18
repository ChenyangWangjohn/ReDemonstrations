"""
Shared utilities for HellaSwag ICL evaluation scripts.
"""

import re
from pathlib import Path


def load_system_prompt():
    """
    Load system prompt from file.
    
    Returns:
        System prompt string, or empty string if file not found
    """
    script_dir = Path(__file__).parent
    prompt_file = script_dir.parent / "system_prompt.txt"
    
    if prompt_file.exists():
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""


def format_hellaswag_example_zeroshot(example, include_system_prompt=True):
    """
    Format Zero-Shot example (only ctx_b, no ctx_a).
    
    Args:
        example: Dictionary with 'ctx_b' and 'endings'
        include_system_prompt: Whether to include system prompt
        
    Returns:
        Formatted prompt string
    """
    prompt = ""
    
    # Add system prompt if requested
    if include_system_prompt:
        system_prompt = load_system_prompt()
        if system_prompt:
            prompt += f"{system_prompt}\n\n"
    
    ctx_b = example.get("ctx_b", "")
    endings = example.get("endings", [])
    
    prompt += f"{ctx_b}\n"
    prompt += f"A) {endings[0]}\n"
    prompt += f"B) {endings[1]}\n"
    prompt += f"C) {endings[2]}\n"
    prompt += f"D) {endings[3]}\n"
    prompt += "Answer:"
    
    return prompt


def create_few_shot_prompt(examples, test_example, ctx_a_type="gold", include_system_prompt=True):
    """
    Create Few-Shot Prompt with examples.
    
    Args:
        examples: List of few-shot examples (from gold or random dataset)
        test_example: Test sample to evaluate
        ctx_a_type: "gold" or "random", determines which ctx_a to use
        include_system_prompt: Whether to include system prompt
        
    Returns:
        Formatted prompt string
    """
    prompt = ""
    
    # Add system prompt if requested
    if include_system_prompt:
        system_prompt = load_system_prompt()
        if system_prompt:
            prompt += f"{system_prompt}\n\n"
    
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


def extract_answer_from_logits(model, tokenizer, prompt, choices=["A", "B", "C", "D"]):
    """
    Extract answer using log probability method (standard for HellaSwag).
    
    This is the standard evaluation method for multiple-choice questions:
    1. Get logits at the last token position (after "Answer:")
    2. Extract logits for choice tokens (A, B, C, D)
    3. Compute probabilities using softmax
    4. Select the choice with highest probability
    
    This method is more stable and accurate than text generation + extraction.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt string (should end with "Answer:")
        choices: List of choice letters (default: ["A", "B", "C", "D"])
        
    Returns:
        Tuple of (predicted_answer, choice_probs_dict, log_probs_dict)
        - predicted_answer: The selected answer (A, B, C, or D)
        - choice_probs_dict: Dictionary mapping choice to probability
        - log_probs_dict: Dictionary mapping choice to log probability
    """
    import torch
    import torch.nn.functional as F
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get logits (no generation, just forward pass)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get logits at the last token position
    # This is where the model would predict the next token (the answer choice)
    sequence_length = inputs['attention_mask'].sum(dim=1) - 1  # Last non-padding token
    last_token_logits = logits[0, sequence_length.item(), :]  # Shape: (vocab_size,)
    
    # Encode choice tokens and get their logits
    choice_logits = []
    choice_token_ids_list = []
    for choice in choices:
        # Encode the choice letter (e.g., "A")
        choice_token_ids = tokenizer.encode(choice, add_special_tokens=False)
        if choice_token_ids:
            # Use the last token ID (in case tokenizer splits into multiple tokens)
            # For most tokenizers, "A" should be a single token
            choice_token_id = choice_token_ids[-1]
            choice_logits.append(last_token_logits[choice_token_id].item())
            choice_token_ids_list.append(choice_token_id)
        else:
            choice_logits.append(float('-inf'))
            choice_token_ids_list.append(None)
    
    # Convert to tensor and compute probabilities
    choice_logits_tensor = torch.tensor(choice_logits, device=model.device)
    choice_probs = F.softmax(choice_logits_tensor, dim=0)
    choice_log_probs = F.log_softmax(choice_logits_tensor, dim=0)
    
    # Select the choice with highest probability
    predicted_idx = torch.argmax(choice_probs).item()
    predicted_answer = choices[predicted_idx]
    
    # Create dictionaries for easy access
    choice_probs_dict = {choice: prob.item() for choice, prob in zip(choices, choice_probs)}
    log_probs_dict = {choice: log_prob.item() for choice, log_prob in zip(choices, choice_log_probs)}
    
    return predicted_answer, choice_probs_dict, log_probs_dict