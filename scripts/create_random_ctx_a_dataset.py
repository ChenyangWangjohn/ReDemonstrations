#!/usr/bin/env python3
"""
Create Random ctx_a dataset from HellaSwag gold dataset.

For each sample in the gold dataset:
1. Find training samples with the same activity_label but different ctx_a
2. Replace ctx_a with a random candidate's ctx_a
3. Keep ctx_b, endings, and label unchanged
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def load_gold_dataset(gold_file):
    """Load gold dataset from JSON file."""
    with open(gold_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def group_by_activity_label(dataset):
    """
    Group dataset samples by activity_label.
    
    Args:
        dataset: HuggingFace dataset or list of dicts
        
    Returns:
        dict: {activity_label: [samples]}
    """
    grouped = defaultdict(list)
    for example in dataset:
        activity = example.get("activity_label", "")
        grouped[activity].append(example)
    return grouped

def create_random_ctx_a_dataset(gold_data, train_dataset, seed=42):
    """
    Create Random ctx_a dataset.
    
    Strategy:
    1. First try: Find candidates from training set with same activity_label but different ctx_a
    2. Fallback: If no candidates in training set, use validation set (same activity_label, different ctx_a)
    3. This ensures ALL samples get a different ctx_a
    
    Args:
        gold_data: List of gold samples (from validation set)
        train_dataset: Training dataset (for finding candidate ctx_a)
        seed: Random seed for reproducibility
        
    Returns:
        List of samples with random ctx_a
    """
    random.seed(seed)
    
    # Group training set by activity_label
    print("Grouping training set by activity_label...")
    grouped_train_by_activity = group_by_activity_label(train_dataset)
    
    # Group validation set by activity_label (for fallback)
    print("Grouping validation set by activity_label...")
    grouped_val_by_activity = group_by_activity_label(gold_data)
    
    print(f"Found {len(grouped_train_by_activity)} unique activity_labels in training set")
    print(f"Found {len(grouped_val_by_activity)} unique activity_labels in validation set")
    
    random_ctx_a_data = []
    stats = {
        "total": len(gold_data),
        "replaced_from_train": 0,
        "replaced_from_val": 0,
        "kept_original": 0,
    }
    
    print("\nCreating Random ctx_a dataset...")
    for idx, example in enumerate(tqdm(gold_data, desc="Processing samples")):
        activity = example.get("activity_label", "")
        original_ctx_a = example.get("ctx_a", "")
        
        # Strategy 1: Find candidates from training set with same activity_label but different ctx_a
        train_samples = grouped_train_by_activity.get(activity, [])
        train_candidates = [
            s for s in train_samples 
            if s.get("ctx_a", "") != original_ctx_a
        ]
        
        if train_candidates:
            # Use training set candidate
            selected = random.choice(train_candidates)
            new_ctx_a = selected.get("ctx_a", original_ctx_a)
            stats["replaced_from_train"] += 1
        else:
            # Strategy 2: Fallback to validation set (same activity_label, different ctx_a)
            val_samples = grouped_val_by_activity.get(activity, [])
            val_candidates = [
                s for s in val_samples 
                if s.get("ctx_a", "") != original_ctx_a and s.get("ind") != example.get("ind")
            ]
            
            if val_candidates:
                # Use validation set candidate (same activity_label, different ctx_a)
                selected = random.choice(val_candidates)
                new_ctx_a = selected.get("ctx_a", original_ctx_a)
                stats["replaced_from_val"] += 1
            else:
                # Should not happen, but keep original as fallback
                new_ctx_a = original_ctx_a
                stats["kept_original"] += 1
                print(f"\nWarning: No candidates found for sample {idx}, activity: {activity}")
        
        # Create random ctx_a sample
        random_sample = {
            "ind": example.get("ind", len(random_ctx_a_data)),
            "activity_label": activity,
            "ctx_a": new_ctx_a,                    # Modified ctx_a
            "ctx_b": example.get("ctx_b", ""),     # Keep original ctx_b
            "ctx": new_ctx_a + " " + example.get("ctx_b", ""),  # New full context
            "endings": example.get("endings", []), # Keep original endings
            "label": example.get("label", ""),     # Keep original label (correct answer)
            "source_id": example.get("source_id", ""),
            "original_ctx_a": original_ctx_a,      # Keep track of original for debugging
        }
        
        random_ctx_a_data.append(random_sample)
    
    # Print statistics
    print("\n" + "="*50)
    print("Statistics:")
    print(f"  Total samples: {stats['total']}")
    print(f"  ctx_a replaced from train: {stats['replaced_from_train']} ({stats['replaced_from_train']/stats['total']*100:.1f}%)")
    print(f"  ctx_a replaced from val: {stats['replaced_from_val']} ({stats['replaced_from_val']/stats['total']*100:.1f}%)")
    print(f"  ctx_a kept original: {stats['kept_original']} ({stats['kept_original']/stats['total']*100:.1f}%)")
    total_replaced = stats['replaced_from_train'] + stats['replaced_from_val']
    print(f"  Total replaced: {total_replaced} ({total_replaced/stats['total']*100:.1f}%)")
    print("="*50)
    
    return random_ctx_a_data

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Random ctx_a dataset from HellaSwag gold dataset")
    parser.add_argument(
        "--gold_file",
        type=str,
        default="/data/johnwang/ICL/data/hellaswag_gold_2k.json",
        help="Path to gold dataset JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/data/johnwang/ICL/data/hellaswag_random_2k.json",
        help="Output path for random ctx_a dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Load gold dataset
    print(f"Loading gold dataset from: {args.gold_file}")
    gold_data = load_gold_dataset(args.gold_file)
    print(f"Loaded {len(gold_data)} gold samples")
    
    # Load training dataset
    print("\nLoading HellaSwag training dataset...")
    train_dataset = load_dataset("Rowan/hellaswag", split="train")
    print(f"Loaded {len(train_dataset)} training samples")
    
    # Create random ctx_a dataset
    random_ctx_a_data = create_random_ctx_a_dataset(
        gold_data=gold_data,
        train_dataset=train_dataset,
        seed=args.seed
    )
    
    # Save to file
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving Random ctx_a dataset to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(random_ctx_a_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully saved {len(random_ctx_a_data)} samples")
    
    # Show sample comparison
    if random_ctx_a_data:
        print("\n" + "="*50)
        print("Sample comparison (first sample):")
        print("="*50)
        sample = random_ctx_a_data[0]
        print(f"Activity Label: {sample['activity_label']}")
        print(f"\nOriginal ctx_a: {sample['original_ctx_a'][:100]}...")
        print(f"New ctx_a:      {sample['ctx_a'][:100]}...")
        print(f"ctx_b:          {sample['ctx_b']}")
        print(f"Label:          {sample['label']} (unchanged)")
        print(f"Changed:        {sample['ctx_a'] != sample['original_ctx_a']}")

if __name__ == "__main__":
    main()

