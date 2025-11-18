#!/usr/bin/env python3
"""
Download HellaSwag validation set (first 2000 samples) as gold dataset.
"""

import json
from datasets import load_dataset
from pathlib import Path

def download_hellaswag_gold(output_dir="/data/johnwang/huggingface_cache", num_samples=2000):
    """
    Download first 2000 samples from HellaSwag validation set.
    
    Args:
        output_dir: Output directory for the dataset
        num_samples: Number of samples to download (default: 2000)
    """
    print(f"Loading HellaSwag dataset...")
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    
    print(f"Total validation samples: {len(dataset)}")
    print(f"Taking first {num_samples} samples...")
    
    # Take first num_samples
    subset = dataset.select(range(num_samples))
    
    # Convert to list of dictionaries
    gold_data = []
    for i, example in enumerate(subset):
        gold_data.append({
            "ind": example.get("ind", i),
            "activity_label": example.get("activity_label", ""),
            "ctx_a": example.get("ctx_a", ""),
            "ctx_b": example.get("ctx_b", ""),
            "ctx": example.get("ctx", ""),
            "endings": example.get("endings", []),
            "label": example.get("label", ""),
            "source_id": example.get("source_id", ""),
        })
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON file
    output_file = output_path / "hellaswag_gold_2k.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(gold_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Successfully saved {len(gold_data)} samples to:")
    print(f"   {output_file}")
    print(f"\nSample data structure:")
    if gold_data:
        print(f"   - activity_label: {gold_data[0].get('activity_label', 'N/A')}")
        print(f"   - ctx_a length: {len(gold_data[0].get('ctx_a', ''))}")
        print(f"   - ctx_b length: {len(gold_data[0].get('ctx_b', ''))}")
        print(f"   - endings count: {len(gold_data[0].get('endings', []))}")
        print(f"   - label: {gold_data[0].get('label', 'N/A')}")
    
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download HellaSwag gold dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/johnwang/huggingface_cache",
        help="Output directory for the dataset"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of samples to download (default: 2000)"
    )
    
    args = parser.parse_args()
    
    download_hellaswag_gold(
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )

