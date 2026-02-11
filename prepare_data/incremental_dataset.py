#!/usr/bin/env python3
"""
Incremental Dataset Manager for MiniCPM-V-4.5

This utility helps manage incremental training by merging old and new datasets
with configurable ratios to prevent catastrophic forgetting.

Usage:
    python incremental_dataset.py --old-data data/train_data_v1.json \
                                   --new-data data/train_data_v2.json \
                                   --output data/train_data_incremental.json \
                                   --old-ratio 0.3
"""

import argparse
import json
import random
import os
from typing import List, Dict, Any
from pathlib import Path


def load_json_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a JSON dataset file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")
    
    return data


def validate_dataset_compatibility(old_data: List[Dict], new_data: List[Dict]) -> bool:
    """
    Validate that old and new datasets have compatible formats.
    
    Returns True if compatible, raises ValueError otherwise.
    """
    if not old_data or not new_data:
        raise ValueError("Cannot validate empty datasets")
    
    # Check a few samples from each dataset
    old_sample = old_data[0]
    new_sample = new_data[0]
    
    # Both should have 'id' and 'conversations' keys
    required_keys = ['id', 'conversations']
    for key in required_keys:
        if key not in old_sample:
            raise ValueError(f"Old dataset missing required key: {key}")
        if key not in new_sample:
            raise ValueError(f"New dataset missing required key: {key}")
    
    print(f"✓ Dataset format validation passed")
    print(f"  Old dataset: {len(old_data)} samples")
    print(f"  New dataset: {len(new_data)} samples")
    
    return True


def create_replay_buffer(dataset: List[Dict], ratio: float, seed: int = 42) -> List[Dict]:
    """
    Create a replay buffer by sampling a subset of the dataset.
    
    Args:
        dataset: Full dataset to sample from
        ratio: Fraction of dataset to keep (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Sampled subset of the dataset
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"Ratio must be between 0 and 1, got {ratio}")
    
    random.seed(seed)
    sample_size = int(len(dataset) * ratio)
    
    if sample_size == 0:
        print(f"⚠️  Warning: Ratio {ratio} results in 0 samples from dataset of size {len(dataset)}")
        return []
    
    return random.sample(dataset, sample_size)


def merge_datasets(
    old_data: List[Dict],
    new_data: List[Dict],
    old_ratio: float = 0.3,
    shuffle: bool = True,
    seed: int = 42
) -> List[Dict]:
    """
    Merge old and new datasets with specified ratio of old data.
    
    Args:
        old_data: Historical training data
        new_data: New training data
        old_ratio: Fraction of old data to include (e.g., 0.3 = 30%)
        shuffle: Whether to shuffle the merged dataset
        seed: Random seed for reproducibility
        
    Returns:
        Merged dataset with old + new data
    """
    print(f"\n🔄 Merging datasets:")
    print(f"  Old data: {len(old_data)} samples")
    print(f"  New data: {len(new_data)} samples")
    print(f"  Old data ratio: {old_ratio:.1%}")
    
    # Sample from old data
    old_samples = create_replay_buffer(old_data, old_ratio, seed)
    print(f"  Sampled old data: {len(old_samples)} samples")
    
    # Combine datasets
    merged = new_data + old_samples
    print(f"  Merged total: {len(merged)} samples")
    print(f"  Composition: {len(new_data)} new ({len(new_data)/len(merged):.1%}) + " 
          f"{len(old_samples)} old ({len(old_samples)/len(merged):.1%})")
    
    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(merged)
        print(f"  ✓ Dataset shuffled")
    
    return merged


def save_json_dataset(data: List[Dict], path: str):
    """Save dataset to JSON file."""
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Created directory: {output_dir}")
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Saved to: {path}")


def create_version_metadata(
    output_path: str,
    old_data_path: str,
    new_data_path: str,
    old_ratio: float,
    old_count: int,
    new_count: int,
    total_count: int
):
    """Create a metadata file documenting the incremental dataset."""
    metadata = {
        "type": "incremental_training_dataset",
        "created_at": str(Path(output_path).stat().st_mtime),
        "sources": {
            "old_data": {
                "path": old_data_path,
                "original_count": old_count,
                "sampled_count": int(old_count * old_ratio),
                "ratio": old_ratio
            },
            "new_data": {
                "path": new_data_path,
                "count": new_count
            }
        },
        "output": {
            "path": output_path,
            "total_count": total_count
        },
        "composition": {
            "new_data_percentage": (new_count / total_count * 100),
            "old_data_percentage": (int(old_count * old_ratio) / total_count * 100)
        }
    }
    
    metadata_path = output_path.replace('.json', '_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge old and new datasets for incremental training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--old-data',
        type=str,
        required=True,
        help='Path to old/historical training data (JSON)'
    )
    
    parser.add_argument(
        '--new-data',
        type=str,
        required=True,
        help='Path to new training data (JSON)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for merged dataset (JSON)'
    )
    
    parser.add_argument(
        '--old-ratio',
        type=float,
        default=0.3,
        help='Fraction of old data to include (0.0-1.0)'
    )
    
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Do not shuffle the merged dataset'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate datasets without merging'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Incremental Dataset Manager")
    print("=" * 60)
    
    # Load datasets
    print("\n�� Loading datasets...")
    old_data = load_json_dataset(args.old_data)
    new_data = load_json_dataset(args.new_data)
    print(f"  ✓ Loaded old data: {len(old_data)} samples")
    print(f"  ✓ Loaded new data: {len(new_data)} samples")
    
    # Validate compatibility
    print("\n🔍 Validating dataset compatibility...")
    validate_dataset_compatibility(old_data, new_data)
    
    if args.validate_only:
        print("\n✅ Validation complete. Exiting (--validate-only mode)")
        return
    
    # Merge datasets
    merged_data = merge_datasets(
        old_data=old_data,
        new_data=new_data,
        old_ratio=args.old_ratio,
        shuffle=not args.no_shuffle,
        seed=args.seed
    )
    
    # Save merged dataset
    print(f"\n💾 Saving merged dataset...")
    save_json_dataset(merged_data, args.output)
    
    # Create metadata
    print(f"\n📝 Creating metadata...")
    create_version_metadata(
        output_path=args.output,
        old_data_path=args.old_data,
        new_data_path=args.new_data,
        old_ratio=args.old_ratio,
        old_count=len(old_data),
        new_count=len(new_data),
        total_count=len(merged_data)
    )
    
    print("\n" + "=" * 60)
    print("✅ Incremental dataset created successfully!")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"  Total samples: {len(merged_data)}")
    print(f"  New data: {len(new_data)} ({len(new_data)/len(merged_data):.1%})")
    print(f"  Old data: {int(len(old_data) * args.old_ratio)} ({int(len(old_data) * args.old_ratio)/len(merged_data):.1%})")
    print(f"\n🚀 Ready for incremental training!")
    print(f"  Use: {args.output}")


if __name__ == '__main__':
    main()
