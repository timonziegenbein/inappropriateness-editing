"""
Extract human edits from the IteraTeR train split and save as JSONL for HMM/ISO training.

This script:
1. Loads the train split from HuggingFace (timonziegenbein/iterater-with-prefixes)
2. Extracts edits using fuzzy matching
3. Saves as human_with_edits_train.jsonl
"""

import json
from datasets import load_dataset
from pathlib import Path
import sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))

# Import edit extraction functions
from precompute_edits import parse_edits_from_texts

script_dir = Path(__file__).parent

print("="*80)
print("EXTRACTING HUMAN EDITS FROM TRAIN SPLIT")
print("="*80)

# Load train split from HuggingFace
print("\nLoading train split from HuggingFace...")
try:
    dataset = load_dataset("timonziegenbein/iterater-with-prefixes", split="train")
    print(f"✓ Loaded {len(dataset)} examples from train split")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Make sure the dataset has been pushed to HuggingFace first.")
    print("Run: python scorers/local_scorers/pattern_conformity/create_iterater_test_dataset.py")
    sys.exit(1)

# Process examples
def process_example(example):
    """Extract edits for a single example."""
    # Use the original_before_sent (without prefix) and after_sent
    original = example.get('original_before_sent', '')
    edited = example.get('after_sent', '')

    if not original or not edited:
        return None

    # Extract edits using latex diff and fuzzy matching
    parsed_edits = parse_edits_from_texts(original, edited)

    # Create output example
    return {
        'original_before_sent': original,
        'parsed_edits': parsed_edits,
        'original_intent': example.get('original_intent', ''),
        'prefix': example.get('prefix', ''),
    }

# Process in parallel
num_workers = max(1, cpu_count() - 1)
print(f"\nProcessing {len(dataset)} examples with {num_workers} workers...")

with Pool(num_workers) as pool:
    results = list(tqdm(
        pool.imap(process_example, dataset, chunksize=50),
        total=len(dataset),
        desc="Extracting edits"
    ))

# Filter out None results
results = [r for r in results if r is not None]
print(f"✓ Processed {len(results)} examples")

# Save to file
output_file = script_dir / "data" / "human_with_edits_train.jsonl"
print(f"\nSaving to {output_file.name}...")

with open(output_file, 'w') as f:
    for example in results:
        f.write(json.dumps(example) + '\n')

print(f"✓ Saved {len(results)} examples to {output_file.name}")

print("\n" + "="*80)
print("DONE")
print("="*80)
print(f"\nCreated: {output_file.name}")
print(f"Use this file to train HMM and Isolation Forest models.")
