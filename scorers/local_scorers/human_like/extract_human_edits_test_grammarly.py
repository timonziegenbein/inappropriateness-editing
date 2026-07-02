"""
Extract human edits from the grammarly/coedit test split and save as JSONL for evaluation.

This script:
1. Loads the test split from datasets/scorers/grammarly_coedit/test.jsonl
2. Removes prefix from original_before_sent (split by ": " and take text after first match)
3. Extracts edits using fuzzy matching (via parse_edits_from_texts)
4. Saves as human_with_edits_test.jsonl
"""

import json
from pathlib import Path
import sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))

# Import edit extraction functions
from precompute_edits import parse_edits_from_texts

script_dir = Path(__file__).parent

print("=" * 80)
print("EXTRACTING HUMAN EDITS FROM GRAMMARLY/COEDIT TEST SPLIT")
print("=" * 80)


def remove_prefix(text):
    """
    Remove prefix from text by splitting on ": " and taking everything after first occurrence.

    Returns: cleaned text without prefix
    """
    if ':' in text:
        parts = text.split(':', 1)
        if len(parts) == 2:
            return parts[1].strip()

    # No colon found, return full text
    return text.strip()


# Load test split from local file
print("\nLoading test split from local file...")
test_file = Path("datasets/scorers/grammarly_coedit/test.jsonl")

if not test_file.exists():
    print(f"✗ Error: {test_file} not found!")
    print("Please run create_grammarly_coedit_splits.py first")
    sys.exit(1)

try:
    dataset = []
    with open(test_file, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    print(f"✓ Loaded {len(dataset)} examples from test split")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    sys.exit(1)


# Process examples
def process_example(example):
    """Extract edits for a single example."""
    # Get original_before_sent (contains prefix) and after_sent
    original_with_prefix = example.get('original_before_sent', '')
    edited = example.get('after_sent', '')

    if not original_with_prefix or not edited:
        return None

    # Remove prefix from original
    original = remove_prefix(original_with_prefix)

    # Extract edits using latex diff and fuzzy matching
    parsed_edits = parse_edits_from_texts(original, edited)

    # Create output example
    return {
        'original_before_sent': original,  # Without prefix
        'parsed_edits': parsed_edits,
        'original_intent': '',  # Not available in grammarly_coedit
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

# Filter out None results and examples with no edits
results = [r for r in results if r is not None and r.get('parsed_edits')]
print(f"✓ Processed {len(results)} examples with valid edits")

# Calculate statistics
total_edits = sum(len(r['parsed_edits']) for r in results)
print(f"  Total edits extracted: {total_edits}")
print(f"  Average edits per example: {total_edits / len(results):.2f}")

# Save to file
output_file = script_dir / "data" / "human_with_edits_test.jsonl"
output_file.parent.mkdir(parents=True, exist_ok=True)

print(f"\nSaving to {output_file.name}...")

with open(output_file, 'w') as f:
    for example in results:
        f.write(json.dumps(example) + '\n')

print(f"✓ Saved {len(results)} examples to {output_file.name}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"\nCreated: {output_file}")
print(f"Use this file for evaluation after training models.")
print(f"\nNext steps:")
print(f"  1. Train models using training data")
print(f"  2. Generate model predictions (CoEdIT, Llama, Gemini) on test set")
print(f"  3. Evaluate all predictions using trained scorers")
