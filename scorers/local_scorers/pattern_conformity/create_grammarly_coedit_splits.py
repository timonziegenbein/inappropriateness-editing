"""
Create train/dev/test splits from grammarly/coedit and dim/grammarly_coedit datasets.

This script:
1. Loads grammarly/coedit dataset (has train/validation splits)
2. Loads dim/grammarly_coedit dataset (full dataset)
3. For train and validation: Use grammarly/coedit splits directly (keeping prefixes)
4. For test: Find examples in dim/grammarly_coedit that are NOT in grammarly/coedit train or validation
5. Keeps full 'src' text including prefixes (needed for model prompting)
6. Extracts prefix separately for reference
7. Saves three JSONL files (train/dev/test) with duplicates allowed
8. Generates statistics report
"""

import json
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm

print("=" * 80)
print("CREATING GRAMMARLY/COEDIT SPLITS WITH TEST SET FROM DIM/GRAMMARLY_COEDIT")
print("=" * 80)


def extract_prefix(src_text):
    """
    Extract prefix from src column for reference, but keep full text.

    Returns: (prefix, full_src_text)
    """
    if ':' in src_text:
        parts = src_text.split(':', 1)
        if len(parts) == 2:
            prefix = parts[0].strip()
            return prefix, src_text.strip()

    # No colon found - empty prefix, use full text
    return "", src_text.strip()


def main():
    output_dir = "datasets/scorers/grammarly_coedit"

    # Step 1: Load grammarly/coedit dataset (train and validation)
    print("\n" + "=" * 80)
    print("STEP 1: LOADING GRAMMARLY/COEDIT DATASET")
    print("=" * 80)

    print("\nLoading grammarly/coedit from HuggingFace...")
    try:
        grammarly_coedit = load_dataset("grammarly/coedit")
        print(f"✓ Loaded grammarly/coedit dataset")
        print(f"  Available splits: {list(grammarly_coedit.keys())}")
        for split in grammarly_coedit.keys():
            print(f"    {split}: {len(grammarly_coedit[split])} examples")
    except Exception as e:
        print(f"✗ Error loading grammarly/coedit: {e}")
        return

    # Step 2: Load dim/grammarly_coedit dataset (full dataset)
    print("\n" + "=" * 80)
    print("STEP 2: LOADING DIM/GRAMMARLY_COEDIT DATASET")
    print("=" * 80)

    print("\nLoading dim/grammarly_coedit from HuggingFace...")
    try:
        dim_coedit = load_dataset("dim/grammarly_coedit")
        print(f"✓ Loaded dim/grammarly_coedit dataset")
        print(f"  Available splits: {list(dim_coedit.keys())}")
        for split in dim_coedit.keys():
            print(f"    {split}: {len(dim_coedit[split])} examples")
    except Exception as e:
        print(f"✗ Error loading dim/grammarly_coedit: {e}")
        return

    # Step 3: Build set of src texts from grammarly/coedit train and validation
    print("\n" + "=" * 80)
    print("STEP 3: BUILDING SET OF TRAIN/VALIDATION SOURCES")
    print("=" * 80)

    train_val_sources = set()

    # Collect from train split
    if 'train' in grammarly_coedit:
        print(f"\nCollecting sources from grammarly/coedit train split...")
        for example in tqdm(grammarly_coedit['train'], desc="Processing train"):
            src = example.get('src', '')
            if src:
                train_val_sources.add(src)
        print(f"✓ Added {len(train_val_sources)} unique sources from train")

    # Collect from validation split
    if 'validation' in grammarly_coedit:
        print(f"\nCollecting sources from grammarly/coedit validation split...")
        initial_count = len(train_val_sources)
        for example in tqdm(grammarly_coedit['validation'], desc="Processing validation"):
            src = example.get('src', '')
            if src:
                train_val_sources.add(src)
        added = len(train_val_sources) - initial_count
        print(f"✓ Added {added} new unique sources from validation")

    print(f"\nTotal unique sources in train+validation: {len(train_val_sources)}")

    # Step 4: Process train split (from grammarly/coedit)
    print("\n" + "=" * 80)
    print("STEP 4: PROCESSING TRAIN SPLIT")
    print("=" * 80)

    train_examples = []

    if 'train' in grammarly_coedit:
        print(f"\nProcessing {len(grammarly_coedit['train'])} examples from grammarly/coedit train...")
        for example in tqdm(grammarly_coedit['train'], desc="Processing train"):
            src = example.get('src', '')
            tgt = example.get('tgt', '')

            if not src or not tgt:
                continue

            prefix, full_src = extract_prefix(src)

            train_examples.append({
                'original_before_sent': full_src,  # Keep full text with prefix
                'after_sent': tgt,
                'split': 'train',
                'prefix': prefix,
                'similarity_score': 1.0
            })

        print(f"✓ Processed {len(train_examples)} train examples")

    # Step 5: Process validation split as dev (from grammarly/coedit)
    print("\n" + "=" * 80)
    print("STEP 5: PROCESSING DEV SPLIT (from validation)")
    print("=" * 80)

    dev_examples = []

    if 'validation' in grammarly_coedit:
        print(f"\nProcessing {len(grammarly_coedit['validation'])} examples from grammarly/coedit validation...")
        for example in tqdm(grammarly_coedit['validation'], desc="Processing validation"):
            src = example.get('src', '')
            tgt = example.get('tgt', '')

            if not src or not tgt:
                continue

            prefix, full_src = extract_prefix(src)

            dev_examples.append({
                'original_before_sent': full_src,  # Keep full text with prefix
                'after_sent': tgt,
                'split': 'dev',
                'prefix': prefix,
                'similarity_score': 1.0
            })

        print(f"✓ Processed {len(dev_examples)} dev examples")

    # Step 6: Create test split (from dim/grammarly_coedit, excluding train+validation)
    print("\n" + "=" * 80)
    print("STEP 6: CREATING TEST SPLIT FROM DIM/GRAMMARLY_COEDIT")
    print("=" * 80)

    test_examples = []
    stats = {
        'total_dim_examples': 0,
        'in_train_val': 0,
        'added_to_test': 0,
        'skipped_no_data': 0
    }

    # Process all splits from dim/grammarly_coedit
    for split_name in dim_coedit.keys():
        print(f"\nProcessing dim/grammarly_coedit '{split_name}' split...")

        for example in tqdm(dim_coedit[split_name], desc=f"Processing {split_name}"):
            stats['total_dim_examples'] += 1

            src = example.get('src', '')
            tgt = example.get('tgt', '')

            if not src or not tgt:
                stats['skipped_no_data'] += 1
                continue

            # Check if this source is in train or validation
            if src in train_val_sources:
                stats['in_train_val'] += 1
                continue

            # Not in train/validation, add to test (keep full text with prefix)
            prefix, full_src = extract_prefix(src)

            test_examples.append({
                'original_before_sent': full_src,  # Keep full text with prefix
                'after_sent': tgt,
                'split': 'test',
                'prefix': prefix,
                'similarity_score': 1.0
            })
            stats['added_to_test'] += 1

    print(f"\n✓ Test split creation complete")
    print(f"  Total dim/grammarly_coedit examples: {stats['total_dim_examples']}")
    print(f"  Already in train/validation: {stats['in_train_val']}")
    print(f"  Added to test: {stats['added_to_test']}")
    print(f"  Skipped (no data): {stats['skipped_no_data']}")

    # Step 7: Save splits
    print("\n" + "=" * 80)
    print("STEP 7: SAVING SPLITS")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    splits_to_save = {
        'train': train_examples,
        'dev': dev_examples,
        'test': test_examples
    }

    for split_name, examples in splits_to_save.items():
        output_file = output_path / f"{split_name}.jsonl"

        with open(output_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')

        print(f"✓ Saved {len(examples)} examples to {split_name}.jsonl")

    # Step 8: Save statistics
    print("\n" + "=" * 80)
    print("STEP 8: SAVING STATISTICS")
    print("=" * 80)

    statistics = {
        'train_examples': len(train_examples),
        'dev_examples': len(dev_examples),
        'test_examples': len(test_examples),
        'total_examples': len(train_examples) + len(dev_examples) + len(test_examples),
        'dim_coedit_stats': stats
    }

    stats_file = output_path / "matching_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)

    print(f"✓ Saved statistics to {stats_file.name}")

    # Step 9: Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_examples)} (from grammarly/coedit train)")
    print(f"  Dev: {len(dev_examples)} (from grammarly/coedit validation)")
    print(f"  Test: {len(test_examples)} (from dim/grammarly_coedit, excluding train+dev)")
    print(f"  Total: {len(train_examples) + len(dev_examples) + len(test_examples)}")

    print(f"\nTest split breakdown:")
    print(f"  Unique to dim/grammarly_coedit: {stats['added_to_test']}")
    print(f"  Overlap with train+dev: {stats['in_train_val']}")

    # Check for duplicates within test set (keeping prefixes, so duplicates are expected)
    test_sentences = [ex['original_before_sent'] for ex in test_examples]
    unique_test = set(test_sentences)
    if len(test_sentences) != len(unique_test):
        print(f"\nℹ️  Info: {len(test_sentences) - len(unique_test)} duplicate sentences in test set (same sentence with different prefixes)")
    else:
        print(f"\n✓ All test sentences are unique")

    print("\n" + "=" * 80)
    print("✓ ALL DONE!")
    print("=" * 80)
    print(f"\nOutput files created in: {output_path}")
    print(f"  - train.jsonl ({len(train_examples)} examples)")
    print(f"  - dev.jsonl ({len(dev_examples)} examples)")
    print(f"  - test.jsonl ({len(test_examples)} examples)")
    print(f"  - matching_stats.json")
    print(f"\nNext steps:")
    print(f"  1. Review matching_stats.json to validate split sizes")
    print(f"  2. Run extract_human_edits_train_grammarly.py to extract edits")


if __name__ == "__main__":
    main()
