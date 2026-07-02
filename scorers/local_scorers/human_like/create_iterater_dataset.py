"""
Create train/dev/test datasets from IteraTeR by replacing intents with CoEdit prefixes.

This script:
1. Loads the grammarly/coedit dataset to build a prefix lexicon
2. Loads the IteraTeR full_sent_level/{train,dev,test}.json datasets
3. Extracts before_sent and after_sent pairs
4. Replaces intents with random prefixes from the lexicon
5. Filters out pairs where intent is not in lexicon
6. Pushes the result to HuggingFace as train/dev/test splits
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
import re


def build_prefix_lexicon():
    """Build a lexicon mapping tasks to their prefixes from the CoEdit dataset."""
    print("Loading grammarly/coedit dataset...")
    coedit = load_dataset("grammarly/coedit")

    all_prefixes = set()

    # Process all splits to collect all prefixes
    for split_name in coedit.keys():
        print(f"Processing {split_name} split...")
        for example in coedit[split_name]:
            src = example['src']
            # Extract prefix (everything before the first colon)
            if ':' in src:
                prefix = src.split(':', 1)[0].strip()
                all_prefixes.add(prefix)

    print(f"\nFound {len(all_prefixes)} unique prefixes in CoEdit")

    # Map IteraTeR intents to relevant CoEdit prefixes
    intent_to_prefixes = {
        'fluency': [],
        'clarity': [],
        'coherence': [],
        'style': []
    }

    # Categorize prefixes based on keywords
    for prefix in all_prefixes:
        prefix_lower = prefix.lower()

        # Fluency: grammar, grammatical, disfluencies
        if any(keyword in prefix_lower for keyword in ['grammar', 'grammatical', 'disfluenc']):
            intent_to_prefixes['fluency'].append(prefix)

        # Clarity: clear, simple, readable, understandable
        if any(keyword in prefix_lower for keyword in ['clear', 'simple', 'simpl', 'readable', 'understand', 'readab']):
            intent_to_prefixes['clarity'].append(prefix)

        # Coherence: coherent, cohesive, consistent, logical, flow, transition
        if any(keyword in prefix_lower for keyword in ['coheren', 'cohesiv', 'consistent', 'logical', 'flow', 'transition']):
            intent_to_prefixes['coherence'].append(prefix)

        # Style: neutral, POV, paraphrase, rephrase, reword, rewrite
        if any(keyword in prefix_lower for keyword in ['neutral', 'pov', 'paraphrase', 'rephrase', 'reword', 'rewrite', 'wording']):
            intent_to_prefixes['style'].append(prefix)

    print(f"\nIntent to prefix mapping:")
    for intent, prefixes in intent_to_prefixes.items():
        print(f"  {intent}: {len(prefixes)} prefix(es)")
        if len(prefixes) > 0:
            print(f"    Examples: {prefixes[:3]}")

    return intent_to_prefixes


def load_iterater_split(split_name):
    """Load a split from the IteraTeR dataset from local JSON Lines file.

    Args:
        split_name: 'train', 'dev', or 'test'
    """
    print(f"\nLoading IteraTeR {split_name} dataset from local file...")

    split_file = f"datasets/scorers/IteraTeR/full_sent_level/{split_name}.json"

    try:
        data = []
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        print(f"Loaded {len(data)} examples from {split_file}")
        return data
    except Exception as e:
        print(f"Error loading IteraTeR dataset from {split_file}: {e}")
        return None


def extract_intent(before_sent_with_intent):
    """Extract the intent from between <> brackets."""
    match = re.search(r'<([^>]+)>', before_sent_with_intent)
    if match:
        return match.group(1).strip()
    return None


def replace_intent_with_prefix(before_sent_with_intent, prefix):
    """Replace the intent in <> with a prefix."""
    return re.sub(r'<[^>]+>', prefix, before_sent_with_intent)


def process_iterater_dataset(iterater_data, prefix_lexicon):
    """Process IteraTeR dataset and replace intents with prefixes."""
    print("\nProcessing IteraTeR dataset...")

    processed_examples = []
    intent_counts = defaultdict(int)
    matched_intents = 0
    unmatched_intents = 0

    for example in iterater_data:
        # Extract the relevant fields
        before_sent = example.get('before_sent', '')
        after_sent = example.get('after_sent', '')
        intent = example.get('labels', '')

        if not before_sent or not after_sent or not intent:
            continue

        intent_counts[intent] += 1

        # Check if intent exists in prefix lexicon
        if intent in prefix_lexicon and len(prefix_lexicon[intent]) > 0:
            matched_intents += 1
            # Replace with random prefix
            prefixes = prefix_lexicon[intent]
            random_prefix = random.choice(prefixes)

            # Format as "prefix: sentence"
            new_before_sent = f"{random_prefix}: {before_sent}"

            processed_examples.append({
                'original_intent': intent,
                'prefix': random_prefix,
                'before_sent': new_before_sent,
                'after_sent': after_sent,
                'original_before_sent': before_sent
            })
        else:
            unmatched_intents += 1

    print(f"\nProcessing statistics:")
    print(f"  Total intents found: {sum(intent_counts.values())}")
    print(f"  Matched intents: {matched_intents}")
    print(f"  Unmatched intents: {unmatched_intents}")
    print(f"  Unique intents: {len(intent_counts)}")
    print(f"\nIntent distribution:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        status = "✓" if intent in prefix_lexicon else "✗"
        print(f"  {status} {intent}: {count}")

    return processed_examples


def push_to_huggingface(train_examples, dev_examples, test_examples, dataset_name="timonziegenbein/iterater-with-prefixes"):
    """Push the processed dataset splits to HuggingFace."""
    print(f"\nCreating dataset splits...")
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Dev: {len(dev_examples)} examples")
    print(f"  Test: {len(test_examples)} examples")

    from datasets import DatasetDict

    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_examples),
        'dev': Dataset.from_list(dev_examples),
        'test': Dataset.from_list(test_examples)
    })

    print(f"\nPushing to HuggingFace as {dataset_name}...")
    dataset_dict.push_to_hub(dataset_name, private=False)

    print(f"✓ Dataset successfully pushed to {dataset_name}")


def main():
    # Set random seed for reproducibility
    random.seed(42)

    # Step 1: Build prefix lexicon from CoEdit
    prefix_lexicon = build_prefix_lexicon()

    # Step 2: Load IteraTeR splits
    iterater_train = load_iterater_split('train')
    iterater_dev = load_iterater_split('dev')
    iterater_test = load_iterater_split('test')

    if iterater_train is None or iterater_dev is None or iterater_test is None:
        print("Failed to load IteraTeR dataset splits. Exiting.")
        return

    # Step 3: Process each split
    print("\n" + "="*80)
    print("PROCESSING TRAIN SPLIT")
    print("="*80)
    processed_train = process_iterater_dataset(iterater_train, prefix_lexicon)

    print("\n" + "="*80)
    print("PROCESSING DEV SPLIT")
    print("="*80)
    processed_dev = process_iterater_dataset(iterater_dev, prefix_lexicon)

    print("\n" + "="*80)
    print("PROCESSING TEST SPLIT")
    print("="*80)
    processed_test = process_iterater_dataset(iterater_test, prefix_lexicon)

    if not processed_train or not processed_dev or not processed_test:
        print("No examples were processed in one or more splits. Exiting.")
        return

    # Step 4: Push to HuggingFace
    push_to_huggingface(processed_train, processed_dev, processed_test)

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
