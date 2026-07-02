"""
Create annotation pairs for the sentence-level annotation interface.
Each row represents a single edit: original sentence -> rewritten sentence.

Sampling strategy (200 edits per approach):
- 100 perfect edits (rewards.perfect == 1.0) randomly sampled
- 100 non-perfect edits (rewards.perfect != 1.0) randomly sampled
- Only samples from valid edits (valid=True)
- No requirement for same sentence across approaches
- Edits must be applicable (substring match validation)

Output CSV format:
- id: POSTID_APPROACH_EDITIDX
- source: original sentence
- inappropriate_part: text being replaced
- rewritten_part: replacement text
- issue: topic of the argument
- batch: always 1
"""

import json
import csv
import random
from pathlib import Path
from collections import defaultdict

# Configuration
PREDICTION_FILES = {
    'v11_only_lm': 'models/predictions/grpo_global_sentence_v11_only_lm.jsonl',
    '50a_50ss': 'models/predictions/ppo_50a_50ss.jsonl',
    'base_model': 'models/predictions/sentence_base_model_ppo_classifier.jsonl'
}

OUTPUT_CSV = 'annotation-interface/appropriateness-study-abs/data/study_pairs.csv'
BATCH_ID = 1
RANDOM_SEED = 42
PERFECT_EDITS_PER_APPROACH = 100
NON_PERFECT_EDITS_PER_APPROACH = 100
EDITS_PER_APPROACH = PERFECT_EDITS_PER_APPROACH + NON_PERFECT_EDITS_PER_APPROACH


def load_predictions(file_path):
    """Load predictions from a JSONL file."""
    predictions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def extract_all_edits(predictions, approach_name):
    """
    Extract all valid and applicable edits from predictions, separated by perfect reward.

    Args:
        predictions: List of prediction objects
        approach_name: Name of the approach (for ID generation)

    Returns:
        Tuple of (perfect_edits, non_perfect_edits) - both are lists of edit dicts
        - Only includes edits where valid=True
        - Ensures inappropriate_part and rewritten_part are non-empty and different
        - Perfect edits have rewards.perfect == 1.0
        - All edits must pass substring match validation
    """
    perfect_edits = []
    non_perfect_edits = []
    edit_counter = 0

    for pred in predictions:
        post_id = pred['post_id']
        issue = pred['issue']

        # Get all edits
        if 'edits' not in pred:
            continue

        for edit in pred['edits']:
            # Only include valid edits
            if not edit.get('valid', False):
                continue

            original_sentence = edit.get('original_sentence', '')
            inappropriate_part = edit.get('inappropriate_part', '')
            rewritten_part = edit.get('rewritten_part', '')

            # Ensure all required fields are non-empty
            if not original_sentence or not inappropriate_part or not rewritten_part:
                continue

            # Ensure the edit actually changes something
            if inappropriate_part == rewritten_part:
                continue

            # Validate that the edit can be applied (substring match)
            if inappropriate_part not in original_sentence:
                continue

            edit_counter += 1

            # Check if this is a perfect edit (rewards.perfect == 1.0)
            rewards = edit.get('rewards', {})
            perfect_reward = rewards.get('perfect', 0.0)
            is_perfect = (perfect_reward == 1.0)

            # Create edit record with unique ID
            edit_record = {
                'id': f"{post_id}_{approach_name}_{edit_counter}",
                'post_id': post_id,
                'source': original_sentence,
                'inappropriate_part': inappropriate_part,
                'rewritten_part': rewritten_part,
                'issue': issue,
                'batch': BATCH_ID,
                'approach': approach_name,
                'is_perfect': is_perfect,
                'perfect_reward': perfect_reward
            }

            # Add to appropriate list
            if is_perfect:
                perfect_edits.append(edit_record)
            else:
                non_perfect_edits.append(edit_record)

    return perfect_edits, non_perfect_edits


def sample_edits_stratified(all_edits_by_approach):
    """
    Sample edits from each approach with stratification (perfect vs non-perfect).

    Args:
        all_edits_by_approach: Dict mapping approach_name to tuple of (perfect_edits, non_perfect_edits)

    Returns:
        Dict mapping approach_name to list of selected edit records
    """
    random.seed(RANDOM_SEED)

    selected_edits = {}

    print("\n=== Stratified Sampling (Perfect + Non-Perfect) ===\n")

    for approach, (perfect_edits, non_perfect_edits) in all_edits_by_approach.items():
        perfect_count = len(perfect_edits)
        non_perfect_count = len(non_perfect_edits)

        # Sample perfect edits
        perfect_sample_count = min(PERFECT_EDITS_PER_APPROACH, perfect_count)
        if perfect_count < PERFECT_EDITS_PER_APPROACH:
            print(f"Warning: {approach} has only {perfect_count} perfect edits (requested {PERFECT_EDITS_PER_APPROACH})")

        selected_perfect = random.sample(perfect_edits, perfect_sample_count) if perfect_count > 0 else []

        # Sample non-perfect edits
        non_perfect_sample_count = min(NON_PERFECT_EDITS_PER_APPROACH, non_perfect_count)
        if non_perfect_count < NON_PERFECT_EDITS_PER_APPROACH:
            print(f"Warning: {approach} has only {non_perfect_count} non-perfect edits (requested {NON_PERFECT_EDITS_PER_APPROACH})")

        selected_non_perfect = random.sample(non_perfect_edits, non_perfect_sample_count) if non_perfect_count > 0 else []

        # Combine
        selected = selected_perfect + selected_non_perfect
        selected_edits[approach] = selected

        print(f"{approach}: {len(selected_perfect)} perfect + {len(selected_non_perfect)} non-perfect = {len(selected)} total edits")
        print(f"  Available: {perfect_count} perfect, {non_perfect_count} non-perfect")

    return selected_edits


def main():
    """Main function to generate the annotation CSV."""
    print("Loading predictions and extracting all edits...\n")

    all_edits_by_approach = {}

    for approach_name, file_path in PREDICTION_FILES.items():
        print(f"Processing {approach_name} from {file_path}...")

        # Load predictions
        predictions = load_predictions(file_path)
        print(f"  Loaded {len(predictions)} predictions")

        # Extract all applicable edits, separated by validity
        perfect_edits, non_perfect_edits = extract_all_edits(predictions, approach_name)
        print(f"  Extracted {len(perfect_edits)} perfect edits and {len(non_perfect_edits)} non-perfect edits")

        all_edits_by_approach[approach_name] = (perfect_edits, non_perfect_edits)
        print()

    # Sample edits with stratification
    selected_edits = sample_edits_stratified(all_edits_by_approach)

    # Flatten to list of all selected edits
    all_selected = []
    for approach, edits in selected_edits.items():
        all_selected.extend(edits)

    print(f"\n=== Final Results ===")
    print(f"Total edits: {len(all_selected)}")
    for approach, edits in selected_edits.items():
        perfect_count = sum(1 for e in edits if e.get('is_perfect', False))
        non_perfect_count = len(edits) - perfect_count
        print(f"  {approach}: {len(edits)} total ({perfect_count} perfect, {non_perfect_count} non-perfect)")

    print(f"\nWriting to {OUTPUT_CSV}...")

    # Write to CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'source', 'inappropriate_part', 'rewritten_part', 'issue', 'batch'])
        writer.writeheader()
        for edit in all_selected:
            # Remove extra fields used for sampling
            row = {k: v for k, v in edit.items() if k in ['id', 'source', 'inappropriate_part', 'rewritten_part', 'issue', 'batch']}
            writer.writerow(row)

    print(f"✓ Successfully created {OUTPUT_CSV} with {len(all_selected)} rows")

    # Print sample
    print("\nSample rows:")
    for i, edit in enumerate(all_selected[:3]):
        perfect_label = "perfect" if edit.get('is_perfect', False) else "non-perfect"
        perfect_reward = edit.get('perfect_reward', 0.0)
        print(f"\n{i+1}. ID: {edit['id']} [{perfect_label}, reward={perfect_reward}]")
        print(f"   Source: {edit['source'][:80]}...")
        print(f"   Inappropriate: {edit['inappropriate_part'][:80]}...")
        print(f"   Rewritten: {edit['rewritten_part'][:80]}...")
        print(f"   Issue: {edit['issue']}")


if __name__ == '__main__':
    main()
