#!/usr/bin/env python3
"""
Script to create annotation pairs for the appropriateness study interface.
Generates pairwise combinations of rewrites from different model predictions.
"""

import json
import csv
import random
from pathlib import Path
from itertools import combinations
from collections import defaultdict

# Define the prediction files and their approach names
PREDICTION_FILES = {
    "v11_only_lm": "models/predictions/grpo_global_sentence_v11_only_lm.jsonl",
    "v11_r11_ppo": "models/predictions/grpo_global_sentence_v11_r11_ppo_classifier.jsonl",
    "ppo_50a_50ss": "models/predictions/ppo_50a_50ss.jsonl",
    "base_model": "models/predictions/sentence_base_model_ppo_classifier.jsonl",
}

OUTPUT_FILE = "annotation-interface/appropriateness-study-heroku/data/study_pairs.csv"

# Set random seed for reproducibility
RANDOM_SEED = 42
# Number of posts to randomly sample (None = use all)
SAMPLE_SIZE = 100


def load_predictions(file_path):
    """Load predictions from a JSONL file."""
    predictions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


def extract_perfect_edits(prediction):
    """Extract only the edits that were actually applied (perfect=1.0)."""
    if 'edits' not in prediction:
        return []

    perfect_edits = []
    for edit in prediction['edits']:
        # Only include edits that were marked as "perfect" (passed all scorers)
        if edit.get('valid', False) and edit.get('rewards', {}).get('perfect', 0.0) == 1.0:
            perfect_edits.append({
                'inappropriate_part': edit['inappropriate_part'],
                'rewritten_part': edit['rewritten_part'],
                'reason': edit.get('reason', 'Unknown')
            })
    return perfect_edits


def create_study_pairs():
    """Create pairwise combinations of rewrites from different approaches."""
    # Load all predictions, organized by approach
    all_predictions = {}
    for approach_name, file_path in PREDICTION_FILES.items():
        print(f"Loading {approach_name} from {file_path}...")
        all_predictions[approach_name] = load_predictions(file_path)
        print(f"  Loaded {len(all_predictions[approach_name])} predictions")

    # Organize predictions by post_id for each approach
    predictions_by_post = defaultdict(dict)
    for approach_name, predictions in all_predictions.items():
        for pred in predictions:
            post_id = pred['post_id']
            predictions_by_post[post_id][approach_name] = {
                'argument': pred['argument'],
                'argument_after_edits': pred['argument_after_edits'],
                'issue': pred['issue'],
                'edits': extract_perfect_edits(pred)
            }

    # Randomly sample posts if SAMPLE_SIZE is set
    all_post_ids = list(predictions_by_post.keys())
    if SAMPLE_SIZE and SAMPLE_SIZE < len(all_post_ids):
        random.seed(RANDOM_SEED)
        sampled_post_ids = random.sample(all_post_ids, SAMPLE_SIZE)
        print(f"\nRandomly sampled {SAMPLE_SIZE} posts out of {len(all_post_ids)} (seed={RANDOM_SEED})")
    else:
        sampled_post_ids = all_post_ids
        print(f"\nUsing all {len(all_post_ids)} posts")

    # Create pairwise combinations
    study_pairs = []
    approach_names = list(PREDICTION_FILES.keys())

    for post_id in sampled_post_ids:
        approaches_data = predictions_by_post[post_id]
        # Only create pairs if we have data from multiple approaches
        available_approaches = [a for a in approach_names if a in approaches_data]

        if len(available_approaches) < 2:
            print(f"Warning: Post {post_id} has data from only {len(available_approaches)} approach(es), skipping")
            continue

        # Create all pairwise combinations
        for approach_a, approach_b in combinations(available_approaches, 2):
            data_a = approaches_data[approach_a]
            data_b = approaches_data[approach_b]

            # Verify that the source argument and issue are the same
            if data_a['argument'] != data_b['argument']:
                print(f"Warning: Post {post_id} has different source arguments for {approach_a} and {approach_b}")
            if data_a['issue'] != data_b['issue']:
                print(f"Warning: Post {post_id} has different issues for {approach_a} and {approach_b}")

            # Create the pair
            pair = {
                'id': f"{post_id}_{approach_a}_{approach_b}",
                'source': data_a['argument'],
                'rewrite_a': data_a['argument_after_edits'],
                'rewrite_b': data_b['argument_after_edits'],
                'issue': data_a['issue'],
                'batch': 1,
                'edits_a': json.dumps(data_a['edits']),  # Store as JSON string
                'edits_b': json.dumps(data_b['edits'])   # Store as JSON string
            }
            study_pairs.append(pair)

    print(f"\nCreated {len(study_pairs)} pairwise combinations")
    return study_pairs


def write_csv(study_pairs, output_file):
    """Write study pairs to CSV file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['id', 'source', 'rewrite_a', 'rewrite_b', 'issue', 'batch', 'edits_a', 'edits_b']

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(study_pairs)

    print(f"Wrote {len(study_pairs)} pairs to {output_file}")


def main():
    """Main function."""
    print("Creating study pairs for annotation interface...\n")

    study_pairs = create_study_pairs()

    if study_pairs:
        write_csv(study_pairs, OUTPUT_FILE)
        print("\nDone!")
        print(f"\nSummary:")
        print(f"  Total pairs created: {len(study_pairs)}")

        # Count unique post IDs
        unique_posts = len(set(pair['id'].split('_')[0] for pair in study_pairs))
        print(f"  Unique posts: {unique_posts}")

        # Count pairs per approach combination
        from collections import Counter
        approach_combinations = Counter('_'.join(pair['id'].split('_')[1:]) for pair in study_pairs)
        print(f"\nPairs per approach combination:")
        for combo, count in sorted(approach_combinations.items()):
            print(f"  {combo}: {count}")
    else:
        print("No study pairs created!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
