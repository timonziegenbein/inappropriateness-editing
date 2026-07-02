#!/usr/bin/env python3
"""
Compute ISO threshold using Q1 - 1.5×IQR method on training data.

The ISO model outputs scores where higher = more pattern conformity (less anomalous).
We compute Q1 - 1.5×IQR as the lower bound threshold.
Scores >= threshold are considered pattern conformity.
"""

import sys
import pickle
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging
from tqdm import tqdm
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_predictions(file_path: str) -> List[Dict[str, Any]]:
    """Load predictions from a JSONL file."""
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def compute_iso_scores(predictions: List[Dict[str, Any]], iso_model, tokenizer) -> List[float]:
    """
    Compute ISO scores for all edits in the predictions.

    Returns list of ISO scores (one per edit).
    """
    from transformers import AutoTokenizer
    from compute_hmm_isolation_scores import (
        generate_edit_sequence_from_edits,
        extract_edit_features_from_sequence
    )

    scores = []

    for pred in tqdm(predictions, desc="Computing ISO scores"):
        original = pred.get("original_before_sent", "")
        edits = pred.get("parsed_edits", [])

        if not original or not edits:
            continue

        # Filter valid edits
        valid_edits = []
        for edit in edits:
            inappropriate_part = edit.get("inappropriate_part", "")
            rewritten_part = edit.get("rewritten_part", "")
            if inappropriate_part and inappropriate_part != rewritten_part:
                valid_edits.append(edit)

        if not valid_edits:
            continue

        # Process each edit separately (edit-level)
        for edit in valid_edits:
            # Generate sequence for single edit
            seq = generate_edit_sequence_from_edits(original, [edit], tokenizer)

            if not seq:
                continue

            # Check if sequence has any edits
            has_edits = any(tok != 'keep' for tok in seq)
            if not has_edits:
                continue

            # Extract token count features
            features = extract_edit_features_from_sequence(seq)
            feature_vector = np.array([[
                features['count_keep'],
                features['count_keep_in_edit'],
                features['count_del'],
                features['count_add'],
                features['count_replace'],
            ]])

            # Get ISO score
            iso_score = iso_model.score_samples(feature_vector)[0]

            # Normalize to [0, 1]
            # score_samples returns negative values, more negative = more anomalous
            normalized_score = 1 / (1 + np.exp(-iso_score))

            scores.append(normalized_score)

    return scores


def compute_threshold(scores: List[float]) -> tuple:
    """
    Compute Q1 - 1.5×IQR threshold.

    Returns (threshold, q1, q3, iqr)
    """
    scores_array = np.array(scores)
    q1 = np.percentile(scores_array, 25)
    q3 = np.percentile(scores_array, 75)
    iqr = q3 - q1
    threshold = q1 - 1.5 * iqr  # Lower bound (lower score = less pattern conformity)

    return threshold, q1, q3, iqr


def main():
    parser = argparse.ArgumentParser(description="Compute ISO threshold from training data")
    parser.add_argument("--per-sentence", action="store_true",
                       help="Compute threshold for strategy-level (per-sentence) model")
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    if args.per_sentence:
        print("\n" + "="*80)
        print("COMPUTING ISO THRESHOLD (strategy-level / per-sentence)")
        print("="*80)
        model_path = script_dir / "models" / "iso_only_iso_model_per_sentence.pkl"
        train_file = script_dir / "data" / "human_with_edits_train.jsonl"
        output_file = script_dir / "models" / "iso_threshold_strategy.txt"
    else:
        print("\n" + "="*80)
        print("COMPUTING ISO THRESHOLD (edit-level)")
        print("="*80)
        model_path = script_dir / "models" / "iso_only_iso_model.pkl"
        train_file = script_dir / "data" / "human_with_edits_train.jsonl"
        output_file = script_dir / "models" / "iso_threshold_edit.txt"

    # Load ISO model
    print(f"\nLoading ISO model from {model_path.name}...")
    with open(model_path, 'rb') as f:
        iso_model = pickle.load(f)
    print("✓ Loaded model")

    # Load tokenizer
    print("\nLoading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    print("✓ Loaded tokenizer")

    # Load training data
    print(f"\nLoading training data from {train_file.name}...")
    predictions = load_predictions(str(train_file))
    print(f"✓ Loaded {len(predictions)} training examples")

    # Compute ISO scores on training data
    print("\nComputing ISO scores for training data...")
    scores = compute_iso_scores(predictions, iso_model, tokenizer)
    print(f"✓ Computed {len(scores)} scores")

    # Compute statistics
    print("\nISO score statistics:")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Median: {np.median(scores):.4f}")
    print(f"  Std: {np.std(scores):.4f}")
    print(f"  Min: {np.min(scores):.4f}")
    print(f"  Max: {np.max(scores):.4f}")

    # Compute threshold using Q1 - 1.5×IQR
    print("\nComputing threshold using Q1 - 1.5×IQR...")
    threshold, q1, q3, iqr = compute_threshold(scores)
    print("✓ Computed threshold")

    print("\nThreshold Statistics:")
    print(f"  Q1 (25th percentile): {q1:.4f}")
    print(f"  Q3 (75th percentile): {q3:.4f}")
    print(f"  IQR: {iqr:.4f}")
    print(f"  Threshold (Q1 - 1.5×IQR): {threshold:.4f}")

    print("\nInterpretation:")
    print(f"  - ISO score >= {threshold:.4f} → pattern conformity edit")
    print(f"  - ISO score < {threshold:.4f} → not pattern conformity")
    print(f"  - Higher score = more pattern conformity")

    # Validation on training data
    num_pass = sum(1 for s in scores if s >= threshold)
    print(f"\nValidation on training data:")
    print(f"  {num_pass}/{len(scores)} ({num_pass/len(scores)*100:.1f}%) pass threshold")

    # Save threshold
    with open(output_file, 'w') as f:
        f.write(f"{threshold}\n")
    print(f"\n✓ Saved threshold to {output_file}")

    print("\n" + "="*80)
    print("DONE")
    print("="*80)
    print(f"\nThreshold: {threshold:.6f}")
    print("\nNext steps:")
    print("  1. Update pattern_conformity_scorer.py with this threshold")
    if args.per_sentence:
        print("     - Update strategy-level scorer threshold")
    else:
        print("     - Update edit-level scorer threshold (line ~30)")
    print("  2. Use this threshold for evaluation")


if __name__ == "__main__":
    main()
