#!/usr/bin/env python3
"""
Compute LSTM perplexity threshold from trained LSTM model on human training data.

This script:
1. Loads trained LSTM model (edit-level or strategy-level)
2. Loads human training sequences
3. Computes perplexity for each training sequence
4. Calculates Q3 + 1.5×IQR threshold (upper bound, since lower perplexity is better)
5. Saves threshold to file and logs to console
"""

import torch
import pickle
import numpy as np
from pathlib import Path
import logging
import argparse
import sys

# Add parent directory to path to import LSTM model
sys.path.insert(0, str(Path(__file__).parent))
from train_lstm_scorer import load_lstm_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_perplexities(model, sequences, max_length, device, batch_size=64):
    """
    Compute perplexity for all sequences.

    Args:
        model: Trained LSTM model
        sequences: List of integer sequences
        max_length: Maximum sequence length from training
        device: Device (cuda/cpu)
        batch_size: Batch size for processing

    Returns:
        perplexities: List of perplexity scores
    """
    model.eval()
    perplexities = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]

            # Pad sequences
            padded_batch = []
            lengths = []

            for seq in batch_sequences:
                length = min(len(seq), max_length)
                lengths.append(length)

                # Truncate if necessary
                seq = seq[:max_length]

                # Pad with 0
                padded = seq + [0] * (max_length - len(seq))
                padded_batch.append(padded)

            # Convert to tensors
            x = torch.tensor(padded_batch, dtype=torch.long, device=device)
            lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

            # Compute perplexity
            batch_perplexities = model.compute_perplexity(x, lengths_tensor)
            perplexities.extend(batch_perplexities.cpu().numpy())

    return perplexities


def compute_threshold(perplexities):
    """
    Compute Q3 + 1.5×IQR threshold.

    For perplexity, lower is better, so we use upper bound as threshold:
    - Perplexity <= threshold → human-like
    - Perplexity > threshold → not human-like

    Args:
        perplexities: List of perplexity scores

    Returns:
        threshold: Q3 + 1.5×IQR
        q1, q3, iqr: Quartile statistics
    """
    # Filter out inf values
    finite_perplexities = [p for p in perplexities if not np.isinf(p)]

    if not finite_perplexities:
        logger.error("All perplexities are infinite!")
        return None, None, None, None

    q1 = np.percentile(finite_perplexities, 25)
    q3 = np.percentile(finite_perplexities, 75)
    iqr = q3 - q1

    # Upper bound threshold (higher perplexity = less human-like)
    threshold = q3 + 1.5 * iqr

    return threshold, q1, q3, iqr


def main():
    parser = argparse.ArgumentParser(description="Compute LSTM perplexity threshold")
    parser.add_argument("--per-sentence", action="store_true",
                       help="Compute threshold for strategy-level (per-sentence) model")
    args = parser.parse_args()

    # Set paths based on mode
    if args.per_sentence:
        model_path = Path("scorers/local_scorers/human_like/models/lstm_model_per_sentence.pt")
        data_path = Path("scorers/local_scorers/human_like/data/train_sequences_per_sentence.pkl")
        output_file = Path("scorers/local_scorers/human_like/models/lstm_threshold_per_sentence.txt")
        mode = "strategy-level (per-sentence)"
    else:
        model_path = Path("scorers/local_scorers/human_like/models/lstm_model.pt")
        data_path = Path("scorers/local_scorers/human_like/data/train_sequences.pkl")
        output_file = Path("scorers/local_scorers/human_like/models/lstm_threshold_edit.txt")
        mode = "edit-level"

    print("=" * 80)
    print(f"COMPUTING LSTM PERPLEXITY THRESHOLD ({mode})")
    print("=" * 80)

    # Check if model exists
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.error(f"Please train the LSTM model first")
        if args.per_sentence:
            logger.error("Run: python train_lstm_scorer.py --per-sentence")
        else:
            logger.error("Run: python train_lstm_scorer.py")
        sys.exit(1)

    # Check if data exists
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error(f"Please prepare sequences first")
        if args.per_sentence:
            logger.error("Run: python prepare_sequence_data.py --per-sentence")
        else:
            logger.error("Run: python prepare_sequence_data.py")
        sys.exit(1)

    # Load model
    print(f"\nLoading LSTM model from {model_path.name}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, max_length = load_lstm_model(model_path, device)
    print(f"✓ Loaded model on {device}")
    print(f"  Max length: {max_length}")

    # Load training sequences
    print(f"\nLoading training sequences from {data_path.name}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    sequences = data['sequences']
    print(f"✓ Loaded {len(sequences)} training sequences")

    # Compute perplexities
    print(f"\nComputing perplexities for training sequences...")
    perplexities = compute_perplexities(model, sequences, max_length, device)

    # Filter out inf values for statistics
    finite_perplexities = [p for p in perplexities if not np.isinf(p)]
    inf_count = len(perplexities) - len(finite_perplexities)

    print(f"✓ Computed perplexities")
    print(f"  Finite values: {len(finite_perplexities)}")
    print(f"  Infinite values: {inf_count}")

    if finite_perplexities:
        print(f"\nPerplexity statistics (finite values only):")
        print(f"  Mean: {np.mean(finite_perplexities):.4f}")
        print(f"  Median: {np.median(finite_perplexities):.4f}")
        print(f"  Std: {np.std(finite_perplexities):.4f}")
        print(f"  Min: {np.min(finite_perplexities):.4f}")
        print(f"  Max: {np.max(finite_perplexities):.4f}")

    # Compute threshold
    print(f"\nComputing threshold using Q3 + 1.5×IQR...")
    threshold, q1, q3, iqr = compute_threshold(perplexities)

    if threshold is None:
        logger.error("Failed to compute threshold")
        sys.exit(1)

    print(f"✓ Computed threshold")
    print(f"\nThreshold Statistics:")
    print(f"  Q1 (25th percentile): {q1:.4f}")
    print(f"  Q3 (75th percentile): {q3:.4f}")
    print(f"  IQR: {iqr:.4f}")
    print(f"  Threshold (Q3 + 1.5×IQR): {threshold:.4f}")

    print(f"\nInterpretation:")
    print(f"  - Perplexity <= {threshold:.4f} → human-like edit")
    print(f"  - Perplexity > {threshold:.4f} → not human-like")
    print(f"  - Lower perplexity = more predictable = more human-like")

    # Calculate percentage of training data passing threshold
    passing = sum(1 for p in finite_perplexities if p <= threshold)
    total = len(finite_perplexities)
    print(f"\nValidation on training data:")
    print(f"  {passing}/{total} ({passing/total*100:.1f}%) pass threshold")

    # Save threshold to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(f"{threshold:.6f}\n")
        f.write(f"# LSTM Perplexity Threshold ({mode})\n")
        f.write(f"# Computed from {len(sequences)} training sequences\n")
        f.write(f"# Q1={q1:.6f}, Q3={q3:.6f}, IQR={iqr:.6f}\n")
        f.write(f"# Threshold = Q3 + 1.5×IQR = {threshold:.6f}\n")

    print(f"\n✓ Saved threshold to {output_file}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"\nThreshold: {threshold:.6f}")
    print(f"\nNext steps:")
    print(f"  1. Update human_like_scorer.py with this threshold")
    if args.per_sentence:
        print(f"     - Update strategy-level scorer threshold")
    else:
        print(f"     - Update edit-level scorer threshold (line ~31)")
    print(f"  2. Use this threshold for evaluation")


if __name__ == "__main__":
    main()
