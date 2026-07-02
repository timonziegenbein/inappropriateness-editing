#!/usr/bin/env python3
"""
Prepare edit sequence data for LSTM training.
Uses the same data format as compute_hmm_isolation_scores.py.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import numpy as np
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_predictions(file_path: str) -> List[Dict]:
    """Load predictions from a JSONL file."""
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def generate_edit_sequence_from_edits(original_text: str, edits: List[Dict], tokenizer) -> List[str]:
    """
    Generate edit operation sequence from pre-computed edits using sequence alignment.

    Uses token-level alignment to identify: keep, del, add, replace operations.
    'keep-in-edit' marks unchanged tokens within edit regions (from sequence alignment).

    This is the same function used in compute_hmm_isolation_scores.py.
    """
    import difflib

    if not edits:
        # No edits - all keep
        encoding = tokenizer(original_text, return_offsets_mapping=True)
        return ['keep'] * len(encoding['input_ids'])

    # Tokenize original text
    encoding = tokenizer(original_text, return_offsets_mapping=True)
    offsets = encoding['offset_mapping']

    # Initialize: all tokens are 'keep' by default
    sequence = ['keep'] * len(offsets)

    # Track insertions: dict mapping original_token_idx -> list of 'add' tokens to insert after it
    insertions_after = {}

    # Process each edit
    for edit in edits:
        inappropriate_part = edit.get("inappropriate_part", "")
        rewritten_part = edit.get("rewritten_part", "")

        # Skip insertions (empty inappropriate_part) - would need position tracking
        if not inappropriate_part or inappropriate_part.strip() == "":
            continue

        # Skip no-edits
        if inappropriate_part == rewritten_part:
            continue

        # Find character positions in original text
        start_char = original_text.find(inappropriate_part)
        if start_char == -1:
            continue
        end_char = start_char + len(inappropriate_part)

        # Tokenize both parts
        inap_tokens = tokenizer.tokenize(inappropriate_part)
        rew_tokens = tokenizer.tokenize(rewritten_part)

        # Find which original tokens overlap with this edit region
        affected_indices = []
        for i, (token_start, token_end) in enumerate(offsets):
            if token_start < end_char and token_end > start_char:
                affected_indices.append(i)

        if not affected_indices:
            continue

        # Use sequence alignment to find operations
        matcher = difflib.SequenceMatcher(None, inap_tokens, rew_tokens)

        # Build list of operations for original tokens and track insertions
        ops_for_original = []  # Operations for tokens in inappropriate_part

        orig_token_idx = 0  # Index into ops_for_original
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Tokens match - mark as keep-in-edit
                for _ in range(i2 - i1):
                    ops_for_original.append('keep-in-edit')
                    orig_token_idx += 1
            elif tag == 'delete':
                # Tokens deleted
                for _ in range(i2 - i1):
                    ops_for_original.append('del')
                    orig_token_idx += 1
            elif tag == 'insert':
                # Pure insertion - add 'add' tokens after previous original token
                num_adds = j2 - j1
                # Insert after the previous original token position
                if orig_token_idx > 0:
                    insert_after_local = orig_token_idx - 1
                else:
                    # Insert before first token
                    insert_after_local = 0

                if insert_after_local < len(affected_indices):
                    global_idx = affected_indices[insert_after_local]
                    if global_idx not in insertions_after:
                        insertions_after[global_idx] = []
                    insertions_after[global_idx].extend(['add'] * num_adds)
            elif tag == 'replace':
                # Replacement: mark originals as 'replace'
                for _ in range(i2 - i1):
                    ops_for_original.append('replace')
                    orig_token_idx += 1
                # Add 'add' tokens after the last replaced token
                num_adds = j2 - j1
                insert_after_local = orig_token_idx - 1
                if insert_after_local < len(affected_indices):
                    global_idx = affected_indices[insert_after_local]
                    if global_idx not in insertions_after:
                        insertions_after[global_idx] = []
                    insertions_after[global_idx].extend(['add'] * num_adds)

        # Apply operations to original token positions
        for i, token_idx in enumerate(affected_indices):
            if i < len(ops_for_original):
                sequence[token_idx] = ops_for_original[i]

    # Build final sequence with 'add' tokens inserted after their positions
    final_sequence = []
    for i, op in enumerate(sequence):
        final_sequence.append(op)
        # Insert any 'add' tokens that come after this position
        if i in insertions_after:
            final_sequence.extend(insertions_after[i])

    return final_sequence


def extract_sequences_from_jsonl(file_path: Path, tokenizer, per_edit: bool = True) -> List[List[str]]:
    """
    Extract edit sequences from JSONL file.

    Args:
        file_path: Path to JSONL file with predictions
        tokenizer: Tokenizer for processing text
        per_edit: If True (default), create one sequence per edit (matches inference).
                 If False, create one sequence per sentence (all edits combined).

    Returns:
        List of edit sequences (each sequence is a list of operation strings)
    """
    logger.info(f"Loading predictions from {file_path.name}...")
    predictions = load_predictions(str(file_path))

    sequences = []

    for pred in predictions:
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

        if per_edit:
            # Process each edit separately (matches inference behavior)
            for edit in valid_edits:
                seq = generate_edit_sequence_from_edits(original, [edit], tokenizer)
                # Filter out sequences with no edits (all 'keep')
                if any(tok != 'keep' for tok in seq):
                    sequences.append(seq)
        else:
            # Process all edits together (sentence-level)
            seq = generate_edit_sequence_from_edits(original, valid_edits, tokenizer)
            # Filter out sequences with no edits (all 'keep')
            if any(tok != 'keep' for tok in seq):
                sequences.append(seq)

    return sequences


def convert_sequences_to_ids(sequences: List[List[str]]) -> Tuple[List[List[int]], Dict[str, int]]:
    """
    Convert string sequences to integer IDs.

    Reserve 0 for padding, shift all tokens by +1.
    Vocabulary: {0: <PAD>, 1: keep, 2: keep-in-edit, 3: del, 4: add, 5: replace}
    """
    # Reserve 0 for padding, shift tokens by +1
    token_to_id = {
        'keep': 1,
        'keep-in-edit': 2,
        'del': 3,
        'add': 4,
        'replace': 5
    }

    id_sequences = []
    for seq in sequences:
        id_seq = [token_to_id.get(tok, 0) for tok in seq]
        id_sequences.append(id_seq)

    return id_sequences, token_to_id


def main():
    """Main data preparation script."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare LSTM training data")
    parser.add_argument("--per-sentence", action="store_true",
                       help="Process per-sentence (all edits combined) instead of per-edit")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    output_dir = data_dir

    # Set mode based on argument
    per_edit = not args.per_sentence

    # File paths (same as HMM script)
    train_file = data_dir / "human_with_edits_train.jsonl"
    test_file = data_dir / "human_with_edits_test.jsonl"
    model_files = {
        'coedit': data_dir / "coedit_with_edits.jsonl",
        'llama': data_dir / "llama_with_edits.jsonl",
        'gemini': data_dir / "gemini_with_edits.jsonl"
    }

    # Check if training file exists
    if not train_file.exists():
        logger.error(f"Training file not found: {train_file}")
        logger.error("Please ensure the data files are available")
        return

    logger.info("="*80)
    logger.info("PREPARING LSTM TRAINING DATA")
    logger.info("="*80)
    logger.info("\nUsing same data format as compute_hmm_isolation_scores.py")
    if per_edit:
        logger.info("Processing per-edit (matches inference behavior)")
    else:
        logger.info("Processing per-sentence (strategy-level, all edits combined)")

    # Initialize tokenizer (same as HMM script)
    logger.info("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.1-8B-Instruct")

    # Extract sequences from training data
    logger.info("\n" + "="*80)
    logger.info("EXTRACTING TRAINING SEQUENCES")
    logger.info("="*80)
    train_sequences = extract_sequences_from_jsonl(train_file, tokenizer, per_edit=per_edit)
    logger.info(f"Extracted {len(train_sequences)} training sequences")

    # Convert to integer IDs
    logger.info("\nConverting sequences to integer IDs...")
    train_sequences_ids, token_to_id = convert_sequences_to_ids(train_sequences)
    logger.info(f"Vocabulary: {token_to_id}")

    # Print sequence statistics
    lengths = [len(seq) for seq in train_sequences_ids]
    logger.info(f"\nSequence length statistics:")
    logger.info(f"  Min: {min(lengths)}")
    logger.info(f"  Max: {max(lengths)}")
    logger.info(f"  Mean: {np.mean(lengths):.2f}")
    logger.info(f"  Median: {np.median(lengths):.2f}")

    # Save training sequences
    suffix = "_per_sentence" if not per_edit else ""
    train_path = output_dir / f"train_sequences{suffix}.pkl"
    train_data = {
        'sequences': train_sequences_ids,
        'token_to_id': token_to_id,
        'vocab_size': len(token_to_id) + 1  # +1 for padding token at index 0
    }

    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    logger.info(f"\n✓ Saved training data to {train_path.name}")

    # Extract sequences from test data
    logger.info("\n" + "="*80)
    logger.info("EXTRACTING TEST SEQUENCES")
    logger.info("="*80)

    test_data = {}

    # Human test set
    if test_file.exists():
        human_test_sequences = extract_sequences_from_jsonl(test_file, tokenizer, per_edit=per_edit)
        human_test_sequences_ids, _ = convert_sequences_to_ids(human_test_sequences)
        test_data['human_sequences'] = human_test_sequences_ids
        logger.info(f"Human test: {len(human_test_sequences_ids)} sequences")

    # Model-generated test sets
    for source_name, source_file in model_files.items():
        if source_file.exists():
            model_sequences = extract_sequences_from_jsonl(source_file, tokenizer, per_edit=per_edit)
            model_sequences_ids, _ = convert_sequences_to_ids(model_sequences)
            test_data[f'{source_name}_sequences'] = model_sequences_ids
            logger.info(f"{source_name}: {len(model_sequences_ids)} sequences")

    # Add metadata
    test_data['token_to_id'] = token_to_id
    test_data['vocab_size'] = len(token_to_id) + 1  # +1 for padding token at index 0

    # Save test data
    test_path = output_dir / f"test_sequences{suffix}.pkl"
    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f)
    logger.info(f"\n✓ Saved test data to {test_path.name}")

    logger.info("\n" + "="*80)
    logger.info("DATA PREPARATION COMPLETE!")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("1. Train LSTM: python scorers/local_scorers/human_like/train_lstm_scorer.py")
    logger.info("2. Compare HMM vs LSTM: python scorers/local_scorers/human_like/compare_lstm_hmm.py")


if __name__ == "__main__":
    main()
