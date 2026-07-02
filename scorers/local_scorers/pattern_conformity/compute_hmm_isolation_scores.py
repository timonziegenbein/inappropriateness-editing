"""
Compute pattern conformity scores using hybrid HMM + Isolation Forest approach.

This replaces the perplexity-based scorer with:
1. HMM: Learns transition patterns in edit sequences
2. Isolation Forest: Detects anomalies in edit characteristics (sizes, ratios, etc.)

The hybrid score combines both: pattern-based + feature-based anomaly detection.

IMPORTANT: Training and evaluation now process ONE SAMPLE PER INDIVIDUAL EDIT.
- If a sentence has multiple edits, each edit gets its own training sample/score
- This matches the inference behavior where edits are scored independently
- Previous behavior: one sample per sentence (all edits combined) - MISMATCHED inference!
"""

import sys
import os
from pathlib import Path

# Go up to project root: pattern_conformity -> local_scorers -> scorers -> appropriateness-edit
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from sklearn.ensemble import IsolationForest
from hmmlearn import hmm
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_edit_sequence_from_edits(original_text: str, edits: List[Dict[str, Any]], tokenizer) -> List[str]:
    """
    Generate edit operation sequence from pre-computed edits using sequence alignment.

    Uses token-level alignment to identify: keep, del, add, replace operations.
    'keep-in-edit' marks unchanged tokens within edit regions (from sequence alignment).

    Returns list of operations including both original token ops and 'add' tokens:
    ['keep', 'keep-in-edit', 'del', 'add', 'replace', ...]

    The sequence represents the flow of operations:
    - Operations on original tokens (keep, del, replace, keep-in-edit)
    - 'add' tokens inserted after the position where they occur
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

    # Track which character positions have been edited
    edited_char_ranges = []

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

        edited_char_ranges.append((start_char, end_char))

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
                    # Insert before first token (we'll handle this by inserting after)
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


def compute_hmm_score_only(seq: List[str], hmm_model, operation_vocab) -> float:
    """
    Compute only the HMM score for a sequence.

    Returns 0.0 for no-edit sequences.
    """
    # Check if sequence has any edits (not all 'keep')
    has_edits = any(tok != 'keep' for tok in seq)

    if not has_edits:
        return 0.0

    int_seq = [operation_vocab.get(op, 0) for op in seq]
    max_len = 100
    if len(int_seq) > max_len:
        int_seq = int_seq[:max_len]

    seq_len = len(int_seq)
    X_hmm = np.array(int_seq).reshape(-1, 1)

    try:
        hmm_log_prob = hmm_model.score(X_hmm, lengths=[seq_len])
        # Normalize by sequence length
        hmm_score = np.exp(hmm_log_prob / seq_len)
        return hmm_score
    except:
        return 0.0


def extract_edit_features_from_sequence(seq: List[str]) -> Dict[str, float]:
    """
    Extract statistical features from edit sequence.

    Counts tokens of each type. Total length and ratios can be learned by Isolation Forest.
    Returns 5 core token count features.
    """
    from collections import Counter

    if not seq:
        return {
            'count_keep': 0,
            'count_keep_in_edit': 0,
            'count_del': 0,
            'count_add': 0,
            'count_replace': 0,
        }

    # Count tokens by type
    token_counts = Counter(seq)

    return {
        'count_keep': token_counts.get('keep', 0),
        'count_keep_in_edit': token_counts.get('keep-in-edit', 0),
        'count_del': token_counts.get('del', 0),
        'count_add': token_counts.get('add', 0),
        'count_replace': token_counts.get('replace', 0),
    }


def load_predictions(file_path: str) -> List[Dict[str, Any]]:
    """Load predictions from a JSONL file."""
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def process_single_prediction_for_training(pred: Dict[str, Any], tokenizer, hmm_model=None, operation_vocab=None, per_sentence=False) -> List[Tuple[List[str], List[float]]]:
    """
    Process a single prediction to extract sequences and features.

    Args:
        per_sentence: If True, combine all edits in a sentence into one sample (editing strategy).
                     If False (default), return one sample per individual edit (matches inference).

    Returns: List of (sequence, feature_vector) tuples
             - per_sentence=False: one tuple per edit
             - per_sentence=True: one tuple per sentence (all edits combined)

    If hmm_model is provided, computes HMM score and adds it as a feature.
    """
    original = pred.get("original_before_sent", "")
    edits = pred.get("parsed_edits", [])

    if not original or not edits:
        return []

    # Filter out empty or identical edits
    valid_edits = []
    for edit in edits:
        inappropriate_part = edit.get("inappropriate_part", "")
        rewritten_part = edit.get("rewritten_part", "")
        if inappropriate_part and inappropriate_part != rewritten_part:
            valid_edits.append(edit)

    if not valid_edits:
        return []

    results = []

    if per_sentence:
        # Process ALL edits together (sentence-level editing strategy)
        seq = generate_edit_sequence_from_edits(original, valid_edits, tokenizer)

        if not seq:
            return []

        # Extract features from combined sequence
        features = extract_edit_features_from_sequence(seq)
        feature_vector = [
            features['count_keep'],
            features['count_keep_in_edit'],
            features['count_del'],
            features['count_add'],
            features['count_replace'],
        ]

        # Add HMM score as a feature if HMM model is provided
        if hmm_model is not None and operation_vocab is not None:
            hmm_score = compute_hmm_score_only(seq, hmm_model, operation_vocab)
            feature_vector.append(hmm_score)

        results.append((seq, feature_vector))
    else:
        # Process EACH edit separately (per-edit, matches inference)
        for edit in valid_edits:
            # Generate sequence from single edit
            seq = generate_edit_sequence_from_edits(original, [edit], tokenizer)

            if not seq:
                continue

            # Extract features from sequence
            features = extract_edit_features_from_sequence(seq)
            feature_vector = [
                features['count_keep'],
                features['count_keep_in_edit'],
                features['count_del'],
                features['count_add'],
                features['count_replace'],
            ]

            # Add HMM score as a feature if HMM model is provided
            if hmm_model is not None and operation_vocab is not None:
                hmm_score = compute_hmm_score_only(seq, hmm_model, operation_vocab)
                feature_vector.append(hmm_score)

            results.append((seq, feature_vector))

    return results


def train_models(human_file: str, tokenizer, max_examples: int = None, num_workers: int = 4, iso_only: bool = False, hmm_only: bool = False, hmm_as_feature_only: bool = False, save_models: bool = False, model_prefix: str = "", load_hmm_if_exists: bool = False, per_sentence: bool = False):
    """
    Train HMM and/or Isolation Forest on human edits.

    TWO-PHASE TRAINING (default):
    1. Train HMM on sequences
    2. Add HMM scores as features and train Isolation Forest

    ISO-ONLY TRAINING (if iso_only=True):
    1. Skip HMM training
    2. Train Isolation Forest on token count features only

    HMM-ONLY TRAINING (if hmm_only=True):
    1. Train HMM on sequences
    2. Skip Isolation Forest training

    HMM-AS-FEATURE-ONLY (if hmm_as_feature_only=True):
    1. Train HMM on sequences
    2. Train Isolation Forest on ONLY HMM score (no token counts)

    Args:
        save_models: If True, save trained models to disk
        model_prefix: Prefix for saved model files (e.g., "iso_only_", "iso_hmm_")
        load_hmm_if_exists: If True, load existing HMM model instead of retraining

    Returns: (hmm_model, isolation_forest_model, operation_vocab)
             hmm_model will be None if iso_only=True
             isolation_forest_model will be None if hmm_only=True
    """
    import pickle
    script_dir = Path(__file__).parent

    # Try to load existing HMM model if requested and not in iso_only mode
    hmm_model = None
    operation_vocab = {'keep': 0, 'keep-in-edit': 1, 'del': 2, 'add': 3, 'replace': 4}

    if load_hmm_if_exists and not iso_only:
        # Look for any existing HMM model (try common prefixes)
        models_dir = script_dir / "models"
        possible_hmm_paths = [
            models_dir / "hmm_model.pkl",
            models_dir / "iso_hmm_hmm_model.pkl",
            models_dir / "hmm_only_hmm_model.pkl",
            models_dir / "hmm_as_iso_feature_hmm_model.pkl",
            # Legacy paths (fallback)
            script_dir / "hmm_model.pkl",
            script_dir / "iso_hmm_hmm_model.pkl",
        ]

        for hmm_path in possible_hmm_paths:
            if hmm_path.exists():
                logger.info(f"Loading existing HMM model from {hmm_path.name}...")
                with open(hmm_path, 'rb') as f:
                    hmm_model = pickle.load(f)
                logger.info("✓ HMM model loaded from disk")
                break

    logger.info("Loading human edits...")
    predictions = load_predictions(human_file)

    if max_examples:
        predictions = predictions[:max_examples]

    # Shuffle training data for better generalization
    import random
    random.seed(42)
    random.shuffle(predictions)
    logger.info(f"Shuffled {len(predictions)} training examples")

    if hmm_model is not None:
        logger.info(f"\n{'='*80}")
        logger.info("PHASE 1: Skipping HMM training (loaded from disk)")
        logger.info(f"{'='*80}")
    else:
        logger.info(f"\n{'='*80}")
        logger.info("PHASE 1: Training HMM on sequences")
        logger.info(f"{'='*80}")

    logger.info(f"Processing {len(predictions)} human examples with {num_workers} workers...")
    if per_sentence:
        logger.info(f"  Mode: PER-SENTENCE (all edits combined, evaluates editing strategy)")
    else:
        logger.info(f"  Mode: PER-EDIT (individual edits, matches inference)")

    # Phase 1: Extract sequences (without HMM scores yet)
    process_func = partial(process_single_prediction_for_training, tokenizer=tokenizer, hmm_model=None, operation_vocab=None, per_sentence=per_sentence)

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, predictions, chunksize=500),
            total=len(predictions),
            desc="Extracting sequences"
        ))

    # Filter out empty results and flatten (now returns list of tuples per prediction)
    sequences = []
    features_list_without_hmm = []

    for result_list in results:
        if result_list:  # result_list is now a list of (seq, features) tuples
            for seq, features in result_list:
                sequences.append(seq)
                features_list_without_hmm.append(features)

    if per_sentence:
        logger.info(f"Collected {len(sequences)} sentence-level sequences from {len(predictions)} sentences")
    else:
        logger.info(f"Collected {len(sequences)} individual edit sequences from {len(predictions)} sentences")

    if iso_only:
        # ISO-only mode: Skip HMM training
        logger.info("\nSkipping HMM training (ISO-only mode)")

        # Filter out no-edit examples
        features_list_final = []
        skipped = 0

        for seq, features_without_hmm in zip(sequences, features_list_without_hmm):
            has_edits = any(tok != 'keep' for tok in seq)
            if not has_edits:
                skipped += 1
                continue
            features_list_final.append(features_without_hmm)

        logger.info(f"  Skipped {skipped} no-edit examples")
        logger.info(f"  Created {len(features_list_final)} feature vectors")
        logger.info(f"  Feature dimensions: 5 token counts only")

        logger.info("\nTraining Isolation Forest on token count features ONLY...")
        X_features = np.array(features_list_final)

    else:
        # HMM-as-feature-only or Full mode: Train HMM if not already loaded
        if hmm_model is None:
            logger.info("Training HMM on token sequences...")

            # Convert sequences to integer representations
            # FILTER OUT NO-EDIT EXAMPLES (all 'keep' tokens)
            int_sequences = []
            lengths = []
            max_len = 100
            filtered_count = 0

            for seq in sequences:
                # Check if sequence has any edits (not all 'keep')
                has_edits = any(tok != 'keep' for tok in seq)
                if not has_edits:
                    filtered_count += 1
                    continue

                int_seq = [operation_vocab.get(op, 0) for op in seq]
                # Truncate if too long
                if len(int_seq) > max_len:
                    int_seq = int_seq[:max_len]
                int_sequences.append(int_seq)
                lengths.append(len(int_seq))

            logger.info(f"  Filtered out {filtered_count} no-edit examples ({filtered_count/len(sequences)*100:.1f}%)")
            logger.info(f"  Training on {len(int_sequences)} sequences with edits")
            logger.info(f"  Avg sequence length: {np.mean(lengths):.2f} tokens")

            # Calculate token distribution (only for sequences with edits)
            all_tokens = [tok for seq in sequences for tok in seq if any(t != 'keep' for t in seq)]
            from collections import Counter
            token_counts = Counter(all_tokens)
            logger.info(f"  Token distribution:")
            for tok, count in token_counts.most_common():
                logger.info(f"    {tok}: {count} ({count/len(all_tokens)*100:.1f}%)")

            # Concatenate all sequences (HMM expects concatenated sequences with lengths)
            X_hmm = np.concatenate([np.array(seq).reshape(-1, 1) for seq in int_sequences])

            # Use CategoricalHMM for discrete categorical observations
            # n_components = number of hidden states (different editing patterns)
            # n_features = 5 (keep, keep-in-edit, del, add, replace)
            logger.info(f"  Starting HMM training (100 iterations)...")
            hmm_model = hmm.CategoricalHMM(n_components=4, n_features=5, n_iter=100, random_state=42, verbose=True)
            hmm_model.fit(X_hmm, lengths=lengths)

            logger.info("✓ HMM trained")

        # Phase 2: Add HMM scores as features
        logger.info(f"\n{'='*80}")
        logger.info("PHASE 2: Adding HMM scores as features")
        logger.info(f"{'='*80}")

        features_list_with_hmm = []
        skipped = 0

        for seq, features_without_hmm in tqdm(zip(sequences, features_list_without_hmm),
                                               total=len(sequences),
                                               desc="Computing HMM scores"):
            # Compute HMM score
            hmm_score = compute_hmm_score_only(seq, hmm_model, operation_vocab)

            if hmm_score == 0.0 and not any(tok != 'keep' for tok in seq):
                # Skip no-edit examples
                skipped += 1
                continue

            # For hmm_as_feature_only: use ONLY HMM score
            # Otherwise: use token counts + HMM score
            if hmm_as_feature_only:
                features_with_hmm = [hmm_score]
            else:
                features_with_hmm = features_without_hmm + [hmm_score]
            features_list_with_hmm.append(features_with_hmm)

        logger.info(f"  Skipped {skipped} no-edit examples")
        logger.info(f"  Created {len(features_list_with_hmm)} feature vectors with HMM scores")
        if hmm_as_feature_only:
            logger.info(f"  Feature dimensions: 1 HMM score ONLY")
        else:
            logger.info(f"  Feature dimensions: 5 token counts + 1 HMM = 6 total")

        if hmm_as_feature_only:
            logger.info("\nTraining Isolation Forest on HMM SCORE ONLY...")
        else:
            logger.info("\nTraining Isolation Forest on features WITH HMM scores...")
        X_features = np.array(features_list_with_hmm)

    # Train Isolation Forest (unless hmm_only mode)
    if hmm_only:
        logger.info("\nSkipping Isolation Forest training (HMM-only mode)")
        iso_model = None
    else:
        iso_model = IsolationForest(
            contamination='auto',
            n_estimators=100,
            random_state=42
        )
        iso_model.fit(X_features)
        logger.info("✓ Isolation Forest trained")

    # Save models if requested
    if save_models:
        import pickle
        script_dir = Path(__file__).parent
        models_dir = script_dir / "models"
        models_dir.mkdir(exist_ok=True)

        # Add per-sentence suffix if applicable
        level_suffix = "_per_sentence" if per_sentence else ""

        if hmm_model is not None:
            hmm_path = models_dir / f"{model_prefix}hmm_model{level_suffix}.pkl"
            with open(hmm_path, 'wb') as f:
                pickle.dump(hmm_model, f)
            logger.info(f"✓ Saved HMM model to {hmm_path.name}")

        if iso_model is not None:
            iso_path = models_dir / f"{model_prefix}iso_model{level_suffix}.pkl"
            with open(iso_path, 'wb') as f:
                pickle.dump(iso_model, f)
            logger.info(f"✓ Saved Isolation Forest model to {iso_path.name}")

        # Save operation vocab
        vocab_path = models_dir / f"{model_prefix}operation_vocab{level_suffix}.json"
        with open(vocab_path, 'w') as f:
            json.dump(operation_vocab, f, indent=2)
        logger.info(f"✓ Saved operation vocabulary to {vocab_path.name}")

    return hmm_model, iso_model, operation_vocab


def compute_hybrid_score(original: str, edits: List[Dict[str, Any]], hmm_model, iso_model, operation_vocab, tokenizer, alpha=None, hmm_as_feature_only: bool = False, return_prediction: bool = False):
    """
    Compute pattern conformity score.

    Args:
        hmm_model: HMM model (can be None for ISO-only mode)
        iso_model: Isolation Forest model (can be None for HMM-only mode)
        alpha: Deprecated/ignored - HMM is now a feature in Isolation Forest
        hmm_as_feature_only: If True, use only HMM score as ISO feature (no token counts)
        return_prediction: If True, also return binary prediction (1=normal, -1=anomaly)

    Returns:
        If return_prediction=False: Score (higher = more pattern conformity), or None for no-edits
        If return_prediction=True: Tuple of (score, prediction), or None for no-edits
    """
    # Generate sequence from pre-computed edits
    seq = generate_edit_sequence_from_edits(original, edits, tokenizer)

    # Check if sequence has any edits (not all 'keep')
    has_edits = any(tok != 'keep' for tok in seq)

    if not has_edits:
        # No-edit example: skip it entirely
        return None

    # HMM-only mode: Return HMM score directly
    if iso_model is None:
        hmm_score = compute_hmm_score_only(seq, hmm_model, operation_vocab)
        if return_prediction:
            # HMM-only doesn't have binary predictions
            return (hmm_score, None)
        return hmm_score

    # Compute HMM score if needed
    if hmm_model is not None:
        hmm_score = compute_hmm_score_only(seq, hmm_model, operation_vocab)

    # Build feature vector based on mode
    if hmm_as_feature_only:
        # HMM-as-feature-only mode: Only HMM score
        feature_vector = np.array([[hmm_score]])
    elif hmm_model is not None:
        # Full mode: Token counts + HMM score
        features = extract_edit_features_from_sequence(seq)
        feature_vector = np.array([[
            features['count_keep'],
            features['count_keep_in_edit'],
            features['count_del'],
            features['count_add'],
            features['count_replace'],
            hmm_score,  # HMM score as 6th feature
        ]])
    else:
        # ISO-only mode: Token counts only
        features = extract_edit_features_from_sequence(seq)
        feature_vector = np.array([[
            features['count_keep'],
            features['count_keep_in_edit'],
            features['count_del'],
            features['count_add'],
            features['count_replace'],
        ]])

    # Get Isolation Forest anomaly score
    iso_score = iso_model.score_samples(feature_vector)[0]

    # Transform from [-inf, ~0] to [0, 1] range
    # Higher values = more normal (pattern conformity)
    iso_score_normalized = (iso_score + 1) / 2

    if return_prediction:
        # Get binary prediction: 1 = normal (inlier), -1 = anomaly (outlier)
        iso_prediction = iso_model.predict(feature_vector)[0]
        return (iso_score_normalized, iso_prediction)

    return iso_score_normalized


def process_single_prediction_for_scoring(pred: Dict[str, Any], per_sentence: bool = False) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Extract original text and pre-computed edits from prediction.

    Args:
        per_sentence: If True, return one tuple with all edits (sentence-level).
                     If False (default), return one tuple per edit.

    Returns: List of (original, edits) tuples
             - per_sentence=False: one tuple per edit, with edits as single-element list
             - per_sentence=True: one tuple per sentence, with all edits in list
    """
    original = pred.get("original_before_sent", "")
    edits = pred.get("parsed_edits", [])

    if not original or not edits:
        return []

    # Filter valid edits
    valid_edits = []
    for edit in edits:
        inappropriate_part = edit.get("inappropriate_part", "")
        rewritten_part = edit.get("rewritten_part", "")
        if inappropriate_part and inappropriate_part != rewritten_part:
            valid_edits.append(edit)

    if not valid_edits:
        return []

    if per_sentence:
        # Return one tuple with all edits
        return [(original, valid_edits)]
    else:
        # Return one tuple per edit (as single-element list for consistency)
        return [(original, [edit]) for edit in valid_edits]


def compute_scores(predictions: List[Dict[str, Any]], source_name: str, hmm_model, iso_model, operation_vocab, tokenizer, num_workers: int = 4, hmm_as_feature_only: bool = False, return_predictions: bool = False, per_sentence: bool = False):
    """Compute scores using Isolation Forest with HMM as a feature.

    Args:
        per_sentence: If True, score per-sentence (all edits combined).
                     If False (default), score per-edit.
        return_predictions: If True, return (scores, binary_predictions) tuple

    Returns:
        List[float] if return_predictions=False
        Tuple[List[float], List[int]] if return_predictions=True
    """
    logger.info(f"Extracting text and edits with {num_workers} workers...")

    # Extract text and edits in parallel (now returns list of lists)
    process_func = partial(process_single_prediction_for_scoring, per_sentence=per_sentence)
    with Pool(num_workers) as pool:
        data_pairs_nested = list(tqdm(
            pool.imap(process_func, predictions, chunksize=500),
            total=len(predictions),
            desc=f"Extracting {source_name}"
        ))

    # Flatten the nested list
    data_pairs = []
    for pair_list in data_pairs_nested:
        if pair_list:  # pair_list is now a list of (original, edits_list) tuples
            data_pairs.extend(pair_list)

    if per_sentence:
        logger.info(f"Computing scores for {len(data_pairs)} sentences from {len(predictions)} predictions...")
    else:
        logger.info(f"Computing scores for {len(data_pairs)} individual edits from {len(predictions)} sentences...")

    # Compute scores sequentially (model scoring is fast)
    scores = []
    binary_preds = []
    skipped_no_edits = 0
    for original, edits_list in tqdm(data_pairs, desc=f"Scoring {source_name}"):
        # Score with the edits_list (single edit or all edits combined)
        result = compute_hybrid_score(original, edits_list, hmm_model, iso_model, operation_vocab, tokenizer, hmm_as_feature_only=hmm_as_feature_only, return_prediction=return_predictions)
        if result is None:
            # No-edit example, skip it
            skipped_no_edits += 1
        else:
            if return_predictions:
                score, pred = result
                scores.append(score)
                binary_preds.append(pred)
            else:
                scores.append(result)

    logger.info(f"  Skipped {skipped_no_edits} no-edit examples")
    if per_sentence:
        logger.info(f"  Scored {len(scores)} sentences")
    else:
        logger.info(f"  Scored {len(scores)} individual edits")

    if return_predictions:
        return scores, binary_preds
    return scores


def compute_statistics(scores: List[float]) -> Dict[str, float]:
    """Compute statistics for a list of scores."""
    if not scores:
        return {"count": 0}

    return {
        "count": len(scores),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "p25": float(np.percentile(scores, 25)),
        "p50": float(np.percentile(scores, 50)),
        "p75": float(np.percentile(scores, 75)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99))
    }


def main():
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Compute HMM + Isolation Forest hybrid scores")
    parser.add_argument("--train-file", type=str, default=str(script_dir / "data" / "human_with_edits_train.jsonl"),
                        help="Training file (human edits)")
    parser.add_argument("--test-file", type=str, default=str(script_dir / "data" / "human_with_edits_test.jsonl"),
                        help="Test file (human edits)")
    parser.add_argument("--coedit-file", type=str, default=str(script_dir / "data" / "coedit_with_edits.jsonl"))
    parser.add_argument("--llama-file", type=str, default=str(script_dir / "data" / "llama_with_edits.jsonl"))
    parser.add_argument("--gemini-file", type=str, default=str(script_dir / "data" / "gemini_with_edits.jsonl"))
    parser.add_argument("--output-file", type=str, default=str(script_dir / "outputs" / "hmm_isolation_scores.json"))
    parser.add_argument("--max-train-examples", type=int, default=None,
                        help="Maximum human examples for training")
    parser.add_argument("--max-score-examples", type=int, default=None,
                        help="Maximum examples to score per source")
    parser.add_argument("--num-workers", type=int, default=max(1, cpu_count() - 1),
                        help="Number of parallel workers")
    parser.add_argument("--iso-only", action="store_true",
                        help="Use ISO-only (no HMM feature)")
    parser.add_argument("--hmm-only", action="store_true",
                        help="Use HMM-only (no ISO, just HMM scores)")
    parser.add_argument("--hmm-as-feature-only", action="store_true",
                        help="Use ISO with ONLY HMM score as feature (no token counts)")
    parser.add_argument("--save-models", action="store_true",
                        help="Save trained models to disk")
    parser.add_argument("--model-prefix", type=str, default="",
                        help="Prefix for saved model files")
    parser.add_argument("--load-hmm", action="store_true",
                        help="Load existing HMM model if available instead of retraining")
    parser.add_argument("--per-sentence", action="store_true",
                        help="Score per-sentence (all edits combined) instead of per-edit. "
                        "Evaluates overall editing strategy rather than individual edits.")

    args = parser.parse_args()

    if sum([args.iso_only, args.hmm_only, args.hmm_as_feature_only]) > 1:
        logger.error("Cannot use multiple mode flags together")
        return

    if args.iso_only:
        logger.info("="*80)
        logger.info("ISOLATION FOREST ONLY (TOKEN-BASED FEATURES)")
        logger.info("Using 5 token count features WITHOUT HMM")
        logger.info("="*80)
    elif args.hmm_only:
        logger.info("="*80)
        logger.info("HMM ONLY")
        logger.info("Using HMM scores on edit sequences only")
        logger.info("="*80)
    elif args.hmm_as_feature_only:
        logger.info("="*80)
        logger.info("ISOLATION FOREST WITH HMM SCORE ONLY")
        logger.info("Using ONLY HMM score as ISO feature (no token counts)")
        logger.info("="*80)
    else:
        logger.info("="*80)
        logger.info("ISOLATION FOREST + HMM SCORER")
        logger.info("HMM score is used as a feature in Isolation Forest")
        logger.info("="*80)

    logger.info("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.1-8B-Instruct")

    # Train on train set
    logger.info(f"\n{'='*80}")
    logger.info("TRAINING ON TRAIN SET")
    logger.info(f"{'='*80}")
    logger.info(f"Train file: {args.train_file}")

    hmm_model, iso_model, operation_vocab = train_models(
        args.train_file,
        tokenizer,
        max_examples=args.max_train_examples,
        num_workers=args.num_workers,
        iso_only=args.iso_only,
        hmm_only=args.hmm_only,
        hmm_as_feature_only=args.hmm_as_feature_only,
        save_models=args.save_models,
        model_prefix=args.model_prefix,
        load_hmm_if_exists=args.load_hmm,
        per_sentence=args.per_sentence
    )

    # Evaluate on test set
    logger.info(f"\n{'='*80}")
    logger.info("EVALUATING ON TEST SET")
    logger.info(f"{'='*80}")

    scores_dict = {}
    statistics_dict = {}

    sources = [
        ("human", args.test_file),
        ("coedit", args.coedit_file),
        ("llama", args.llama_file),
        ("gemini", args.gemini_file)
    ]

    for source_name, file_path in sources:
        if not os.path.exists(file_path):
            logger.warning(f"Skipping {source_name} - file not found")
            continue

        logger.info(f"\n" + "="*80)
        logger.info(f"SCORING {source_name.upper()}")
        logger.info("="*80)

        predictions = load_predictions(file_path)
        logger.info(f"Loaded {len(predictions)} predictions")

        if args.max_score_examples:
            predictions = predictions[:args.max_score_examples]
            logger.info(f"Limited to {args.max_score_examples} examples")

        scores = compute_scores(predictions, source_name, hmm_model, iso_model, operation_vocab, tokenizer, num_workers=args.num_workers, hmm_as_feature_only=args.hmm_as_feature_only, per_sentence=args.per_sentence)
        stats = compute_statistics(scores)

        scores_dict[source_name] = scores
        statistics_dict[source_name] = stats

        logger.info(f"\nStatistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

    # Save results
    output = {
        "perplexities": scores_dict,  # Keep same key name for compatibility with visualization
        "statistics": statistics_dict
    }

    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n✓ Saved results to {args.output_file}")

    # Print comparison
    logger.info(f"\n" + "="*80)
    logger.info("COMPARISON")
    logger.info("="*80)

    for source in scores_dict.keys():
        stats = statistics_dict[source]
        logger.info(f"\n{source.upper()}:")
        logger.info(f"  Mean: {stats.get('mean', 0):.4f}")
        logger.info(f"  Median: {stats.get('median', 0):.4f}")
        logger.info(f"  95th percentile: {stats.get('p95', 0):.4f}")


if __name__ == "__main__":
    main()
