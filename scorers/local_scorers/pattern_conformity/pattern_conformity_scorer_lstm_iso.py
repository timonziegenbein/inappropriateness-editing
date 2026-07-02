"""
Dual-Component Pattern Conformity Scorer

Uses TWO independent trained models to score edit pattern conformity:
1. ISO-only: Isolation Forest on token count features (5 dimensions)
2. LSTM-only: LSTM language model on edit operation sequences (perplexity-based)

Both components must pass their thresholds for an edit to be considered pattern conformity.
This provides complementary signals: ISO captures distributional patterns,
LSTM captures sequential predictability.
"""

import torch
import numpy as np
import difflib
import logging
import pickle
import json
from typing import List, Dict
from pathlib import Path
from transformers import AutoTokenizer
import weave

logger = logging.getLogger(__name__)


class PatternConformityScorer:
    def __init__(self, device,
                 model_path="scorers/local_scorers/pattern_conformity",
                 iso_threshold=0.186,
                 lstm_threshold=3.076,
                 max_len=None):
        """
        Initialize Dual-Component Pattern Conformity Scorer (Edit-Level).

        Args:
            device: torch device
            model_path: Directory containing trained models
            iso_threshold: ISO-only score threshold (default: 0.186, edit-level Q1-1.5×IQR)
            lstm_threshold: LSTM perplexity threshold (default: 3.076, edit-level Q3+1.5×IQR, lower is better)
            max_len: Maximum sequence length for LSTM
        """
        self.device = device
        self.model_dir = Path(model_path)
        self.iso_threshold = iso_threshold
        self.lstm_threshold = lstm_threshold

        # Load models (ISO-only and LSTM-only separately)
        self.lstm_model, self.max_length, self.iso_model, self.operation_vocab = self._load_models()
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

        logger.info(f"Loaded Dual-Component PatternConformityScorer:")
        logger.info(f"  ISO-only threshold: {iso_threshold}")
        logger.info(f"  LSTM perplexity threshold: {lstm_threshold} (lower is better)")

    def _load_models(self):
        """Load trained LSTM-only and ISO-only models (independent components)."""
        models_dir = self.model_dir / "models"

        # Load LSTM model
        lstm_path = models_dir / "lstm_model.pt"
        checkpoint = torch.load(lstm_path, map_location=self.device)

        # Import LSTM model class
        import sys
        sys.path.append(str(self.model_dir))
        from train_lstm_scorer import LSTMSequenceModel

        vocab_size = checkpoint['vocab_size']
        max_length = checkpoint['max_length']

        lstm_model = LSTMSequenceModel(
            vocab_size=vocab_size,
            embedding_dim=checkpoint.get('embedding_dim', 32),
            hidden_dim=checkpoint.get('hidden_dim', 64),
            num_layers=checkpoint.get('num_layers', 2),
            dropout=checkpoint.get('dropout', 0.2)
        )
        lstm_model.load_state_dict(checkpoint['model_state_dict'])
        lstm_model.to(self.device)
        lstm_model.eval()

        # Load ISO-only model
        iso_path = models_dir / "iso_only_iso_model.pkl"

        with open(iso_path, 'rb') as f:
            iso_model = pickle.load(f)

        # Load vocabulary (for converting tokens to IDs)
        # LSTM vocab: {keep: 1, keep-in-edit: 2, del: 3, add: 4, replace: 5, <PAD>: 0}
        operation_vocab = {
            'keep': 1,
            'keep-in-edit': 2,
            'del': 3,
            'add': 4,
            'replace': 5
        }

        logger.info(f"✓ Loaded LSTM model from {lstm_path.name}")
        logger.info(f"✓ Loaded ISO-only model from {iso_path.name}")
        logger.info(f"✓ LSTM vocab size: {vocab_size}, max_length: {max_length}")

        return lstm_model, max_length, iso_model, operation_vocab

    def _generate_sequence_for_edit(self, before_revision: str, start_char: int,
                                    end_char: int, rewritten_part: str) -> List[str]:
        """
        Generate edit operation sequence for a single edit using sequence alignment.

        Returns list of operations: ['keep', 'keep-in-edit', 'del', 'add', 'replace']
        """
        encoding = self.tokenizer(before_revision, return_offsets_mapping=True)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        offsets = encoding['offset_mapping']

        if len(tokens) == 0:
            return []

        # Find tokens overlapping with edit region
        token_start_index = -1
        token_end_index = -1
        for i, offset in enumerate(offsets):
            token_start, token_end = offset
            if start_char < token_end and end_char > token_start:
                if token_start_index == -1:
                    token_start_index = i
                token_end_index = i

        if token_start_index == -1:
            # No overlap, all tokens are 'keep'
            return ['keep'] * len(tokens)

        # Build sequence
        tags = []
        tags.extend(['keep'] * token_start_index)

        # Get tokens in edit region
        before_edit_tokens = tokens[token_start_index:token_end_index+1]
        if not isinstance(rewritten_part, str):
            rewritten_part = ""
        after_edit_tokens = self.tokenizer.tokenize(rewritten_part)

        # Use difflib to align and classify operations
        matcher = difflib.SequenceMatcher(None, before_edit_tokens, after_edit_tokens)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                tags.extend(['keep-in-edit'] * (i2 - i1))
            elif tag == 'delete':
                tags.extend(['del'] * (i2 - i1))
            elif tag == 'replace':
                tags.extend(['replace'] * (i2 - i1))
            elif tag == 'insert':
                tags.extend(['add'] * (j2 - j1))

        tags.extend(['keep'] * (len(tokens) - token_end_index - 1))

        return tags

    def _compute_token_counts(self, sequence: List[str]) -> np.ndarray:
        """Compute 5-dimensional token count feature vector."""
        counts = {
            'keep': 0,
            'keep-in-edit': 0,
            'del': 0,
            'add': 0,
            'replace': 0
        }

        for op in sequence:
            if op in counts:
                counts[op] += 1

        return np.array([
            counts['keep'],
            counts['keep-in-edit'],
            counts['del'],
            counts['add'],
            counts['replace']
        ])

    @weave.op()
    def _compute_lstm_score(self, sequence: List[str]) -> float:
        """
        Compute LSTM perplexity score for sequence.
        Lower perplexity = more pattern conformity (more predictable given training data).

        Returns:
            Perplexity score (lower is better)
        """
        # Convert sequence to integers using LSTM vocab
        int_seq = [self.operation_vocab.get(op, 0) for op in sequence]

        if len(int_seq) <= 1:
            return float('inf')  # Very high perplexity for degenerate sequences

        # Truncate if necessary
        int_seq = int_seq[:self.max_length]
        length = len(int_seq)

        # Pad to max_length
        padded_seq = int_seq + [0] * (self.max_length - length)

        # Convert to tensor
        x = torch.tensor([padded_seq], dtype=torch.long, device=self.device)
        lengths = torch.tensor([length], dtype=torch.long, device=self.device)

        try:
            # Compute perplexity
            perplexities = self.lstm_model.compute_perplexity(x, lengths)
            return perplexities[0].item()
        except Exception as e:
            logger.warning(f"Error computing LSTM perplexity: {e}")
            return float('inf')

    @weave.op()
    def _compute_iso_score(self, sequence: List[str]) -> float:
        """
        Compute ISO-only anomaly score using token counts.

        Returns:
            Score in [0, 1] where higher = more pattern conformity (less anomalous)
        """
        # Compute token count features ONLY (5 dimensions)
        token_counts = self._compute_token_counts(sequence)
        features = token_counts.reshape(1, -1)

        # Get anomaly score from ISO-only model
        iso_score = self.iso_model.score_samples(features)[0]

        # Normalize to [0, 1]
        # score_samples returns negative values, more negative = more anomalous
        # We need to map this to [0, 1] where 1 = most pattern conformity
        # Use sigmoid-like normalization
        normalized_score = 1 / (1 + np.exp(-iso_score))

        return normalized_score

    @weave.op()
    def calculate_pattern_conformity(self, original_argument: str, original_sentence: str,
                                 inappropriate_part: str, rewritten_part: str) -> float:
        """
        Calculate pattern conformity reward for an edit using BOTH components.

        Args:
            original_argument: Full original text (unused, kept for API compatibility)
            original_sentence: The sentence being edited
            inappropriate_part: The part being replaced
            rewritten_part: The replacement text

        Returns:
            Binary reward: 1.0 if BOTH components pass, 0.0 otherwise
        """
        # Find character positions within the sentence
        start_char_in_sentence = original_sentence.find(inappropriate_part)
        if start_char_in_sentence == -1:
            logger.warning(f"Could not find inappropriate part in sentence")
            return 0.0

        end_char_in_sentence = start_char_in_sentence + len(inappropriate_part)

        # Generate sequence from the sentence only (edit-level scorer is trained on sentences)
        sequence = self._generate_sequence_for_edit(original_sentence, start_char_in_sentence,
                                                    end_char_in_sentence, rewritten_part)

        if not sequence:
            return 0.0

        # Compute BOTH scores independently
        iso_score = self._compute_iso_score(sequence)
        lstm_score = self._compute_lstm_score(sequence)

        # BOTH components must pass their thresholds
        # ISO: higher is better (>= threshold)
        # LSTM: lower is better (<= threshold)
        iso_passes = iso_score >= self.iso_threshold
        lstm_passes = lstm_score <= self.lstm_threshold

        #if iso_passes and lstm_passes:
        if lstm_passes:
            return 1.0
        else:
            return 0.0
