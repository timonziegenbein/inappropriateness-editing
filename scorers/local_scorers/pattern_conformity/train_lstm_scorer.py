#!/usr/bin/env python3
"""
Train LSTM-based sequence model for pattern conformity edit scoring.
This replaces the HMM component with a more flexible sequence model
that doesn't assume fixed-order Markov properties.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EditSequenceDataset(Dataset):
    """Dataset for edit operation sequences."""

    def __init__(self, sequences: List[List[int]], max_length: int = None):
        """
        Args:
            sequences: List of sequences, where each sequence is a list of token IDs
            max_length: Maximum sequence length (will pad/truncate)
        """
        self.sequences = sequences

        # Determine max_length if not provided
        if max_length is None:
            self.max_length = max(len(seq) for seq in sequences)
        else:
            self.max_length = max_length

        # Pad sequences
        self.padded_sequences = []
        self.lengths = []

        for seq in sequences:
            length = min(len(seq), self.max_length)
            self.lengths.append(length)

            # Truncate if necessary
            seq = seq[:self.max_length]

            # Pad with 0 (assuming 0 is padding token)
            padded = seq + [0] * (self.max_length - len(seq))
            self.padded_sequences.append(padded)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.padded_sequences[idx], dtype=torch.long),
            torch.tensor(self.lengths[idx], dtype=torch.long)
        )


class LSTMSequenceModel(nn.Module):
    """LSTM-based sequence model for scoring edit sequences."""

    def __init__(self, vocab_size: int, embedding_dim: int = 32,
                 hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        """
        Args:
            vocab_size: Number of unique tokens (edit operations)
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer (padding_idx=0 since 0 is reserved for padding)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer (predicts next token)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        """
        Args:
            x: Input sequences [batch_size, seq_len]
            lengths: Actual lengths of sequences [batch_size]

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
        """
        # Embed tokens
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)

        # Pack padded sequences for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM forward pass
        lstm_out, _ = self.lstm(packed)

        # Unpack sequences
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Apply dropout and output layer
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)  # [batch_size, seq_len, vocab_size]

        return logits

    def compute_perplexity(self, x, lengths):
        """
        Compute perplexity of sequences (lower = more pattern conformity).
        Uses the same logic as training loss computation.

        Args:
            x: Input sequences [batch_size, seq_len]
            lengths: Actual lengths of sequences [batch_size]

        Returns:
            perplexities: Per-sequence perplexity [batch_size]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, lengths)  # [batch_size, seq_len_unpacked, vocab_size]

            # Handle shape mismatch (same as training)
            batch_size, logits_seq_len, vocab_size_dim = logits.shape
            x_seq_len = x.size(1)

            if logits_seq_len >= x_seq_len:
                logits = logits[:, :x_seq_len-1, :].contiguous()
                logits_seq_len = x_seq_len - 1

            # Get targets (same as training)
            targets = x[:, 1:logits_seq_len+1].contiguous()

            # Flatten (same as training)
            logits_flat = logits.reshape(-1, vocab_size_dim)
            targets_flat = targets.reshape(-1)

            # Create mask (same as training)
            mask = torch.zeros(batch_size, logits_seq_len, dtype=torch.bool, device=x.device)
            for i, length in enumerate(lengths):
                if length > 1:
                    valid_len = min(length - 1, logits_seq_len)
                    mask[i, :valid_len] = True

            mask_flat = mask.reshape(-1)

            # Compute log probabilities
            log_probs_flat = torch.log_softmax(logits_flat, dim=-1)

            # Gather log probs for actual targets
            selected_log_probs = log_probs_flat.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

            # Compute per-sequence NLL
            perplexities = []
            start_idx = 0
            for i in range(batch_size):
                end_idx = start_idx + logits_seq_len
                seq_mask = mask_flat[start_idx:end_idx]
                seq_log_probs = selected_log_probs[start_idx:end_idx]

                # Get valid log probs
                valid_log_probs = seq_log_probs[seq_mask]

                if len(valid_log_probs) > 0:
                    # Mean NLL for this sequence
                    mean_nll = -valid_log_probs.mean()
                    perplexity = torch.exp(mean_nll)
                else:
                    perplexity = torch.tensor(float('inf'))

                perplexities.append(perplexity.item())
                start_idx = end_idx

            return torch.tensor(perplexities, device=x.device)

    def compute_score(self, x, lengths):
        """
        Compute normalized score (inverse of perplexity, scaled to [0, 1] range).
        Higher score = more pattern conformity.

        Args:
            x: Input sequences [batch_size, seq_len]
            lengths: Actual lengths of sequences [batch_size]

        Returns:
            scores: Per-sequence scores [batch_size]
        """
        perplexities = self.compute_perplexity(x, lengths)

        # Convert perplexity to score: score = exp(-perplexity)
        # This maps [1, inf] -> [1/e, 0]
        # We use exp(-log(perplexity)) = 1/perplexity for numerical stability
        scores = 1.0 / perplexities

        return scores


def train_lstm_model(
    sequences: List[List[int]],
    vocab_size: int,
    output_path: Path,
    embedding_dim: int = 32,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 64,
    num_epochs: int = 20,
    learning_rate: float = 0.001,
    device: str = None
):
    """
    Train LSTM sequence model on human edit sequences.

    Args:
        sequences: List of edit sequences (lists of token IDs)
        vocab_size: Number of unique tokens
        output_path: Where to save the trained model
        embedding_dim: Embedding dimension
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on (cuda/cpu)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f"Training LSTM model on {device}")
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Number of sequences: {len(sequences)}")

    # Create dataset and dataloader
    dataset = EditSequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = LSTMSequenceModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_x, batch_lengths in pbar:
            batch_x = batch_x.to(device)
            batch_lengths = batch_lengths.to(device)

            # Forward pass
            logits = model(batch_x, batch_lengths)  # [batch_size, seq_len, vocab_size]

            # Prepare targets (next token prediction)
            # logits has shape [batch_size, seq_len_unpacked, vocab_size] from LSTM output
            # batch_x has shape [batch_size, seq_len_padded] from dataloader
            # These might differ because LSTM unpacks to the max length in THIS batch

            batch_size, logits_seq_len, vocab_size_dim = logits.shape
            batch_x_seq_len = batch_x.size(1)

            # Handle case where logits is longer than batch_x
            # This can happen if unpacking gives a slightly longer sequence
            if logits_seq_len >= batch_x_seq_len:
                # Truncate logits to fit within batch_x bounds
                logits = logits[:, :batch_x_seq_len-1, :].contiguous()
                logits_seq_len = batch_x_seq_len - 1
                vocab_size_dim = logits.size(2)

            # Slice targets to match logits sequence length
            # We predict position i+1 from position i, so:
            # - Use batch_x[:, 1:logits_seq_len+1] as targets
            # - Use logits[:, :logits_seq_len, :] as predictions
            targets = batch_x[:, 1:logits_seq_len+1].contiguous()  # [batch_size, logits_seq_len]

            # Flatten for loss computation
            logits_flat = logits.reshape(-1, vocab_size_dim)  # [batch_size * logits_seq_len, vocab_size]
            targets_flat = targets.reshape(-1)  # [batch_size * logits_seq_len]

            # Create mask to ignore padding tokens
            mask = torch.zeros(batch_size, logits_seq_len, dtype=torch.bool, device=device)
            for i, length in enumerate(batch_lengths):
                # Mark valid positions (up to length-1, since we're predicting next token)
                if length > 1:
                    valid_len = min(length - 1, logits_seq_len)
                    mask[i, :valid_len] = True

            mask_flat = mask.reshape(-1)  # [batch_size * logits_seq_len]

            # Select only non-padded positions
            valid_logits = logits_flat[mask_flat]
            valid_targets = targets_flat[mask_flat]

            # Skip batch if no valid tokens
            if valid_logits.size(0) == 0:
                continue

            # Compute loss
            loss = criterion(valid_logits, valid_targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'max_length': dataset.max_length
    }, output_path)

    logger.info(f"Model saved to {output_path}")

    return model


def load_lstm_model(model_path: Path, device: str = None):
    """Load trained LSTM model from checkpoint."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(model_path, map_location=device)

    model = LSTMSequenceModel(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint['max_length']


if __name__ == "__main__":
    # Parse command line arguments
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Train LSTM sequence model")
    parser.add_argument("--per-sentence", action="store_true",
                       help="Train on per-sentence sequences (strategy-level) instead of per-edit (edit-level)")
    parser.add_argument("--num-epochs", type=int, default=20,
                       help="Number of training epochs (default: 20)")
    args = parser.parse_args()

    # Load sequences from your existing data format
    # This should match the format used in your HMM training
    if args.per_sentence:
        data_path = Path("scorers/local_scorers/pattern_conformity/data/train_sequences_per_sentence.pkl")
        output_path = Path("scorers/local_scorers/pattern_conformity/models/lstm_model_per_sentence.pt")
        logger.info("Training strategy-level (per-sentence) LSTM model")
    else:
        data_path = Path("scorers/local_scorers/pattern_conformity/data/train_sequences.pkl")
        output_path = Path("scorers/local_scorers/pattern_conformity/models/lstm_model.pt")
        logger.info("Training edit-level LSTM model")

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("Please prepare edit sequences first")
        if args.per_sentence:
            logger.error("Run: python prepare_sequence_data.py --per-sentence")
        else:
            logger.error("Run: python prepare_sequence_data.py")
        sys.exit(1)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    sequences = data['sequences']
    vocab_size = data['vocab_size']

    # Train model
    model = train_lstm_model(
        sequences=sequences,
        vocab_size=vocab_size,
        output_path=output_path,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        batch_size=64,
        num_epochs=args.num_epochs,
        learning_rate=0.001
    )

    logger.info("Training complete!")
