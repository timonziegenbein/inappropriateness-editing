#!/usr/bin/env python3
"""Debug perplexity computation."""

import torch
import pickle
from pathlib import Path
from train_lstm_scorer import load_lstm_model, EditSequenceDataset
import sys

sys.path.insert(0, 'scorers/local_scorers/human_like')

# Load model
print("Loading model...")
model, max_len = load_lstm_model(
    Path('scorers/local_scorers/human_like/models/lstm_model.pt'),
    'cuda'
)

# Load one test sequence
print("Loading test data...")
with open('scorers/local_scorers/human_like/data/test_sequences.pkl', 'rb') as f:
    data = pickle.load(f)

# Test on first few sequences
for i in range(min(5, len(data['human_sequences']))):
    seq = data['human_sequences'][i]
    print(f'\n=== Sequence {i} ===')
    print(f'Length: {len(seq)}')
    print(f'Tokens: {seq[:20]}...')

    # Create dataset
    dataset = EditSequenceDataset([seq], max_length=max_len)
    x, length = dataset[0]
    x = x.unsqueeze(0).cuda()
    length = length.unsqueeze(0).cuda()

    # Compute perplexity
    perp = model.compute_perplexity(x, length)
    print(f'Perplexity: {perp.item():.4f}')

print('\n=== Expected ===')
print('Training loss was ~0.4, so expected perplexity: exp(0.4) ≈ 1.5')
