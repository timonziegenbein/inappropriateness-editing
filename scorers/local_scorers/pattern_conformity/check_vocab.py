#!/usr/bin/env python3
"""Check vocabulary and sample sequences."""

import pickle

with open('scorers/local_scorers/pattern_conformity/data/test_sequences.pkl', 'rb') as f:
    data = pickle.load(f)

print('Vocabulary:', data['token_to_id'])
print('Vocab size:', data['vocab_size'])
print('\nFirst 5 human sequences (first 30 tokens each):')
for i in range(min(5, len(data['human_sequences']))):
    seq = data['human_sequences'][i]
    print(f'  Seq {i} (len={len(seq)}): {seq[:30]}')
