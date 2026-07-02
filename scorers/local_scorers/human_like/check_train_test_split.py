"""
Check if the 14,023 samples include both train and test data.
"""

import json
from pathlib import Path

script_dir = Path(__file__).parent

def count_valid_edits(filepath):
    """Count valid edits in a file."""
    total_sentences = 0
    sentences_with_valid_edits = 0
    total_valid_edits = 0

    with open(filepath, 'r') as f:
        for line in f:
            pred = json.loads(line)
            total_sentences += 1

            edits = pred.get('parsed_edits', [])
            # Match the filtering logic from compute_hmm_isolation_scores.py line 688
            valid_edits = [e for e in edits if e.get('inappropriate_part') and
                          e.get('inappropriate_part') != e.get('rewritten_part')]

            if valid_edits:
                sentences_with_valid_edits += 1
                total_valid_edits += len(valid_edits)

    return total_sentences, sentences_with_valid_edits, total_valid_edits

print('='*80)
print('TRAIN/TEST SPLIT ANALYSIS')
print('='*80)
print()

# Check train
train_file = script_dir / 'data/human_with_edits_train.jsonl'
if train_file.exists():
    train_total, train_sentences, train_edits = count_valid_edits(train_file)
    print(f'TRAIN (human_with_edits_train.jsonl):')
    print(f'  Total sentences: {train_total}')
    print(f'  Sentences with valid edits: {train_sentences}')
    print(f'  Total valid edits: {train_edits}')
    print()

# Check test
test_file = script_dir / 'data/human_with_edits_test.jsonl'
if test_file.exists():
    test_total, test_sentences, test_edits = count_valid_edits(test_file)
    print(f'TEST (human_with_edits_test.jsonl):')
    print(f'  Total sentences: {test_total}')
    print(f'  Sentences with valid edits: {test_sentences}')
    print(f'  Total valid edits: {test_edits}')
    print()

# Combined
if train_file.exists() and test_file.exists():
    combined_sentences = train_sentences + test_sentences
    combined_edits = train_edits + test_edits

    print(f'COMBINED (train + test):')
    print(f'  Sentences with valid edits: {combined_sentences}')
    print(f'  Total valid edits: {combined_edits}')
    print()

    print(f'COMPARISON TO OUTPUTS:')
    print(f'  Edit-level samples in outputs: 14,023')
    print(f'  Combined valid edits: {combined_edits}')
    print(f'  Difference: {14023 - combined_edits}')
    print()

    print(f'  Strategy-level samples in outputs: 13,070')
    print(f'  Combined sentences with valid edits: {combined_sentences}')
    print(f'  Difference: {13070 - combined_sentences}')
    print()

    if abs(14023 - combined_edits) < 100:
        print('✓ Edit-level outputs match train+test combined')
    if abs(13070 - combined_sentences) < 100:
        print('✓ Strategy-level outputs match train+test combined')
