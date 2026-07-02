"""
Check edit counts in test data to understand the discrepancy.
"""

import json
from pathlib import Path

script_dir = Path(__file__).parent

total_sentences = 0
sentences_with_any_edits = 0
sentences_with_valid_edits = 0
total_valid_edits = 0

with open(script_dir / 'data/human_with_edits_test.jsonl', 'r') as f:
    for line in f:
        pred = json.loads(line)
        total_sentences += 1

        edits = pred.get('parsed_edits', [])
        if edits:
            sentences_with_any_edits += 1

        # Match the filtering logic from compute_hmm_isolation_scores.py line 688
        valid_edits = [e for e in edits if e.get('inappropriate_part') and
                      e.get('inappropriate_part') != e.get('rewritten_part')]
        if valid_edits:
            sentences_with_valid_edits += 1
            total_valid_edits += len(valid_edits)

print('='*80)
print('HUMAN TEST DATA ANALYSIS')
print('='*80)
print()
print(f'Total sentences: {total_sentences}')
print(f'Sentences with any edits: {sentences_with_any_edits}')
print(f'Sentences with valid edits: {sentences_with_valid_edits}')
print(f'Total valid edits: {total_valid_edits}')
print()
print(f'Expected from outputs:')
print(f'  Edit-level samples: 14,023')
print(f'  Strategy-level samples: 13,070')
print()
print(f'Discrepancy:')
print(f'  Edit-level: {14023 - total_valid_edits} missing')
print(f'  Strategy-level: {13070 - sentences_with_valid_edits} missing')
