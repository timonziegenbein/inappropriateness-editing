"""
Analyze instance-level agreement between edit-level and strategy-level classifications.

For each sentence, we compare:
- Edit-level: Does every individual edit in the sentence pass? (all edits must pass)
- Strategy-level: Does the sentence-level aggregation pass?

This tells us if strategy-level captures the same filtering decisions as edit-level.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

script_dir = Path(__file__).parent

print("="*80)
print("INSTANCE-LEVEL AGREEMENT: EDIT vs STRATEGY")
print("="*80)
print()

# We need to load the raw prediction data to match edits to sentences
# The strategy-level data has one score per sentence
# The edit-level data has one score per edit
# We need to load the original predictions to group edits by sentence

import sys
sys.path.append(str(script_dir))

# Load the test data with edits
test_files = {
    'human': script_dir / 'data' / 'human_with_edits_test.jsonl',
    'coedit': script_dir / 'data' / 'coedit_with_edits.jsonl',
    'llama': script_dir / 'data' / 'llama_with_edits.jsonl',
    'gemini': script_dir / 'data' / 'gemini_with_edits.jsonl',
}

# Load edit-level and strategy-level results
edit_iso_path = script_dir / "outputs" / "iso_only_token_features.json"
edit_hmm_path = script_dir / "outputs" / "hmm_only_token_features.json"
strategy_iso_path = script_dir / "outputs" / "per_sentence_iso_only_token_features.json"
strategy_hmm_path = script_dir / "outputs" / "per_sentence_hmm_only_token_features.json"

with open(edit_iso_path, 'r') as f:
    edit_iso = json.load(f)['perplexities']
with open(edit_hmm_path, 'r') as f:
    edit_hmm = json.load(f)['perplexities']
with open(strategy_iso_path, 'r') as f:
    strategy_iso = json.load(f)['perplexities']
with open(strategy_hmm_path, 'r') as f:
    strategy_hmm = json.load(f)['perplexities']

# Compute IQR-based thresholds
def compute_iqr_threshold(scores):
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr

edit_iso_threshold = compute_iqr_threshold(edit_iso['human'])
edit_hmm_threshold = compute_iqr_threshold(edit_hmm['human'])
strategy_iso_threshold = compute_iqr_threshold(strategy_iso['human'])
strategy_hmm_threshold = compute_iqr_threshold(strategy_hmm['human'])

print("THRESHOLDS:")
print(f"  Edit ISO: {edit_iso_threshold:.4f}")
print(f"  Edit HMM: {edit_hmm_threshold:.4f}")
print(f"  Strategy ISO: {strategy_iso_threshold:.4f}")
print(f"  Strategy HMM: {strategy_hmm_threshold:.4f}")
print()

sources = ['human', 'coedit', 'llama', 'gemini']

for source in sources:
    print("="*80)
    print(f"{source.upper()}: INSTANCE-LEVEL AGREEMENT")
    print("="*80)
    print()

    # Load predictions
    if not test_files[source].exists():
        print(f"  ⚠ Test file not found: {test_files[source]}")
        print()
        continue

    predictions = []
    with open(test_files[source], 'r') as f:
        for line in f:
            predictions.append(json.loads(line))

    print(f"  Loaded {len(predictions)} sentences")
    print()

    # Process predictions to count edits per sentence
    edit_counts = []
    sentences_with_edits = 0
    total_edits = 0

    for pred in predictions:
        edits = pred.get('parsed_edits', [])
        # Match the filtering logic from compute_hmm_isolation_scores.py line 688
        valid_edits = [e for e in edits if e.get('inappropriate_part') and
                      e.get('inappropriate_part') != e.get('rewritten_part')]

        if valid_edits:
            sentences_with_edits += 1
            edit_counts.append(len(valid_edits))
            total_edits += len(valid_edits)

    print(f"  Sentences with edits: {sentences_with_edits}")
    print(f"  Total edits: {total_edits}")
    print(f"  Average edits per sentence: {total_edits/sentences_with_edits:.2f}")
    print()

    # Now reconstruct the classifications
    # Edit-level: each edit gets a score, sentence passes if ALL edits pass
    # Strategy-level: sentence gets one score, sentence passes if that score passes

    edit_iso_scores = edit_iso[source]
    edit_hmm_scores = edit_hmm[source]
    strategy_iso_scores = strategy_iso[source]
    strategy_hmm_scores = strategy_hmm[source]

    # Build sentence-level classifications
    # For edit-level: group edits by sentence and require ALL to pass
    edit_idx = 0
    strategy_idx = 0

    edit_level_passes = []  # One per sentence: does every edit pass?
    strategy_level_passes = []  # One per sentence: does sentence score pass?

    for pred in predictions:
        edits = pred.get('parsed_edits', [])
        # Match the filtering logic from compute_hmm_isolation_scores.py line 688
        valid_edits = [e for e in edits if e.get('inappropriate_part') and
                      e.get('inappropriate_part') != e.get('rewritten_part')]

        if not valid_edits:
            continue  # Skip sentences with no edits

        # Edit-level: check if ALL edits in this sentence pass
        sentence_edits_pass = []
        for _ in valid_edits:
            if edit_idx < len(edit_iso_scores) and edit_idx < len(edit_hmm_scores):
                iso_pass = edit_iso_scores[edit_idx] >= edit_iso_threshold
                hmm_pass = edit_hmm_scores[edit_idx] >= edit_hmm_threshold
                both_pass = iso_pass and hmm_pass
                sentence_edits_pass.append(both_pass)
                edit_idx += 1

        # Sentence passes at edit-level if ALL its edits pass
        all_edits_pass = all(sentence_edits_pass) if sentence_edits_pass else False
        edit_level_passes.append(all_edits_pass)

        # Strategy-level: check if sentence score passes
        if strategy_idx < len(strategy_iso_scores) and strategy_idx < len(strategy_hmm_scores):
            iso_pass = strategy_iso_scores[strategy_idx] >= strategy_iso_threshold
            hmm_pass = strategy_hmm_scores[strategy_idx] >= strategy_hmm_threshold
            strategy_pass = iso_pass and hmm_pass
            strategy_level_passes.append(strategy_pass)
            strategy_idx += 1

    # Convert to arrays
    edit_level_passes = np.array(edit_level_passes)
    strategy_level_passes = np.array(strategy_level_passes)

    print(f"  Matched {len(edit_level_passes)} sentences")
    print()

    # Compute agreement
    print("  CLASSIFICATION RESULTS:")
    print("  " + "-" * 60)

    edit_pass_rate = np.mean(edit_level_passes) * 100
    strategy_pass_rate = np.mean(strategy_level_passes) * 100

    print(f"    Edit-level pass rate:     {edit_pass_rate:.1f}%")
    print(f"    Strategy-level pass rate: {strategy_pass_rate:.1f}%")
    print()

    # Agreement metrics
    agreement = np.mean(edit_level_passes == strategy_level_passes) * 100

    # Confusion matrix
    both_pass = np.sum((edit_level_passes == 1) & (strategy_level_passes == 1))
    both_fail = np.sum((edit_level_passes == 0) & (strategy_level_passes == 0))
    edit_pass_strategy_fail = np.sum((edit_level_passes == 1) & (strategy_level_passes == 0))
    edit_fail_strategy_pass = np.sum((edit_level_passes == 0) & (strategy_level_passes == 1))

    print("  AGREEMENT MATRIX:")
    print("  " + "-" * 60)
    print(f"    Both pass:                {both_pass} ({both_pass/len(edit_level_passes)*100:.1f}%)")
    print(f"    Both fail:                {both_fail} ({both_fail/len(edit_level_passes)*100:.1f}%)")
    print(f"    Edit pass, Strategy fail: {edit_pass_strategy_fail} ({edit_pass_strategy_fail/len(edit_level_passes)*100:.1f}%)")
    print(f"    Edit fail, Strategy pass: {edit_fail_strategy_pass} ({edit_fail_strategy_pass/len(edit_level_passes)*100:.1f}%)")
    print()
    print(f"    Overall agreement:        {agreement:.1f}%")
    print()

    # Conditional analysis
    if np.sum(edit_level_passes) > 0:
        precision = both_pass / np.sum(strategy_level_passes) if np.sum(strategy_level_passes) > 0 else 0
        recall = both_pass / np.sum(edit_level_passes)
        print("  CONDITIONAL PROBABILITIES:")
        print("  " + "-" * 60)
        print(f"    P(Edit pass | Strategy pass):     {precision*100:.1f}%")
        print(f"    P(Strategy pass | Edit pass):     {recall*100:.1f}%")
        print()

print("="*80)
print("SUMMARY")
print("="*80)
print()
print("Key metrics for each source:")
print("  - Overall agreement: % of sentences classified the same at both levels")
print("  - Both pass: Strategy-level captures edit-level 'pass' decisions")
print("  - Both fail: Strategy-level captures edit-level 'fail' decisions")
print()
print("INTERPRETATION:")
print("  High agreement (>80%) → Strategy-level captures edit-level information")
print("  Low 'Edit pass, Strategy fail' → Strategy-level is not overly strict")
print("  Low 'Edit fail, Strategy pass' → Strategy-level is not too lenient")
print()
print("If agreement is high, strategy-level is sufficient for GRPO rewards.")
