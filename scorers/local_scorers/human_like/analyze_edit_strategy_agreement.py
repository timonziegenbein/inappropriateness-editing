"""
Analyze agreement between edit-level and strategy-level classifications.

This checks whether strategy-level scores capture most of the information
from edit-level scores, justifying using only strategy-level for GRPO rewards.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

script_dir = Path(__file__).parent

print("="*80)
print("EDIT-LEVEL vs STRATEGY-LEVEL AGREEMENT ANALYSIS")
print("="*80)
print()

# Load both edit-level and strategy-level results for both components
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

sources = ['human', 'coedit', 'llama', 'gemini']

print("THRESHOLD COMPUTATION")
print("-" * 80)
print()

# Compute IQR-based thresholds from human scores
def compute_iqr_threshold(scores):
    """Compute Q1 - 1.5*IQR threshold."""
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr

edit_iso_threshold = compute_iqr_threshold(edit_iso['human'])
edit_hmm_threshold = compute_iqr_threshold(edit_hmm['human'])
strategy_iso_threshold = compute_iqr_threshold(strategy_iso['human'])
strategy_hmm_threshold = compute_iqr_threshold(strategy_hmm['human'])

print(f"Edit-level thresholds:")
print(f"  ISO: {edit_iso_threshold:.4f}")
print(f"  HMM: {edit_hmm_threshold:.4f}")
print()
print(f"Strategy-level thresholds:")
print(f"  ISO: {strategy_iso_threshold:.4f}")
print(f"  HMM: {strategy_hmm_threshold:.4f}")
print()

print("="*80)
print("CLASSIFICATION AGREEMENT ANALYSIS")
print("="*80)
print()

# For each source, classify samples as pass/fail at both levels
# Then compute agreement

def classify_samples(scores, threshold):
    """Classify samples as 1 (pass) or 0 (fail) based on threshold."""
    return np.array(scores) >= threshold

for source in sources:
    print(f"{source.upper()}:")
    print("-" * 80)

    # Edit-level classifications
    edit_iso_pass = classify_samples(edit_iso[source], edit_iso_threshold)
    edit_hmm_pass = classify_samples(edit_hmm[source], edit_hmm_threshold)
    edit_both_pass = edit_iso_pass & edit_hmm_pass  # Conjunction

    # Strategy-level classifications
    strategy_iso_pass = classify_samples(strategy_iso[source], strategy_iso_threshold)
    strategy_hmm_pass = classify_samples(strategy_hmm[source], strategy_hmm_threshold)
    strategy_both_pass = strategy_iso_pass & strategy_hmm_pass  # Conjunction

    print(f"  Sample counts:")
    print(f"    Edit-level samples:     {len(edit_iso_pass)}")
    print(f"    Strategy-level samples: {len(strategy_iso_pass)}")
    print()

    print(f"  Edit-level pass rates:")
    print(f"    ISO pass:  {np.mean(edit_iso_pass)*100:.1f}%")
    print(f"    HMM pass:  {np.mean(edit_hmm_pass)*100:.1f}%")
    print(f"    Both pass: {np.mean(edit_both_pass)*100:.1f}%")
    print()

    print(f"  Strategy-level pass rates:")
    print(f"    ISO pass:  {np.mean(strategy_iso_pass)*100:.1f}%")
    print(f"    HMM pass:  {np.mean(strategy_hmm_pass)*100:.1f}%")
    print(f"    Both pass: {np.mean(strategy_both_pass)*100:.1f}%")
    print()

    # Key question: If strategy-level says "pass", what % of edits in those sentences pass at edit-level?
    # And vice versa: If strategy-level says "fail", what % of edits fail?

    # We can't directly match edit-to-sentence without the original data structure,
    # but we can compare overall discrimination

    print(f"  Discrimination comparison:")
    edit_discrimination = np.mean(edit_both_pass)
    strategy_discrimination = np.mean(strategy_both_pass)
    print(f"    Edit-level (both pass):     {edit_discrimination*100:.1f}%")
    print(f"    Strategy-level (both pass): {strategy_discrimination*100:.1f}%")

    if abs(edit_discrimination - strategy_discrimination) < 0.05:
        print(f"    → Similar discrimination (diff < 5pp)")
    elif strategy_discrimination > edit_discrimination:
        print(f"    → Strategy-level MORE selective")
    else:
        print(f"    → Strategy-level LESS selective")

    print()

print("="*80)
print("CORRELATION BETWEEN EDIT AND STRATEGY SCORES")
print("="*80)
print()

from scipy.stats import pearsonr

print(f"{'Source':<12} {'ISO r':<12} {'HMM r':<12} {'Both r':<12}")
print("-" * 60)

for source in sources:
    # We can't directly correlate because different sample counts
    # But we can look at the distributions

    # Instead, let's compute how well strategy-level discriminates the same way edit-level does
    # by comparing pass rates

    edit_iso_pass_rate = np.mean(classify_samples(edit_iso[source], edit_iso_threshold))
    strategy_iso_pass_rate = np.mean(classify_samples(strategy_iso[source], strategy_iso_threshold))

    edit_hmm_pass_rate = np.mean(classify_samples(edit_hmm[source], edit_hmm_threshold))
    strategy_hmm_pass_rate = np.mean(classify_samples(strategy_hmm[source], strategy_hmm_threshold))

    edit_both_pass_rate = np.mean(classify_samples(edit_iso[source], edit_iso_threshold) &
                                   classify_samples(edit_hmm[source], edit_hmm_threshold))
    strategy_both_pass_rate = np.mean(classify_samples(strategy_iso[source], strategy_iso_threshold) &
                                       classify_samples(strategy_hmm[source], strategy_hmm_threshold))

    # Compare pass rates as a proxy for discrimination
    iso_diff = abs(edit_iso_pass_rate - strategy_iso_pass_rate)
    hmm_diff = abs(edit_hmm_pass_rate - strategy_hmm_pass_rate)
    both_diff = abs(edit_both_pass_rate - strategy_both_pass_rate)

    print(f"{source.capitalize():<12} {iso_diff:<12.3f} {hmm_diff:<12.3f} {both_diff:<12.3f}")

print()
print("(Lower values = more agreement in pass rates)")
print()

print("="*80)
print("CONCLUSION")
print("="*80)
print()

print("Key Questions:")
print()
print("1. Does strategy-level capture edit-level information?")
print("   → Check if pass rates are similar across levels")
print()
print("2. Is strategy-level sufficient for GRPO rewards?")
print("   → If discrimination patterns match, strategy-level captures the signal")
print()
print("3. What's the benefit of strategy-level over edit-level?")
print("   → Strategy-level has MUCH stronger effect sizes (d=1.02 vs d=0.25)")
print("   → Especially for HMM: reversed at edit-level, correct at strategy-level")
print()

print("RECOMMENDATION:")
print("If pass rates are reasonably similar (< 10pp difference) across most sources,")
print("then strategy-level captures sufficient information and should be used alone")
print("for GRPO rewards due to its superior discrimination power.")
