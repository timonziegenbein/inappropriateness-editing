"""
Analyze what percentile of human test data is affected by the IQR threshold.
"""
import json
import numpy as np
from pathlib import Path

script_dir = Path(__file__).parent

# Load the ISO+HMM results
results_file = script_dir / "outputs" / "iso_with_hmm_token_features.json"

with open(results_file, 'r') as f:
    data = json.load(f)

human_scores = np.array(data['perplexities']['human'])
threshold = 0.232

# Calculate statistics
below_threshold = human_scores < threshold
n_below = np.sum(below_threshold)
n_total = len(human_scores)
percentage_below = (n_below / n_total) * 100

# Calculate what percentile 0.232 corresponds to
percentile = (np.sum(human_scores < threshold) / len(human_scores)) * 100

print(f"Human Test Data Statistics:")
print(f"=" * 60)
print(f"Total human test examples: {n_total}")
print(f"Threshold value: {threshold}")
print(f"")
print(f"Examples below threshold: {n_below} ({percentage_below:.2f}%)")
print(f"Examples above threshold: {n_total - n_below} ({100 - percentage_below:.2f}%)")
print(f"")
print(f"The threshold of {threshold} corresponds to the {percentile:.2f}th percentile")
print(f"")
print(f"Score distribution:")
print(f"  Min:    {np.min(human_scores):.4f}")
print(f"  Q1:     {np.percentile(human_scores, 25):.4f}")
print(f"  Median: {np.median(human_scores):.4f}")
print(f"  Q3:     {np.percentile(human_scores, 75):.4f}")
print(f"  Max:    {np.max(human_scores):.4f}")
print(f"  Mean:   {np.mean(human_scores):.4f}")
print(f"  Std:    {np.std(human_scores):.4f}")
print(f"")

# Also check model scores for comparison
print(f"Model Scores (percentage below threshold):")
print(f"=" * 60)
for source in ['coedit', 'llama', 'gemini']:
    model_scores = np.array(data['perplexities'][source])
    pct_below = (np.sum(model_scores < threshold) / len(model_scores)) * 100
    print(f"  {source.capitalize()}: {pct_below:.2f}%")
