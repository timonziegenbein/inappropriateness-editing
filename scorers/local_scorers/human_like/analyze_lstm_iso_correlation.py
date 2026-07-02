"""
Analyze correlation and agreement between LSTM and ISO at both edit and strategy levels.
"""

import torch
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import pickle
import json

from compare_lstm_hmm import (
    compute_lstm_scores,
    load_test_data, load_lstm_model
)

script_dir = Path(__file__).parent

# Load LSTM models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

lstm_edit_model, max_length_edit = load_lstm_model(
    script_dir / "models" / "lstm_model.pt", device=device
)
lstm_strategy_model, max_length_strategy = load_lstm_model(
    script_dir / "models" / "lstm_model_per_sentence.pt", device=device
)

# Load test data
edit_test_data = load_test_data(script_dir / "data" / "test_sequences.pkl")
strategy_test_data = load_test_data(script_dir / "data" / "test_sequences_per_sentence.pkl")

# Load ISO scores
with open(script_dir / "outputs" / "iso_only_token_features.json", 'r') as f:
    iso_edit_data = json.load(f)['perplexities']

with open(script_dir / "outputs" / "per_sentence_iso_only_token_features.json", 'r') as f:
    iso_strategy_data = json.load(f)['perplexities']

sources = ['human', 'coedit', 'llama', 'gemini']

# Compute LSTM scores for EDIT level
print("Computing edit-level LSTM scores...")
edit_scores = {'ISO': iso_edit_data, 'LSTM': {}}
for source in sources:
    sequences = edit_test_data[f'{source}_sequences']
    edit_scores['LSTM'][source] = compute_lstm_scores(lstm_edit_model, sequences, max_length_edit, device)

# Compute LSTM scores for STRATEGY level
print("Computing strategy-level LSTM scores...")
strategy_scores = {'ISO': iso_strategy_data, 'LSTM': {}}
for source in sources:
    sequences = strategy_test_data[f'{source}_sequences']
    strategy_scores['LSTM'][source] = compute_lstm_scores(lstm_strategy_model, sequences, max_length_strategy, device)

# Open output file
output_file = script_dir / "outputs" / "lstm_iso_correlation_analysis.txt"
output_file.parent.mkdir(exist_ok=True)

with open(output_file, 'w') as f:
    # Redirect both stdout and file
    def log(msg):
        print(msg)
        f.write(msg + '\n')

    log("="*80)
    log("CORRELATION ANALYSIS: LSTM vs ISO")
    log("="*80)
    log("")

    # Analyze both levels
    for level_name, level_scores in [("EDIT LEVEL", edit_scores), ("STRATEGY LEVEL", strategy_scores)]:
        log("="*80)
        log(level_name)
        log("="*80)
        log("")

        # Pairwise correlation (combined across all sources)
        log("OVERALL CORRELATION (Combined across all sources)")
        log("-" * 80)
        log("")

        # Combine all sources
        iso_combined = []
        lstm_combined = []
        for source in sources:
            iso_combined.extend(level_scores['ISO'][source])
            lstm_combined.extend(level_scores['LSTM'][source])

        iso_combined = np.array(iso_combined)
        lstm_combined = np.array(lstm_combined)

        pearson_r, _ = pearsonr(iso_combined, lstm_combined)
        spearman_r, _ = spearmanr(iso_combined, lstm_combined)

        log(f"Pearson correlation:  {pearson_r:.4f}")
        log(f"Spearman correlation: {spearman_r:.4f}")
        log("")

        # Per-source correlations
        log("PER-SOURCE CORRELATIONS")
        log("-" * 80)
        log("")

        log(f"{'Source':<12} {'Pearson r':<15} {'Spearman ρ':<15} {'Interpretation'}")
        log("-" * 70)

        for source in sources:
            iso = np.array(level_scores['ISO'][source])
            lstm = np.array(level_scores['LSTM'][source])

            # Ensure same length
            min_len = min(len(iso), len(lstm))
            iso = iso[:min_len]
            lstm = lstm[:min_len]

            pearson_r, _ = pearsonr(iso, lstm)
            spearman_r, _ = spearmanr(iso, lstm)

            if abs(pearson_r) > 0.7:
                interpretation = "Strong correlation"
            elif abs(pearson_r) > 0.4:
                interpretation = "Moderate correlation"
            elif abs(pearson_r) > 0.2:
                interpretation = "Weak correlation"
            else:
                interpretation = "Very weak/no correlation"

            log(f"{source.capitalize():<12} {pearson_r:<15.4f} {spearman_r:<15.4f} {interpretation}")

        log("")

        # Ranking agreement analysis
        log("="*80)
        log("RANKING AGREEMENT ANALYSIS")
        log("="*80)
        log("")
        log("Do ISO and LSTM agree on which examples score highest?")
        log("(Using top 20% as 'high scoring')")
        log("")

        for source in sources:
            iso = np.array(level_scores['ISO'][source])
            lstm = np.array(level_scores['LSTM'][source])

            # Get top 20% indices for each
            # Note: for ISO higher is better, for LSTM lower is better (perplexity)
            iso_threshold = np.percentile(iso, 80)
            lstm_threshold = np.percentile(lstm, 20)  # Bottom 20% = best for perplexity

            iso_top = set(np.where(iso >= iso_threshold)[0])
            lstm_top = set(np.where(lstm <= lstm_threshold)[0])

            # Calculate overlap
            overlap = len(iso_top & lstm_top)
            iso_only = len(iso_top - lstm_top)
            lstm_only = len(lstm_top - iso_top)

            jaccard = overlap / len(iso_top | lstm_top) if len(iso_top | lstm_top) > 0 else 0

            log(f"{source.capitalize()}:")
            log(f"  Both high: {overlap}/{len(iso_top)} ({overlap/len(iso_top)*100:.1f}%)")
            log(f"  ISO high only: {iso_only}")
            log(f"  LSTM high only: {lstm_only}")
            log(f"  Jaccard similarity: {jaccard:.3f}")
            log("")

        # Disagreement analysis
        log("="*80)
        log("DISAGREEMENT ANALYSIS")
        log("="*80)
        log("")
        log("Examples where models strongly disagree:")
        log("(ISO high but LSTM low, or vice versa)")
        log("")

        for source in sources:
            iso = np.array(level_scores['ISO'][source])
            lstm = np.array(level_scores['LSTM'][source])

            # Normalize to [0,1] for comparison
            # For ISO: higher is better, keep as is
            iso_norm = (iso - iso.min()) / (iso.max() - iso.min() + 1e-10)
            # For LSTM: lower is better, so invert
            lstm_norm = 1 - (lstm - lstm.min()) / (lstm.max() - lstm.min() + 1e-10)

            # Find disagreements (large difference in normalized scores)
            diff = np.abs(iso_norm - lstm_norm)
            strong_disagreements = np.where(diff > 0.5)[0]

            # Categorize disagreements
            iso_high_lstm_low = np.where((iso_norm > 0.7) & (lstm_norm < 0.3))[0]
            iso_low_lstm_high = np.where((iso_norm < 0.3) & (lstm_norm > 0.7))[0]

            log(f"{source.capitalize()}:")
            log(f"  Strong disagreements: {len(strong_disagreements)} ({len(strong_disagreements)/len(iso)*100:.1f}%)")
            log(f"  ISO high + LSTM low: {len(iso_high_lstm_low)}")
            log(f"  ISO low + LSTM high: {len(iso_low_lstm_high)}")
            log("")

        log("")

    # Overall conclusion
    log("="*80)
    log("CONCLUSION")
    log("="*80)
    log("")

    # Get correlations for both levels
    iso_edit_combined = []
    lstm_edit_combined = []
    for source in sources:
        iso_edit_combined.extend(edit_scores['ISO'][source])
        lstm_edit_combined.extend(edit_scores['LSTM'][source])

    iso_strategy_combined = []
    lstm_strategy_combined = []
    for source in sources:
        iso_strategy_combined.extend(strategy_scores['ISO'][source])
        lstm_strategy_combined.extend(strategy_scores['LSTM'][source])

    pearson_edit, _ = pearsonr(iso_edit_combined, lstm_edit_combined)
    pearson_strategy, _ = pearsonr(iso_strategy_combined, lstm_strategy_combined)

    log(f"Edit Level Correlation (Pearson): {pearson_edit:.4f}")
    log(f"Strategy Level Correlation (Pearson): {pearson_strategy:.4f}")
    log("")

    # Interpretation for edit level
    log("EDIT LEVEL:")
    if abs(pearson_edit) > 0.7:
        log("  ⚠️  Strong correlation - models capture similar information")
        log("  Consider using just one model at edit level")
    elif abs(pearson_edit) > 0.4:
        log("  ✓ Moderate correlation - some overlap but also unique signals")
        log("  Combining provides complementary information")
    else:
        log("  ✓✓ Weak correlation - models capture different information")
        log("  Combining provides strong complementary signals")
    log("")

    # Interpretation for strategy level
    log("STRATEGY LEVEL:")
    if abs(pearson_strategy) > 0.7:
        log("  ⚠️  Strong correlation - models capture similar information")
        log("  Consider using just one model at strategy level")
    elif abs(pearson_strategy) > 0.4:
        log("  ✓ Moderate correlation - some overlap but also unique signals")
        log("  Combining provides complementary information")
    else:
        log("  ✓✓ Weak correlation - models capture different information")
        log("  Combining provides strong complementary signals")
    log("")

    log("="*80)
    log("KEY INSIGHTS")
    log("="*80)
    log("")
    log("ISO: Captures distributional anomalies in edit operation features")
    log("LSTM: Captures sequence-level predictability (perplexity)")
    log("")
    log("Both provide different perspectives on 'human-likeness':")
    log("- ISO: Are the edit features statistically similar to human patterns?")
    log("- LSTM: Is the sequence predictable based on learned patterns?")
    log("")

print(f"\n✓ Analysis complete. Results saved to {output_file.name}")
