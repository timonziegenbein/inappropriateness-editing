"""
Analyze correlation and agreement between HMM and LSTM at both edit and strategy levels.
"""

import torch
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import pickle

from compare_lstm_hmm import (
    compute_lstm_scores, compute_hmm_scores,
    load_test_data, load_lstm_model
)

script_dir = Path(__file__).parent

# Load models and compute scores
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models
lstm_edit_model, max_length_edit = load_lstm_model(
    script_dir / "models" / "lstm_model.pt", device=device
)
lstm_strategy_model, max_length_strategy = load_lstm_model(
    script_dir / "models" / "lstm_model_per_sentence.pt", device=device
)

with open(script_dir / "models" / "hmm_only_hmm_model.pkl", 'rb') as f:
    hmm_model = pickle.load(f)

# Load test data
edit_test_data = load_test_data(script_dir / "data" / "test_sequences.pkl")
strategy_test_data = load_test_data(script_dir / "data" / "test_sequences_per_sentence.pkl")

sources = ['human', 'coedit', 'llama', 'gemini']

# Compute scores for EDIT level
print("Computing edit-level scores...")
edit_scores = {'HMM': {}, 'LSTM': {}}
for source in sources:
    sequences = edit_test_data[f'{source}_sequences']
    edit_scores['HMM'][source] = compute_hmm_scores(hmm_model, sequences)
    edit_scores['LSTM'][source] = compute_lstm_scores(lstm_edit_model, sequences, max_length_edit, device)

# Compute scores for STRATEGY level
print("Computing strategy-level scores...")
strategy_scores = {'HMM': {}, 'LSTM': {}}
for source in sources:
    sequences = strategy_test_data[f'{source}_sequences']
    strategy_scores['HMM'][source] = compute_hmm_scores(hmm_model, sequences)
    strategy_scores['LSTM'][source] = compute_lstm_scores(lstm_strategy_model, sequences, max_length_strategy, device)

# Open output file
output_file = script_dir / "outputs" / "hmm_lstm_correlation_analysis.txt"
output_file.parent.mkdir(exist_ok=True)

with open(output_file, 'w') as f:
    # Redirect both stdout and file
    def log(msg):
        print(msg)
        f.write(msg + '\n')

    log("="*80)
    log("CORRELATION ANALYSIS: HMM vs LSTM")
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
        hmm_combined = []
        lstm_combined = []
        for source in sources:
            hmm_combined.extend(level_scores['HMM'][source])
            lstm_combined.extend(level_scores['LSTM'][source])

        hmm_combined = np.array(hmm_combined)
        lstm_combined = np.array(lstm_combined)

        pearson_r, _ = pearsonr(hmm_combined, lstm_combined)
        spearman_r, _ = spearmanr(hmm_combined, lstm_combined)

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
            hmm = level_scores['HMM'][source]
            lstm = level_scores['LSTM'][source]

            # Ensure same length
            min_len = min(len(hmm), len(lstm))
            hmm = hmm[:min_len]
            lstm = lstm[:min_len]

            pearson_r, _ = pearsonr(hmm, lstm)
            spearman_r, _ = spearmanr(hmm, lstm)

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
        log("Do HMM and LSTM agree on which examples score highest?")
        log("(Using top 20% as 'high scoring')")
        log("")

        for source in sources:
            hmm = np.array(level_scores['HMM'][source])
            lstm = np.array(level_scores['LSTM'][source])

            # Get top 20% indices for each
            # Note: for HMM higher is better, for LSTM lower is better (perplexity)
            hmm_threshold = np.percentile(hmm, 80)
            lstm_threshold = np.percentile(lstm, 20)  # Bottom 20% = best for perplexity

            hmm_top = set(np.where(hmm >= hmm_threshold)[0])
            lstm_top = set(np.where(lstm <= lstm_threshold)[0])

            # Calculate overlap
            overlap = len(hmm_top & lstm_top)
            hmm_only = len(hmm_top - lstm_top)
            lstm_only = len(lstm_top - hmm_top)

            jaccard = overlap / len(hmm_top | lstm_top) if len(hmm_top | lstm_top) > 0 else 0

            log(f"{source.capitalize()}:")
            log(f"  Both high: {overlap}/{len(hmm_top)} ({overlap/len(hmm_top)*100:.1f}%)")
            log(f"  HMM high only: {hmm_only}")
            log(f"  LSTM high only: {lstm_only}")
            log(f"  Jaccard similarity: {jaccard:.3f}")
            log("")

        # Disagreement analysis
        log("="*80)
        log("DISAGREEMENT ANALYSIS")
        log("="*80)
        log("")
        log("Examples where models strongly disagree:")
        log("(HMM high but LSTM low, or vice versa)")
        log("")

        for source in sources:
            hmm = np.array(level_scores['HMM'][source])
            lstm = np.array(level_scores['LSTM'][source])

            # Normalize to [0,1] for comparison
            # For HMM: higher is better, keep as is
            hmm_norm = (hmm - hmm.min()) / (hmm.max() - hmm.min() + 1e-10)
            # For LSTM: lower is better, so invert
            lstm_norm = 1 - (lstm - lstm.min()) / (lstm.max() - lstm.min() + 1e-10)

            # Find disagreements (large difference in normalized scores)
            diff = np.abs(hmm_norm - lstm_norm)
            strong_disagreements = np.where(diff > 0.5)[0]

            # Categorize disagreements
            hmm_high_lstm_low = np.where((hmm_norm > 0.7) & (lstm_norm < 0.3))[0]
            hmm_low_lstm_high = np.where((hmm_norm < 0.3) & (lstm_norm > 0.7))[0]

            log(f"{source.capitalize()}:")
            log(f"  Strong disagreements: {len(strong_disagreements)} ({len(strong_disagreements)/len(hmm)*100:.1f}%)")
            log(f"  HMM high + LSTM low: {len(hmm_high_lstm_low)}")
            log(f"  HMM low + LSTM high: {len(hmm_low_lstm_high)}")
            log("")

        log("")

    # Overall conclusion
    log("="*80)
    log("CONCLUSION")
    log("="*80)
    log("")

    # Get correlations for both levels
    hmm_edit_combined = []
    lstm_edit_combined = []
    for source in sources:
        hmm_edit_combined.extend(edit_scores['HMM'][source])
        lstm_edit_combined.extend(edit_scores['LSTM'][source])

    hmm_strategy_combined = []
    lstm_strategy_combined = []
    for source in sources:
        hmm_strategy_combined.extend(strategy_scores['HMM'][source])
        lstm_strategy_combined.extend(strategy_scores['LSTM'][source])

    pearson_edit, _ = pearsonr(hmm_edit_combined, lstm_edit_combined)
    pearson_strategy, _ = pearsonr(hmm_strategy_combined, lstm_strategy_combined)

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
    log("HMM: Captures transition patterns between edit operations")
    log("LSTM: Captures sequence-level predictability (perplexity)")
    log("")
    log("Both provide different perspectives on 'human-likeness':")
    log("- HMM: Are the edit patterns similar to human editing patterns?")
    log("- LSTM: Is the sequence predictable based on learned patterns?")
    log("")

print(f"\n✓ Analysis complete. Results saved to {output_file.name}")
