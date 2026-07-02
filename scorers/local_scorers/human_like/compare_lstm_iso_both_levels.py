#!/usr/bin/env python3
"""
Compare LSTM and ISO components at both edit and strategy levels.
Generate visualizations showing score distributions and discriminative power.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from typing import Dict, List
import logging

from compare_lstm_hmm import (
    compute_lstm_scores,
    load_test_data, load_lstm_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")


def load_iso_scores(json_path: Path) -> Dict[str, List[float]]:
    """Load ISO scores from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['perplexities']


def plot_both_levels_comparison(
    scores_by_source_iso_edit: Dict[str, np.ndarray],
    scores_by_source_lstm_edit: Dict[str, np.ndarray],
    scores_by_source_iso_strategy: Dict[str, np.ndarray],
    scores_by_source_lstm_strategy: Dict[str, np.ndarray],
    output_path: Path
):
    """
    Create comparison plots for ISO vs LSTM at both edit and strategy levels.

    Creates a 4x2 grid:
    - Row 1: ISO edit level (KDE + Boxplot)
    - Row 2: LSTM edit level (KDE + Boxplot)
    - Row 3: ISO strategy level (KDE + Boxplot)
    - Row 4: LSTM strategy level (KDE + Boxplot)
    """
    sources = ['human', 'coedit', 'llama', 'gemini']
    source_labels = {
        'human': 'Human',
        'coedit': 'CoEdIT',
        'llama': 'Llama',
        'gemini': 'Gemini'
    }
    colors = {
        'human': '#2ecc71',
        'coedit': '#3498db',
        'llama': '#e74c3c',
        'gemini': '#f39c12'
    }

    # Create 4x2 figure
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))

    component_configs = [
        ('ISO only - Edit', scores_by_source_iso_edit, False),
        ('LSTM only - Edit', scores_by_source_lstm_edit, True),
        ('ISO only - Strategy', scores_by_source_iso_strategy, False),
        ('LSTM only - Strategy', scores_by_source_lstm_strategy, True),
    ]

    for row_idx, (component_name, component_scores, is_perplexity) in enumerate(component_configs):
        # Calculate threshold using IQR method on human scores
        human_scores = np.array(component_scores['human'])
        q1 = np.percentile(human_scores, 25)
        q3 = np.percentile(human_scores, 75)
        iqr = q3 - q1

        # For ISO: higher is better, threshold is lower bound (Q1 - 1.5*IQR)
        # For LSTM (perplexity): lower is better, threshold is upper bound (Q3 + 1.5*IQR)
        if is_perplexity:
            threshold = q3 + 1.5 * iqr
        else:
            threshold = q1 - 1.5 * iqr

        # Left column: KDE plot
        ax_kde = axes[row_idx, 0]

        for source in sources:
            scores = component_scores[source]
            sns.kdeplot(data=scores, ax=ax_kde, label=source_labels[source],
                       color=colors[source], linewidth=2.5, alpha=0.8)

        ax_kde.axvline(threshold, color='darkgreen', linestyle='--', linewidth=2,
                      label=f'Threshold ({threshold:.3f})')

        # Use log scale for perplexity (LSTM) to better visualize differences
        if is_perplexity:
            ax_kde.set_xscale('log')
            ax_kde.set_xlabel('Perplexity (log scale)', fontsize=11)
        else:
            ax_kde.set_xlabel('Score', fontsize=11)

        ax_kde.set_ylabel('Density', fontsize=11)
        ax_kde.set_title(component_name, fontsize=12, fontweight='bold')
        ax_kde.legend(fontsize=9, loc='best')
        ax_kde.grid(True, alpha=0.3)

        # Right column: Boxplot
        ax_box = axes[row_idx, 1]

        box_data = [component_scores[source] for source in sources]
        box_colors = [colors[source] for source in sources]

        bp = ax_box.boxplot(box_data, tick_labels=[source_labels[s] for s in sources],
                            patch_artist=True, widths=0.6)

        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax_box.axhline(threshold, color='darkgreen', linestyle='--', linewidth=2,
                      label=f'Threshold ({threshold:.3f})')

        # Use log scale for perplexity (LSTM) to better visualize differences
        if is_perplexity:
            ax_box.set_yscale('log')
            ax_box.set_ylabel('Perplexity (log scale)', fontsize=11)
        else:
            ax_box.set_ylabel('Score', fontsize=11)

        ax_box.set_title(component_name, fontsize=12, fontweight='bold')
        ax_box.legend(fontsize=9, loc='best')
        ax_box.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plot saved to {output_path}")


def compute_metrics(human_scores: np.ndarray, model_scores: np.ndarray) -> dict:
    """Compute separation metrics."""
    from scipy import stats

    human_mean = np.mean(human_scores)
    model_mean = np.mean(model_scores)

    separation_pct = ((human_mean - model_mean) / human_mean) * 100

    # Cohen's d
    pooled_std = np.sqrt((np.var(human_scores) + np.var(model_scores)) / 2)
    cohens_d = (human_mean - model_mean) / pooled_std if pooled_std > 0 else 0

    # Statistical test
    _, p_value = stats.mannwhitneyu(human_scores, model_scores, alternative='two-sided')

    return {
        'separation_pct': separation_pct,
        'cohens_d': cohens_d,
        'p_value': p_value
    }


def main():
    """Main comparison script for both levels."""

    # Paths
    lstm_edit_model_path = Path("scorers/local_scorers/human_like/models/lstm_model.pt")
    lstm_strategy_model_path = Path("scorers/local_scorers/human_like/models/lstm_model_per_sentence.pt")

    iso_edit_path = Path("scorers/local_scorers/human_like/outputs/iso_only_token_features.json")
    iso_strategy_path = Path("scorers/local_scorers/human_like/outputs/per_sentence_iso_only_token_features.json")

    edit_test_data_path = Path("scorers/local_scorers/human_like/data/test_sequences.pkl")
    strategy_test_data_path = Path("scorers/local_scorers/human_like/data/test_sequences_per_sentence.pkl")

    output_dir = Path("scorers/local_scorers/human_like/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if models exist
    if not lstm_edit_model_path.exists():
        logger.error(f"Edit-level LSTM model not found: {lstm_edit_model_path}")
        return

    if not lstm_strategy_model_path.exists():
        logger.error(f"Strategy-level LSTM model not found: {lstm_strategy_model_path}")
        logger.info("Using edit-level model for strategy level as fallback...")
        lstm_strategy_model_path = lstm_edit_model_path

    if not iso_edit_path.exists():
        logger.error(f"ISO edit scores not found: {iso_edit_path}")
        return

    if not iso_strategy_path.exists():
        logger.error(f"ISO strategy scores not found: {iso_strategy_path}")
        return

    # Load models
    logger.info("Loading LSTM models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    lstm_edit_model, max_length_edit = load_lstm_model(lstm_edit_model_path, device=device)
    lstm_strategy_model, max_length_strategy = load_lstm_model(lstm_strategy_model_path, device=device)

    # Load ISO scores
    logger.info("\n" + "="*80)
    logger.info("EDIT LEVEL")
    logger.info("="*80)
    logger.info("Loading ISO edit-level scores...")
    scores_by_source_iso_edit = load_iso_scores(iso_edit_path)

    logger.info("\n" + "="*80)
    logger.info("STRATEGY LEVEL")
    logger.info("="*80)
    logger.info("Loading ISO strategy-level scores...")
    scores_by_source_iso_strategy = load_iso_scores(iso_strategy_path)

    # Load test data for LSTM
    logger.info("\n" + "="*80)
    logger.info("LOADING LSTM TEST DATA")
    logger.info("="*80)
    logger.info("Loading edit-level test data...")
    edit_test_data = load_test_data(edit_test_data_path)

    logger.info("Loading strategy-level test data...")
    strategy_test_data = load_test_data(strategy_test_data_path)

    # Extract sequences by source for both levels
    edit_sequences_by_source = {
        'human': edit_test_data['human_sequences'],
        'coedit': edit_test_data['coedit_sequences'],
        'llama': edit_test_data['llama_sequences'],
        'gemini': edit_test_data['gemini_sequences']
    }

    strategy_sequences_by_source = {
        'human': strategy_test_data['human_sequences'],
        'coedit': strategy_test_data['coedit_sequences'],
        'llama': strategy_test_data['llama_sequences'],
        'gemini': strategy_test_data['gemini_sequences']
    }

    # Compute LSTM scores for EDIT level
    logger.info("\n" + "="*80)
    logger.info("COMPUTING EDIT-LEVEL LSTM SCORES")
    logger.info("="*80)

    scores_by_source_lstm_edit = {}
    for source, sequences in edit_sequences_by_source.items():
        logger.info(f"  {source.capitalize()}: {len(sequences)} sequences")
        scores_by_source_lstm_edit[source] = compute_lstm_scores(
            lstm_edit_model, sequences, max_length_edit, device
        )

    # Compute LSTM scores for STRATEGY level
    logger.info("\n" + "="*80)
    logger.info("COMPUTING STRATEGY-LEVEL LSTM SCORES")
    logger.info("="*80)

    scores_by_source_lstm_strategy = {}
    for source, sequences in strategy_sequences_by_source.items():
        logger.info(f"  {source.capitalize()}: {len(sequences)} sequences")
        scores_by_source_lstm_strategy[source] = compute_lstm_scores(
            lstm_strategy_model, sequences, max_length_strategy, device
        )

    # Generate comparison plots
    logger.info("\n" + "="*80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*80)

    plot_both_levels_comparison(
        scores_by_source_iso_edit,
        scores_by_source_lstm_edit,
        scores_by_source_iso_strategy,
        scores_by_source_lstm_strategy,
        output_dir / "lstm_vs_iso_both_levels.png"
    )

    # Print summary statistics
    logger.info("\n" + "="*80)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*80)

    for level_name, iso_scores, lstm_scores in [
        ("EDIT LEVEL", scores_by_source_iso_edit, scores_by_source_lstm_edit),
        ("STRATEGY LEVEL", scores_by_source_iso_strategy, scores_by_source_lstm_strategy)
    ]:
        logger.info(f"\n{level_name}")
        logger.info("-" * 80)

        # Compute metrics for ISO
        human_iso = np.array(iso_scores['human'])
        model_iso = np.concatenate([np.array(iso_scores[s]) for s in ['coedit', 'llama', 'gemini']])
        iso_metrics = compute_metrics(human_iso, model_iso)

        # Compute metrics for LSTM (note: for perplexity, lower is better, so we invert)
        human_lstm = np.array(lstm_scores['human'])
        model_lstm = np.concatenate([np.array(lstm_scores[s]) for s in ['coedit', 'llama', 'gemini']])
        # For LSTM, we want to measure if human has LOWER perplexity
        lstm_metrics = compute_metrics(model_lstm, human_lstm)  # Swap to get correct direction

        logger.info("\nISO:")
        logger.info(f"  Separation: {iso_metrics['separation_pct']:.2f}%")
        logger.info(f"  Cohen's d: {iso_metrics['cohens_d']:.3f}")
        logger.info(f"  p-value: {iso_metrics['p_value']:.6f}")

        logger.info("\nLSTM:")
        logger.info(f"  Separation: {lstm_metrics['separation_pct']:.2f}%")
        logger.info(f"  Cohen's d: {lstm_metrics['cohens_d']:.3f}")
        logger.info(f"  p-value: {lstm_metrics['p_value']:.6f}")

    logger.info("\n" + "="*80)
    logger.info("DONE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
