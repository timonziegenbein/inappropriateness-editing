#!/usr/bin/env python3
"""
Compare LSTM and HMM components for pattern conformity edit scoring.
Generate visualizations showing score distributions and discriminative power.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
import logging
from scipy import stats

from train_lstm_scorer import load_lstm_model, EditSequenceDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def load_test_data(data_path: Path) -> Dict:
    """Load test sequences for human and model sources."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_lstm_scores(model, sequences: List[List[int]], max_length: int, device: str = 'cuda') -> np.ndarray:
    """
    Compute LSTM perplexity scores for sequences.

    Args:
        model: Trained LSTM model
        sequences: List of edit sequences
        max_length: Maximum sequence length
        device: Device to run on

    Returns:
        perplexities: Array of perplexity scores (lower = more pattern conformity)
    """
    model.eval()
    model = model.to(device)

    # Create dataset
    dataset = EditSequenceDataset(sequences, max_length=max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

    all_perplexities = []

    with torch.no_grad():
        for batch_x, batch_lengths in dataloader:
            batch_x = batch_x.to(device)
            batch_lengths = batch_lengths.to(device)

            perplexities = model.compute_perplexity(batch_x, batch_lengths)
            all_perplexities.extend(perplexities.cpu().numpy())

    # Return raw perplexities (lower = more pattern conformity)
    return np.array(all_perplexities)


def compute_hmm_scores(hmm_model, sequences: List[List[int]]) -> np.ndarray:
    """
    Compute HMM scores for sequences (from existing implementation).

    Args:
        hmm_model: Trained HMM model
        sequences: List of edit sequences (LSTM vocab: 1-5)

    Returns:
        scores: Array of HMM scores
    """
    scores = []
    for seq in sequences:
        if len(seq) == 0:
            scores.append(0.0)
            continue

        # Convert from LSTM vocab (1-5) to HMM vocab (0-4)
        # LSTM: {1: keep, 2: keep-in-edit, 3: del, 4: add, 5: replace}
        # HMM: {0: keep, 1: keep-in-edit, 2: del, 3: add, 4: replace}
        hmm_seq = [max(0, token - 1) for token in seq]

        # Compute log probability
        log_prob = hmm_model.score(np.array(hmm_seq).reshape(-1, 1))

        # Length-normalized score: exp(log_prob / length)
        normalized_score = np.exp(log_prob / len(hmm_seq))
        scores.append(normalized_score)

    return np.array(scores)


def compute_metrics(human_scores: np.ndarray, model_scores: np.ndarray) -> Dict:
    """
    Compute discriminative metrics.

    Args:
        human_scores: Scores for human examples
        model_scores: Scores for model examples

    Returns:
        metrics: Dictionary of metrics
    """
    # Mean difference
    human_mean = np.mean(human_scores)
    model_mean = np.mean(model_scores)
    separation_pct = (human_mean - model_mean) / human_mean * 100

    # Cohen's d (effect size)
    pooled_std = np.sqrt((np.std(human_scores)**2 + np.std(model_scores)**2) / 2)
    cohens_d = (human_mean - model_mean) / pooled_std if pooled_std > 0 else 0

    # Statistical significance (t-test)
    t_stat, p_value = stats.ttest_ind(human_scores, model_scores)

    return {
        'human_mean': human_mean,
        'model_mean': model_mean,
        'separation_pct': separation_pct,
        'cohens_d': cohens_d,
        't_statistic': t_stat,
        'p_value': p_value
    }


def plot_comparison(
    scores_by_source_hmm: Dict[str, np.ndarray],
    scores_by_source_lstm: Dict[str, np.ndarray],
    output_path: Path,
    level: str = "edit"
):
    """
    Create comparison plots for HMM vs LSTM matching the dual components visualization style.

    Args:
        scores_by_source_hmm: Dict mapping source names to HMM scores
        scores_by_source_lstm: Dict mapping source names to LSTM scores
        output_path: Where to save the plot
        level: "edit" or "strategy"
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

    # Create 2x2 figure: rows = (HMM, LSTM), cols = (KDE, Boxplot)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    component_names = ['HMM only', 'LSTM only']
    scores_by_component = [scores_by_source_hmm, scores_by_source_lstm]

    for row_idx, (component_name, component_scores) in enumerate(zip(component_names, scores_by_component)):
        # Calculate threshold using IQR method on human scores
        human_scores = np.array(component_scores['human'])
        q1 = np.percentile(human_scores, 25)
        q3 = np.percentile(human_scores, 75)
        iqr = q3 - q1

        # For HMM: higher is better, threshold is lower bound (Q1 - 1.5*IQR)
        # For LSTM (perplexity): lower is better, threshold is upper bound (Q3 + 1.5*IQR)
        if component_name == 'LSTM only':
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

        ax_kde.set_xlabel('Score', fontsize=11)
        ax_kde.set_ylabel('Density', fontsize=11)
        ax_kde.set_title(f'{component_name} - {level.capitalize()} Level',
                        fontsize=12, fontweight='bold')
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

        ax_box.set_ylabel('Score', fontsize=11)
        ax_box.set_title(f'{component_name} - {level.capitalize()} Level',
                        fontsize=12, fontweight='bold')
        ax_box.legend(fontsize=9, loc='best')
        ax_box.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison plot saved to {output_path}")

    # Compute metrics for both
    human_scores_hmm = scores_by_source_hmm['human']
    model_scores_hmm = np.concatenate([scores_by_source_hmm[s] for s in ['coedit', 'llama', 'gemini']])
    hmm_metrics = compute_metrics(human_scores_hmm, model_scores_hmm)

    human_scores_lstm = scores_by_source_lstm['human']
    model_scores_lstm = np.concatenate([scores_by_source_lstm[s] for s in ['coedit', 'llama', 'gemini']])
    lstm_metrics = compute_metrics(human_scores_lstm, model_scores_lstm)

    return {
        'hmm': hmm_metrics,
        'lstm': lstm_metrics
    }


def plot_correlation(
    scores_hmm: np.ndarray,
    scores_lstm: np.ndarray,
    source_labels: List[str],
    output_path: Path
):
    """
    Plot correlation between HMM and LSTM scores.

    Args:
        scores_hmm: HMM scores
        scores_lstm: LSTM scores
        source_labels: Labels indicating source (Human/Model)
        output_path: Where to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create DataFrame
    df = pd.DataFrame({
        'HMM': scores_hmm,
        'LSTM': scores_lstm,
        'Source': source_labels
    })

    # Scatter plot colored by source
    for source in ['Human', 'Model']:
        mask = df['Source'] == source
        ax.scatter(
            df.loc[mask, 'HMM'],
            df.loc[mask, 'LSTM'],
            alpha=0.3,
            s=10,
            label=source,
            color='green' if source == 'Human' else 'red'
        )

    # Compute correlation
    pearson_r, pearson_p = stats.pearsonr(scores_hmm, scores_lstm)
    spearman_r, spearman_p = stats.spearmanr(scores_hmm, scores_lstm)

    ax.set_xlabel('HMM Score')
    ax.set_ylabel('LSTM Score')
    ax.set_title(f'HMM vs LSTM Correlation\nPearson r={pearson_r:.3f}, Spearman ρ={spearman_r:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Correlation plot saved to {output_path}")

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p
    }


def main():
    """Main comparison script."""

    # Paths
    lstm_model_path = Path("scorers/local_scorers/pattern_conformity/models/lstm_model.pt")
    # Use the hmm_only model (pure HMM without Isolation Forest)
    hmm_model_path = Path("scorers/local_scorers/pattern_conformity/models/hmm_only_hmm_model.pkl")
    test_data_path = Path("scorers/local_scorers/pattern_conformity/data/test_sequences.pkl")
    output_dir = Path("scorers/local_scorers/pattern_conformity/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if models exist
    if not lstm_model_path.exists():
        logger.error(f"LSTM model not found: {lstm_model_path}")
        logger.error("Please train the LSTM model first using train_lstm_scorer.py")
        return

    if not hmm_model_path.exists():
        logger.error(f"HMM model not found: {hmm_model_path}")
        logger.error("Please train the HMM model first using compute_hmm_isolation_scores.py")
        logger.error("Example: python scorers/local_scorers/pattern_conformity/compute_hmm_isolation_scores.py --hmm-only --save-models --model-prefix hmm_only_")
        return

    if not test_data_path.exists():
        logger.error(f"Test data not found: {test_data_path}")
        return

    # Load models
    logger.info("Loading models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lstm_model, max_length = load_lstm_model(lstm_model_path, device=device)

    with open(hmm_model_path, 'rb') as f:
        hmm_model = pickle.load(f)

    # Load test data
    logger.info("Loading test data...")
    test_data = load_test_data(test_data_path)

    # Extract sequences by source
    sequences_by_source = {
        'human': test_data['human_sequences'],
        'coedit': test_data['coedit_sequences'],
        'llama': test_data['llama_sequences'],
        'gemini': test_data['gemini_sequences']
    }

    logger.info(f"Human sequences: {len(sequences_by_source['human'])}")
    logger.info(f"CoEdIT sequences: {len(sequences_by_source['coedit'])}")
    logger.info(f"Llama sequences: {len(sequences_by_source['llama'])}")
    logger.info(f"Gemini sequences: {len(sequences_by_source['gemini'])}")

    # Compute scores for each source
    logger.info("\nComputing LSTM scores...")
    scores_by_source_lstm = {}
    for source, sequences in sequences_by_source.items():
        logger.info(f"  {source.capitalize()}...")
        scores_by_source_lstm[source] = compute_lstm_scores(lstm_model, sequences, max_length, device)

    logger.info("\nComputing HMM scores...")
    scores_by_source_hmm = {}
    for source, sequences in sequences_by_source.items():
        logger.info(f"  {source.capitalize()}...")
        scores_by_source_hmm[source] = compute_hmm_scores(hmm_model, sequences)

    # Generate comparison plots
    logger.info("\nGenerating comparison plots...")
    comparison_metrics = plot_comparison(
        scores_by_source_hmm,
        scores_by_source_lstm,
        output_dir / "lstm_vs_hmm_comparison.png",
        level="edit"
    )

    # Generate correlation plot
    logger.info("Generating correlation plot...")
    all_scores_hmm = np.concatenate([scores_by_source_hmm[s] for s in ['human', 'coedit', 'llama', 'gemini']])
    all_scores_lstm = np.concatenate([scores_by_source_lstm[s] for s in ['human', 'coedit', 'llama', 'gemini']])
    all_labels = (['Human'] * len(scores_by_source_hmm['human']) +
                 ['Model'] * sum(len(scores_by_source_hmm[s]) for s in ['coedit', 'llama', 'gemini']))

    correlation_metrics = plot_correlation(
        all_scores_hmm,
        all_scores_lstm,
        all_labels,
        output_dir / "lstm_hmm_correlation.png"
    )

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)
    logger.info("\nHMM Results:")
    for key, value in comparison_metrics['hmm'].items():
        logger.info(f"  {key}: {value:.4f}")

    logger.info("\nLSTM Results:")
    for key, value in comparison_metrics['lstm'].items():
        logger.info(f"  {key}: {value:.4f}")

    logger.info("\nCorrelation:")
    for key, value in correlation_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    logger.info("\nKey Findings:")
    if comparison_metrics['lstm']['separation_pct'] > comparison_metrics['hmm']['separation_pct']:
        logger.info("✓ LSTM shows better separation than HMM")
    else:
        logger.info("✗ HMM shows better separation than LSTM")

    if abs(comparison_metrics['lstm']['cohens_d']) > abs(comparison_metrics['hmm']['cohens_d']):
        logger.info("✓ LSTM has larger effect size than HMM")
    else:
        logger.info("✗ HMM has larger effect size than LSTM")

    if abs(correlation_metrics['pearson_r']) < 0.3:
        logger.info("✓ HMM and LSTM provide complementary signals (weak correlation)")
    else:
        logger.info("✗ HMM and LSTM may provide redundant signals (strong correlation)")

    logger.info("="*60)


if __name__ == "__main__":
    main()
