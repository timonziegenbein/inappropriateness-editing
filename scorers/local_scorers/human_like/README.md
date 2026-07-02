# Human-Like Scorer

This directory contains the ISO+HMM (Isolation Forest + Hidden Markov Model) human-like scorer for evaluating edit human-likeness in the appropriateness editing task.

## Overview

The human-like scorer uses anomaly detection to identify whether an edit follows human editing patterns. It combines:

1. **Isolation Forest**: Detects anomalies in token-level edit features (5 dimensions)
2. **Hidden Markov Model**: Captures sequential patterns in edit operations (1 dimension)

The combined 6-dimensional feature space provides complementary information:
- ISO features capture edit composition (counts of keep/del/add/replace operations)
- HMM score captures sequential transition patterns

## Quick Start

### Using the Scorer

```python
from scorers.local_scorers.human_like.human_like_scorer import HumanLikeScorer

scorer = HumanLikeScorer(device="cuda")
score = scorer.calculate_human_likeness(
    original_argument="The full original text...",
    original_sentence="The sentence being edited...",
    inappropriate_part="text to replace",
    rewritten_part="replacement text"
)
# Returns 1.0 if human-like (score >= 0.232), else 0.0
```

### Training New Models

```bash
# Extract edits from train/test splits
python extract_human_edits_train.py
python extract_human_edits_test.py

# Train models (automatically saves to current directory)
python compute_hmm_isolation_scores.py \
    --train-file data/human_with_edits_train.jsonl \
    --test-file data/human_with_edits_test.jsonl \
    --save-models \
    --model-prefix iso_hmm_ \
    --output-file outputs/iso_with_hmm_token_features.json

# Visualize results
python visualize_all_approaches.py

# Analyze correlations
python analyze_correlation.py
```

## Architecture

### Model Components

**Trained Models** (in `models/` directory):
- `iso_hmm_hmm_model.pkl`: Categorical HMM with 4 hidden states
- `iso_hmm_iso_model.pkl`: Isolation Forest trained on human edits only
- `iso_hmm_operation_vocab.json`: Vocabulary mapping operations to integers

**Edit Operations**:
- `keep`: Tokens outside edit region
- `keep-in-edit`: Unchanged tokens within edit region
- `del`: Deleted tokens
- `add`: Added tokens
- `replace`: Replaced tokens

### Scoring Process

1. **Sequence Generation**: Uses sequence alignment (difflib) to classify each token
2. **Feature Extraction**:
   - Token counts: [n_keep, n_keep_in_edit, n_del, n_add, n_replace]
   - HMM score: exp(log_likelihood / length)
3. **Anomaly Detection**: Isolation Forest scores combined features
4. **Binary Classification**: Threshold = 0.232 (IQR-based from human training data)

## Directory Structure

```
human_like/
├── README.md                          # This file
├── human_like_scorer.py               # Main scorer implementation
├── data/                              # Input data files
│   ├── human_with_edits_train.jsonl   # Training data (from IteraTeR train split)
│   ├── human_with_edits_test.jsonl    # Test data (from IteraTeR test split)
│   ├── coedit_predictions.jsonl       # CoEdIT model predictions
│   ├── llama_predictions.jsonl        # Llama model predictions
│   └── gemini_predictions.jsonl       # Gemini model predictions
├── models/                            # Trained models
│   ├── iso_hmm_hmm_model.pkl          # HMM model (used by scorer)
│   ├── iso_hmm_iso_model.pkl          # Isolation Forest model (used by scorer)
│   └── iso_hmm_operation_vocab.json   # Operation vocabulary
├── outputs/                           # Results and visualizations
│   ├── *.json                         # Scoring results
│   ├── *.png                          # Visualization plots
│   └── correlation_analysis.txt       # Correlation statistics
├── compute_hmm_isolation_scores.py    # Main training script
├── visualize_all_approaches.py        # Generate comparison plots
├── analyze_correlation.py             # Compute correlations between approaches
├── compare_iso_predictions.py         # Compare ISO variants with metrics
├── extract_human_edits_train.py       # Extract training data from HuggingFace
├── extract_human_edits_test.py        # Extract test data from HuggingFace
├── precompute_edits.py                # Utilities for edit extraction
├── generate_coedit_predictions.py     # Generate CoEdIT predictions
├── generate_llama_predictions.py      # Generate Llama predictions
├── generate_llm_predictions.py        # Generate Gemini predictions
├── create_iterater_dataset.py         # Create HuggingFace dataset splits
└── acl_paper_section.tex              # LaTeX documentation
```

## Experimental Results

### Approach Comparison

Training on human edits only, evaluation on held-out test set:

| Approach | Human | CoEdIT | Llama | Gemini-2.5 | Separation* |
|----------|-------|--------|-------|------------|-------------|
| HMM only | 1.4381 | 1.4040 | 1.3886 | 1.3918 | 0.0495 |
| HMM as ISO feat. | 0.2822 | 0.2452 | 0.2311 | 0.2430 | 0.0511 |
| ISO only | 0.2695 | 0.2348 | 0.2148 | 0.2214 | 0.0481 |
| **ISO+HMM** | **0.2851** | **0.2445** | **0.2242** | **0.2312** | **0.0539** |

*Separation = Human score - Min(model scores). Higher is better.

### Key Findings

1. **Complementarity**: Pearson r=0.61 between ISO and HMM shows moderate correlation
   - Human edits: r=0.35 (weak - largely independent signals)
   - Model edits: r=0.48-0.66 (moderate correlation)
2. **ISO+HMM achieves best separation**: 0.0539 vs 0.0481 (ISO-only)
3. **All approaches correctly rank Human > Models**
4. **IQR-based threshold**: Q1 - 1.5×IQR = 0.232 from human training scores

### Visualizations

Generated by `visualize_all_approaches.py`:
- `outputs/all_approaches_comparison.png`: Bar chart of mean scores by source
- `outputs/all_approaches_table.png`: Detailed statistics table
- `outputs/comparison_distributions.png`: 4×2 layout with histograms and boxplots
- `outputs/all_approaches_distributions.png`: Side-by-side histograms
- `outputs/comparison_cdfs.png`: Cumulative distribution functions

## Dataset Information

### Training/Test Split

Data comes from the IteraTeR dataset with intent prefixes:
- **Train**: ~80% of IteraTeR (timonziegenbein/iterater-with-prefixes, split="train")
- **Test**: ~20% of IteraTeR (timonziegenbein/iterater-with-prefixes, split="test")

### Model Predictions

For comparison, we include predictions from:
- **CoEdIT**: Supervised editing model
- **Llama-3.1-8B**: General-purpose LLM
- **Gemini-2.5-Flash**: Google's LLM

These are generated using:
- `generate_coedit_predictions.py`
- `generate_llama_predictions.py`
- `generate_llm_predictions.py`

## Training Details

### HMM Configuration
- Type: CategoricalHMM
- Hidden states: 4
- Observations: 5 (keep, keep-in-edit, del, add, replace)
- Training: EM algorithm with 100 iterations

### Isolation Forest Configuration
- Contamination: 'auto' (affects predict() only, not score_samples())
- Training: Human edits only (anomaly detection approach)
- Features: 6D [token counts (5) + HMM score (1)]

### Normalization
- ISO scores: Sigmoid normalization `1 / (1 + exp(-score))`
- HMM scores: Exponential of normalized log-likelihood
- Threshold: 0.232 (IQR-based from human training distribution)

## Command Reference

### Full Training Pipeline

```bash
# 1. Create HuggingFace dataset (one-time setup)
python create_iterater_dataset.py

# 2. Extract train/test data locally
python extract_human_edits_train.py
python extract_human_edits_test.py

# 3. Train all approaches
python compute_hmm_isolation_scores.py --save-models --model-prefix iso_hmm_ \
    --output-file outputs/iso_with_hmm_token_features.json

python compute_hmm_isolation_scores.py --iso-only --save-models --model-prefix iso_only_ \
    --output-file outputs/iso_only_token_features.json

python compute_hmm_isolation_scores.py --hmm-only --save-models --model-prefix hmm_only_ \
    --output-file outputs/hmm_only_token_features.json

python compute_hmm_isolation_scores.py --hmm-as-feature-only --save-models \
    --model-prefix hmm_as_iso_feature_ \
    --output-file outputs/hmm_as_iso_feature_only.json

# 4. Generate visualizations and analysis
python visualize_all_approaches.py
python analyze_correlation.py
python compare_iso_predictions.py
```

### Testing Scorer

```bash
# Test on examples
python -c "
from scorers.local_scorers.human_like.human_like_scorer import HumanLikeScorer
import torch

scorer = HumanLikeScorer(device='cuda' if torch.cuda.is_available() else 'cpu')
score = scorer.calculate_human_likeness(
    original_argument='This is a stupid idea that nobody likes.',
    original_sentence='This is a stupid idea that nobody likes.',
    inappropriate_part='stupid idea that nobody likes',
    rewritten_part='questionable proposal'
)
print(f'Score: {score}')
"
```

## Implementation Notes

### Why Isolation Forest?

Isolation Forest is well-suited for this task because:
1. **One-class learning**: Trains on human edits only, treats models as anomalies
2. **No distributional assumptions**: Works with arbitrary feature distributions
3. **Efficient**: O(n log n) training and O(log n) inference
4. **Interpretable**: Anomaly score measures "isolation depth"

### Why HMM?

HMM captures sequential dependencies:
1. **Transition patterns**: Learns typical edit operation sequences
2. **State modeling**: Discovers latent editing "modes"
3. **Probabilistic**: Provides log-likelihood scores
4. **Complementary**: Different from count-based ISO features (r=0.61)

### Threshold Selection

The threshold of 0.232 was chosen using the IQR method on human training scores:
- Q1 (25th percentile) = 0.2682
- Q3 (75th percentile) = 0.2969
- IQR = Q3 - Q1 = 0.0287
- Threshold = Q1 - 1.5×IQR = 0.2682 - 0.0431 = 0.232

This is a standard statistical method for outlier detection.

## Citation

If you use this scorer, please cite:

```bibtex
@inproceedings{ziegenbein2025humanlike,
  title={Human-Like Edit Detection via Isolation Forest and Hidden Markov Models},
  author={Ziegenbein, Timon},
  booktitle={Proceedings of ACL},
  year={2025}
}
```

## License

This code is part of the appropriateness-edit project.
