"""
Compare per-edit vs per-sentence analyses side-by-side.

This shows how the two research questions differ:
- Per-edit: Can we distinguish individual edits? (matches inference)
- Per-sentence: Can we distinguish editing strategies? (multiple edits combined)
"""
import json
import numpy as np
from pathlib import Path


def load_results(prefix):
    """Load all four approach results for a given prefix."""
    script_dir = Path(__file__).parent

    if prefix:
        # Per-sentence results
        approaches = {
            'ISO+HMM': f'{prefix}_iso_with_hmm_token_features.json',
            'ISO-only': f'{prefix}_iso_only_token_features.json',
            'HMM-as-ISO-feat': f'{prefix}_hmm_as_iso_feature_only.json',
            'HMM-only': f'{prefix}_hmm_only_token_features.json',
        }
    else:
        # Per-edit results (no prefix)
        approaches = {
            'ISO+HMM': 'iso_with_hmm_token_features.json',
            'ISO-only': 'iso_only_token_features.json',
            'HMM-as-ISO-feat': 'hmm_as_iso_feature_only.json',
            'HMM-only': 'hmm_only_token_features.json',
        }

    results = {}
    for name, filename in approaches.items():
        file_path = script_dir / 'outputs' / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                results[name] = data['perplexities']
        else:
            print(f"Warning: {filename} not found")

    return results


def calculate_metrics(data):
    """Calculate separation metrics."""
    human = np.array(data['human'])
    coedit = np.array(data['coedit'])
    llama = np.array(data['llama'])
    gemini = np.array(data['gemini'])

    models_mean = np.mean(np.concatenate([coedit, llama, gemini]))
    human_mean = np.mean(human)

    pct_diff = ((human_mean - models_mean) / human_mean) * 100

    return {
        'human_mean': human_mean,
        'coedit_mean': np.mean(coedit),
        'llama_mean': np.mean(llama),
        'gemini_mean': np.mean(gemini),
        'models_mean': models_mean,
        'pct_diff': pct_diff,
        'n_samples': len(human)
    }


def main():
    print("=" * 120)
    print("COMPARISON: Per-Edit vs Per-Sentence Analysis")
    print("=" * 120)
    print()
    print("Per-Edit Analysis:")
    print("  - Research Question: Can we distinguish individual edits?")
    print("  - Granularity: One score per edit")
    print("  - Use Case: Matches inference behavior during GRPO training")
    print()
    print("Per-Sentence Analysis:")
    print("  - Research Question: Can we distinguish editing strategies?")
    print("  - Granularity: One score per sentence (all edits combined)")
    print("  - Use Case: Evaluates how humans vs models approach multi-edit scenarios")
    print()

    # Load both sets of results
    per_edit = load_results('')  # Default prefix (no prefix)
    per_sentence = load_results('per_sentence')

    # Compare each approach
    print("=" * 120)
    print(f"{'Approach':<20} {'Level':<15} {'Human':<10} {'Models':<10} {'Sep.%':<10} {'N Samples':<12}")
    print("-" * 120)

    for approach in ['ISO+HMM', 'ISO-only', 'HMM-as-ISO-feat', 'HMM-only']:
        if approach in per_edit:
            metrics_edit = calculate_metrics(per_edit[approach])
            print(f"{approach:<20} {'Per-Edit':<15} "
                  f"{metrics_edit['human_mean']:<10.4f} "
                  f"{metrics_edit['models_mean']:<10.4f} "
                  f"{metrics_edit['pct_diff']:<10.2f}% "
                  f"{metrics_edit['n_samples']:<12}")

        if approach in per_sentence:
            metrics_sent = calculate_metrics(per_sentence[approach])
            print(f"{approach:<20} {'Per-Sentence':<15} "
                  f"{metrics_sent['human_mean']:<10.4f} "
                  f"{metrics_sent['models_mean']:<10.4f} "
                  f"{metrics_sent['pct_diff']:<10.2f}% "
                  f"{metrics_sent['n_samples']:<12}")

        print()

    print("=" * 120)
    print("KEY INSIGHTS")
    print("=" * 120)
    print()

    # Calculate averages
    if 'ISO+HMM' in per_edit and 'ISO+HMM' in per_sentence:
        edit_metrics = calculate_metrics(per_edit['ISO+HMM'])
        sent_metrics = calculate_metrics(per_sentence['ISO+HMM'])

        print(f"ISO+HMM (Best Approach):")
        print(f"  Per-Edit:     {edit_metrics['pct_diff']:+.2f}% separation ({edit_metrics['n_samples']} samples)")
        print(f"  Per-Sentence: {sent_metrics['pct_diff']:+.2f}% separation ({sent_metrics['n_samples']} samples)")
        print()

        if abs(sent_metrics['pct_diff']) > abs(edit_metrics['pct_diff']):
            print(f"  → Per-Sentence shows STRONGER discrimination ({sent_metrics['pct_diff'] - edit_metrics['pct_diff']:+.2f}% more)")
            print(f"  → Editing strategies are more distinguishable than individual edits")
        else:
            print(f"  → Per-Edit shows STRONGER discrimination ({edit_metrics['pct_diff'] - sent_metrics['pct_diff']:+.2f}% more)")
            print(f"  → Individual edits are more distinguishable than editing strategies")

    print()
    print("Sample Size Comparison:")
    if 'ISO+HMM' in per_edit and 'ISO+HMM' in per_sentence:
        print(f"  Per-Edit has {edit_metrics['n_samples']} samples (many edits per sentence)")
        print(f"  Per-Sentence has {sent_metrics['n_samples']} samples (one per sentence)")
        print(f"  Ratio: {edit_metrics['n_samples'] / sent_metrics['n_samples']:.1f}x more edit samples")

    print()
    print("=" * 120)


if __name__ == "__main__":
    main()
