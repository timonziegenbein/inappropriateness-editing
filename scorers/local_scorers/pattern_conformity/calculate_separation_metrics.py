"""
Calculate separation metrics between human and model scores.

Computes:
1. Absolute difference (Human - Mean(Models))
2. Percentage difference ((Human - Mean(Models)) / Human * 100)
3. Effect size (Cohen's d)
"""
import json
import numpy as np
from pathlib import Path


def calculate_cohens_d(human_scores, model_scores):
    """Calculate Cohen's d effect size."""
    mean_diff = np.mean(human_scores) - np.mean(model_scores)
    pooled_std = np.sqrt((np.std(human_scores)**2 + np.std(model_scores)**2) / 2)
    return mean_diff / pooled_std if pooled_std > 0 else 0.0


def analyze_approach(approach_name, results_file):
    """Analyze separation metrics for one approach."""
    with open(results_file, 'r') as f:
        data = json.load(f)

    # Get scores
    human_scores = np.array(data['perplexities']['human'])
    coedit_scores = np.array(data['perplexities']['coedit'])
    llama_scores = np.array(data['perplexities']['llama'])
    gemini_scores = np.array(data['perplexities']['gemini'])

    # Combine all model scores
    all_model_scores = np.concatenate([coedit_scores, llama_scores, gemini_scores])

    # Calculate metrics
    human_mean = np.mean(human_scores)
    models_mean = np.mean(all_model_scores)

    # Absolute difference
    abs_diff = human_mean - models_mean

    # Percentage difference
    pct_diff = (abs_diff / human_mean) * 100

    # Cohen's d
    cohens_d = calculate_cohens_d(human_scores, all_model_scores)

    # Individual model means
    coedit_mean = np.mean(coedit_scores)
    llama_mean = np.mean(llama_scores)
    gemini_mean = np.mean(gemini_scores)

    return {
        'approach': approach_name,
        'human_mean': human_mean,
        'models_mean': models_mean,
        'coedit_mean': coedit_mean,
        'llama_mean': llama_mean,
        'gemini_mean': gemini_mean,
        'abs_diff': abs_diff,
        'pct_diff': pct_diff,
        'cohens_d': cohens_d
    }


def main():
    script_dir = Path(__file__).parent

    approaches = [
        ('ISO+HMM', script_dir / 'outputs' / 'iso_with_hmm_token_features.json'),
        ('ISO-only', script_dir / 'outputs' / 'iso_only_token_features.json'),
        ('HMM-as-ISO-feat', script_dir / 'outputs' / 'hmm_as_iso_feature_only.json'),
        ('HMM-only', script_dir / 'outputs' / 'hmm_only_token_features.json'),
    ]

    results = []
    for name, file_path in approaches:
        if file_path.exists():
            results.append(analyze_approach(name, file_path))
        else:
            print(f"Warning: {file_path.name} not found, skipping {name}")

    # Print results
    print("=" * 100)
    print("SEPARATION METRICS: Human vs Models")
    print("=" * 100)
    print()

    # Header
    cohens_d_header = "Cohen's d"
    print(f"{'Approach':<20} {'Human':<10} {'Models':<10} {'Abs Diff':<12} {'% Diff':<10} {cohens_d_header:<10}")
    print(f"{'':20} {'Mean':<10} {'Mean':<10} {'(H-M)':<12} {'(H-M)/H':<10} {'Effect':<10}")
    print("-" * 100)

    # Sort by percentage difference (descending)
    results.sort(key=lambda x: x['pct_diff'], reverse=True)

    for r in results:
        print(f"{r['approach']:<20} "
              f"{r['human_mean']:<10.4f} "
              f"{r['models_mean']:<10.4f} "
              f"{r['abs_diff']:<12.4f} "
              f"{r['pct_diff']:<10.2f}% "
              f"{r['cohens_d']:<10.3f}")

    print()
    print("=" * 100)
    print("DETAILED BREAKDOWN BY MODEL")
    print("=" * 100)
    print()

    for r in results:
        print(f"{r['approach']}:")
        print(f"  Human:  {r['human_mean']:.4f}")
        print(f"  CoEdIT: {r['coedit_mean']:.4f} (diff: {r['human_mean'] - r['coedit_mean']:+.4f})")
        print(f"  Llama:  {r['llama_mean']:.4f} (diff: {r['human_mean'] - r['llama_mean']:+.4f})")
        print(f"  Gemini: {r['gemini_mean']:.4f} (diff: {r['human_mean'] - r['gemini_mean']:+.4f})")
        print()

    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)
    print()
    print("Absolute Difference: How much higher human scores are than model scores")
    print("Percentage Difference: Relative separation (higher = better discrimination)")
    print("Cohen's d: Effect size (0.2=small, 0.5=medium, 0.8=large)")
    print()
    print(f"Best approach by % difference: {results[0]['approach']} ({results[0]['pct_diff']:.2f}%)")
    print(f"Best approach by effect size: {max(results, key=lambda x: x['cohens_d'])['approach']} "
          f"(d={max(results, key=lambda x: x['cohens_d'])['cohens_d']:.3f})")


if __name__ == "__main__":
    main()
