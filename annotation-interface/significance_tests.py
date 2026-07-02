"""
Compute significance tests between base model and v11 models on absolute annotation scores.
"""

import pandas as pd
from ast import literal_eval
from scipy import stats
import numpy as np


def load_annotation_data():
    """Load and preprocess annotation data."""
    df = pd.read_csv('./study_pairs_abs_results.csv')
    df['result'] = df['result'].apply(literal_eval)
    df['nat'] = df['result'].apply(lambda x: int(x['otherErrorQuestion1'][-1]))
    df['sim'] = df['result'].apply(lambda x: int(x['otherErrorQuestion2'][-1])-5
                                    if x['otherErrorQuestion2'][-2:] != '10'
                                    else int(x['otherErrorQuestion2'][-2:])-5)
    df['flu'] = df['result'].apply(lambda x: int(x['otherErrorQuestion3'][-2:])-10)
    df['hl'] = df['result'].apply(lambda x: int(x['otherErrorQuestion4'][-2:])-15)

    # Extract model name from post_id (simple split by underscore)
    df['id_model'] = df['post_id'].apply(lambda x: x.split('_')[1])

    # Keep only users 3, 4, 5
    df = df[df['user_id'].isin([3, 4, 5])]

    # Only keep post_ids that appear 3 times (once per user)
    ids_to_keep = df['post_id'].value_counts()[df['post_id'].value_counts() == 3].index.tolist()
    df = df[df['post_id'].isin(ids_to_keep)]

    return df


def compute_significance_tests(df):
    """
    Compute significance tests between base and v11 models using Wilcoxon signed-rank test.

    Pairs data at the (source_id, user_id) level to account for within-source comparisons.
    """
    categories = ['nat', 'sim', 'flu', 'hl']
    category_names = {
        'nat': 'Naturalness',
        'sim': 'Similarity',
        'flu': 'Fluency',
        'hl': 'Human-likeness'
    }

    # Identify base model and v11 model to compare
    base_model = 'base'
    comparison_model = 'v11'

    # Verify models exist in data
    available_models = df['id_model'].unique()
    print(f'\nAvailable models in data: {sorted(available_models)}')

    if base_model not in available_models or comparison_model not in available_models:
        raise ValueError(f'Required models not found in data!')

    # Extract source_id from post_id (first part before underscore)
    df['id_source'] = df['post_id'].apply(lambda x: x.split('_')[0])

    print('='*80)
    print('SIGNIFICANCE TESTS: Base Model vs V11 Model')
    print('='*80)
    print(f'\nBase model: {base_model}')
    print(f'V11 model: {comparison_model}')
    print(f'\nCategories: {list(category_names.values())}')
    print('\nUsing Wilcoxon signed-rank test (paired samples, one-sided)')
    print('Testing H1: V11 > Base')
    print('Pairing at (source_id, user_id) level')
    print('='*80)

    # Diagnostics: Check unique edits
    print(f'\nData diagnostics:')
    print(f'  Total unique post_ids (edits): {df["post_id"].nunique()}')
    print(f'  Total unique source_ids: {df["id_source"].nunique()}')
    print(f'  Total annotations: {len(df)}')

    # Create paired data by merging base and v11 on (id_source, user_id)
    base_data = df[df['id_model'] == base_model].copy()
    v11_data = df[df['id_model'] == comparison_model].copy()

    print(f'\nBase model:')
    print(f'  Unique post_ids: {base_data["post_id"].nunique()}')
    print(f'  Unique source_ids: {base_data["id_source"].nunique()}')
    print(f'  Total annotations: {len(base_data)}')

    print(f'\nV11 model:')
    print(f'  Unique post_ids: {v11_data["post_id"].nunique()}')
    print(f'  Unique source_ids: {v11_data["id_source"].nunique()}')
    print(f'  Total annotations: {len(v11_data)}')

    # Sort both datasets to ensure consistent ordering
    base_data = base_data.sort_values(['id_source', 'user_id']).reset_index(drop=True)
    v11_data = v11_data.sort_values(['id_source', 'user_id']).reset_index(drop=True)

    # Merge on source and user to create pairs
    paired_data = base_data.merge(
        v11_data,
        on=['id_source', 'user_id'],
        suffixes=('_base', '_v11')
    )

    # Sort the paired data for consistency
    paired_data = paired_data.sort_values(['id_source', 'user_id']).reset_index(drop=True)

    print(f'\n{"-"*80}')
    print(f'Comparing: {base_model} vs {comparison_model}')
    print(f'{"-"*80}')
    print(f'\nPaired samples: {len(paired_data)} pairs')
    print(f'  Unique source texts in pairs: {paired_data["id_source"].nunique()}')
    print(f'  Unique users: {paired_data["user_id"].nunique()}')
    print(f'  Unique base edits: {paired_data["post_id_base"].nunique()}')
    print(f'  Unique v11 edits: {paired_data["post_id_v11"].nunique()}')

    results = {}

    for category in categories:
        base_scores = paired_data[f'{category}_base'].values
        v11_scores = paired_data[f'{category}_v11'].values

        # Compute descriptive statistics
        base_mean = np.mean(base_scores)
        base_std = np.std(base_scores, ddof=1)
        v11_mean = np.mean(v11_scores)
        v11_std = np.std(v11_scores, ddof=1)

        # Compute differences
        differences = v11_scores - base_scores
        diff_mean = np.mean(differences)
        diff_std = np.std(differences, ddof=1)

        # Wilcoxon signed-rank test (paired samples, one-sided: v11 > base)
        # alternative='less' tests if base < v11, which is equivalent to v11 > base
        statistic, p_value = stats.wilcoxon(base_scores, v11_scores, alternative='less')

        # Effect size (rank-biserial correlation for paired data)
        # r = Z / sqrt(N)
        n_pairs = len(differences)
        z_score = stats.norm.ppf(1 - p_value/2)  # Approximate Z from p-value
        effect_size = z_score / np.sqrt(n_pairs)

        results[category] = {
            'base_mean': base_mean,
            'base_std': base_std,
            'v11_mean': v11_mean,
            'v11_std': v11_std,
            'diff_mean': diff_mean,
            'diff_std': diff_std,
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'n_pairs': n_pairs
        }

        # Print results
        print(f'\n{category_names[category]} ({category}):')
        print(f'  {base_model}: M={base_mean:.2f}, SD={base_std:.2f}')
        print(f'  {comparison_model}: M={v11_mean:.2f}, SD={v11_std:.2f}')
        print(f'  Difference (v11 - base): M={diff_mean:.2f}, SD={diff_std:.2f}')
        print(f'  Wilcoxon W = {statistic:.2f}, p = {p_value:.4f}')
        print(f'  Effect size (r) = {effect_size:.3f}')
        print(f'  N pairs = {n_pairs}')

        # Interpret significance
        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'
        print(f'  Significance: {sig}')

        # Interpret direction
        if diff_mean > 0:
            direction = f'{comparison_model} > {base_model}'
        elif diff_mean < 0:
            direction = f'{comparison_model} < {base_model}'
        else:
            direction = 'No difference'
        print(f'  Direction: {direction}')

    print('\n' + '='*80)
    print('SUMMARY TABLE')
    print('='*80)

    # Create summary table
    print(f'\n{"Category":<15} {"Base M(SD)":<15} {"V11 M(SD)":<15} {"Diff M(SD)":<15} {"p-value":<12} {"r":<8} {"Sig":<5}')
    print('-'*90)

    for category in categories:
        res = results[category]
        p_val = res['p_value']

        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'

        base_str = f'{res["base_mean"]:.2f}({res["base_std"]:.2f})'
        v11_str = f'{res["v11_mean"]:.2f}({res["v11_std"]:.2f})'
        diff_str = f'{res["diff_mean"]:.2f}({res["diff_std"]:.2f})'

        print(f'{category_names[category]:<15} {base_str:<15} {v11_str:<15} {diff_str:<15} '
              f'{p_val:<12.4f} {res["effect_size"]:<8.3f} {sig:<5}')

    print('\n' + '='*80)
    print('Significance levels: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant')
    print('One-sided test: Tests if V11 > Base')
    print('Effect size r = Z / sqrt(N) (small: 0.1, medium: 0.3, large: 0.5)')
    print('Positive difference means V11 > Base')
    print('='*80)

    return results


if __name__ == '__main__':
    print('Loading annotation data...')
    df = load_annotation_data()

    print(f'\nTotal annotations: {len(df)}')
    print(f'Unique post_ids: {df["post_id"].nunique()}')
    print(f'Models: {sorted(df["id_model"].unique())}')

    print('\nComputing significance tests...\n')
    results = compute_significance_tests(df)
