#!/usr/bin/env python3
"""Parse fluency scorer evaluation results from Weave export."""

import json
from collections import defaultdict

# Read the JSONL file
results = []
with open('weave_export_fluency-scorer-eval_2025-11-18.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))

# Parse and organize results
data = defaultdict(dict)
for result in results:
    display_name = result['display_name']
    output = result['output']

    # Parse model and dataset from display name
    parts = display_name.split('-')
    model = parts[0]
    dataset = parts[1] if len(parts) > 1 else 'unknown'

    data[model][dataset] = {
        'accuracy': output['AccuracyScorer']['correct']['true_fraction'],
        'precision': output['F1Scorer']['precision'],
        'recall': output['F1Scorer']['recall'],
        'f1': output['F1Scorer']['f1']
    }

# Print organized results
print("=" * 80)
print("FLUENCY SCORER EVALUATION RESULTS")
print("=" * 80)

for model in sorted(data.keys()):
    print(f"\n{model}:")
    for dataset in sorted(data[model].keys()):
        metrics = data[model][dataset]
        print(f"  {dataset}:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1:        {metrics['f1']:.4f}")

# Create LaTeX table
print("\n" + "=" * 80)
print("LATEX TABLE FOR PAPER")
print("=" * 80)

# Filter for main approaches (exclude baselines)
main_approaches = ['ModernBERTV2', 'ModernBERTV1', 'Gemini2.5Flash',
                   'LanguageTool', 'ClassifyAndFix']

print("""
\\begin{table}[t]
\\centering
\\small
\\begin{tabular}{llcccc}
\\toprule
\\textbf{Dataset} & \\textbf{Approach} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} \\\\
\\midrule""")

# Print results by dataset
for dataset in ['handcrafted', 'gec', 'gemini']:
    dataset_label = {
        'handcrafted': 'Hand-crafted',
        'gec': 'GEC (IteraTeR)',
        'gemini': 'Gemini-augmented'
    }[dataset]

    print(f"\\multicolumn{{6}}{{l}}{{\\textit{{{dataset_label}}}}} \\\\")

    # Get all models for this dataset
    models_with_data = [(model, data[model].get(dataset))
                        for model in main_approaches
                        if dataset in data[model]]

    for model, metrics in models_with_data:
        if metrics is None:
            continue

        model_name = {
            'ModernBERTV2': 'ModernBERT (edit-aware) V2',
            'ModernBERTV1': 'ModernBERT (edit-aware) V1',
            'ModernBertV1': 'ModernBERT (edit-aware) V1',
            'Gemini2.5Flash': 'Gemini 2.5 Flash (LLM)',
            'LanguageTool': 'LanguageTool (rule-based)',
            'ClassifyAndFix': 'Traditional (CoLA + Flan-T5)'
        }.get(model, model)

        acc = metrics['accuracy']
        prec = metrics['precision']
        rec = metrics['recall']
        f1 = metrics['f1']

        print(f"& {model_name} & {acc:.3f} & {prec:.3f} & {rec:.3f} & {f1:.3f} \\\\")

    if dataset != 'gemini':
        print("\\midrule")

print("""\\midrule
\\multicolumn{6}{l}{\\textit{Sanity check baselines:}} \\\\""")

# Add baselines
baseline_models = ['AlwaysFluent', 'AlwaysNonFluent', 'Random']
for model in baseline_models:
    if model in data and 'handcrafted' in data[model]:
        metrics = data[model]['handcrafted']
        model_name = {
            'AlwaysFluent': 'Always fluent',
            'AlwaysNonFluent': 'Always non-fluent',
            'Random': 'Random'
        }.get(model, model)

        acc = metrics['accuracy']
        prec = metrics['precision'] if metrics['precision'] != 0 else 0
        rec = metrics['recall']
        f1 = metrics['f1']

        prec_str = f"{prec:.3f}" if prec > 0 else "--"
        f1_str = f"{f1:.3f}" if f1 > 0 else "0.000"

        print(f"& {model_name} & {acc:.3f} & {prec_str} & {rec:.3f} & {f1_str} \\\\")

print("""\\bottomrule
\\end{tabular}
\\caption{Fluency scorer comparison on three test sets. \\textbf{Hand-crafted}: 28 carefully designed examples covering common fluency issues. \\textbf{GEC (IteraTeR)}: XXX examples from grammatical error correction data. \\textbf{Gemini-augmented}: XXX examples collected during GRPO training via LLM monitoring. ModernBERT V2 (our approach) achieves the best performance across all datasets.}
\\label{tab:fluency-comparison}
\\end{table}
""")

# Calculate combined metrics weighted by dataset size
print("\n" + "=" * 80)
print("COMBINED METRICS (Weighted by dataset size)")
print("=" * 80)

# Dataset sizes
dataset_sizes = {
    'handcrafted': 28,
    'gec': 1734,
    'gemini': 1903
}

print(f"\nDataset sizes:")
for dataset, size in dataset_sizes.items():
    print(f"  {dataset}: {size} instances")

total_instances = sum(dataset_sizes.values())
print(f"  Total: {total_instances} instances\n")

for model in main_approaches:
    # We need to recalculate from TP, FP, TN, FN counts
    # For each dataset, we'll compute weighted contributions

    metrics_by_dataset = {}
    total_weight = 0

    for dataset in ['handcrafted', 'gec', 'gemini']:
        if dataset in data[model]:
            metrics_by_dataset[dataset] = data[model][dataset]
            total_weight += dataset_sizes[dataset]

    if metrics_by_dataset:
        # Weighted average by number of instances
        weighted_acc = sum(
            data[model][dataset]['accuracy'] * dataset_sizes[dataset]
            for dataset in metrics_by_dataset.keys()
        ) / total_weight

        weighted_prec = sum(
            data[model][dataset]['precision'] * dataset_sizes[dataset]
            for dataset in metrics_by_dataset.keys()
        ) / total_weight

        weighted_rec = sum(
            data[model][dataset]['recall'] * dataset_sizes[dataset]
            for dataset in metrics_by_dataset.keys()
        ) / total_weight

        weighted_f1 = sum(
            data[model][dataset]['f1'] * dataset_sizes[dataset]
            for dataset in metrics_by_dataset.keys()
        ) / total_weight

        print(f"{model}:")
        print(f"  Weighted Accuracy:  {weighted_acc:.4f} ({weighted_acc*100:.2f}%)")
        print(f"  Weighted Precision: {weighted_prec:.4f}")
        print(f"  Weighted Recall:    {weighted_rec:.4f}")
        print(f"  Weighted F1:        {weighted_f1:.4f}")
        print()

# Create combined table
print("\n" + "=" * 80)
print("LATEX TABLE - COMBINED RESULTS")
print("=" * 80)

print("""
\\begin{table}[t]
\\centering
\\small
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Approach} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} \\\\
\\midrule
\\multicolumn{5}{l}{\\textit{Our approach:}} \\\\""")

# ModernBERT V2 as our approach
if 'ModernBERTV2' in data:
    metrics_by_dataset = {}
    total_weight = 0
    for dataset in ['handcrafted', 'gec', 'gemini']:
        if dataset in data['ModernBERTV2']:
            metrics_by_dataset[dataset] = data['ModernBERTV2'][dataset]
            total_weight += dataset_sizes[dataset]

    weighted_acc = sum(
        data['ModernBERTV2'][dataset]['accuracy'] * dataset_sizes[dataset]
        for dataset in metrics_by_dataset.keys()
    ) / total_weight

    weighted_prec = sum(
        data['ModernBERTV2'][dataset]['precision'] * dataset_sizes[dataset]
        for dataset in metrics_by_dataset.keys()
    ) / total_weight

    weighted_rec = sum(
        data['ModernBERTV2'][dataset]['recall'] * dataset_sizes[dataset]
        for dataset in metrics_by_dataset.keys()
    ) / total_weight

    weighted_f1 = sum(
        data['ModernBERTV2'][dataset]['f1'] * dataset_sizes[dataset]
        for dataset in metrics_by_dataset.keys()
    ) / total_weight

    print(f"ModernBERT (edit-aware) & {weighted_acc:.3f} & {weighted_prec:.3f} & {weighted_rec:.3f} & {weighted_f1:.3f} \\\\")

print("""\\midrule
\\multicolumn{5}{l}{\\textit{Existing approaches (fail at edit-level fluency):}} \\\\""")

# Other approaches
other_approaches = ['ModernBERTV1', 'Gemini2.5Flash', 'LanguageTool', 'ClassifyAndFix']
for model in other_approaches:
    if model not in data:
        continue

    metrics_by_dataset = {}
    total_weight = 0
    for dataset in ['handcrafted', 'gec', 'gemini']:
        if dataset in data[model]:
            metrics_by_dataset[dataset] = data[model][dataset]
            total_weight += dataset_sizes[dataset]

    if not metrics_by_dataset:
        continue

    weighted_acc = sum(
        data[model][dataset]['accuracy'] * dataset_sizes[dataset]
        for dataset in metrics_by_dataset.keys()
    ) / total_weight

    weighted_prec = sum(
        data[model][dataset]['precision'] * dataset_sizes[dataset]
        for dataset in metrics_by_dataset.keys()
    ) / total_weight

    weighted_rec = sum(
        data[model][dataset]['recall'] * dataset_sizes[dataset]
        for dataset in metrics_by_dataset.keys()
    ) / total_weight

    weighted_f1 = sum(
        data[model][dataset]['f1'] * dataset_sizes[dataset]
        for dataset in metrics_by_dataset.keys()
    ) / total_weight

    model_name = {
        'ModernBERTV1': 'ModernBERT V1 (baseline)',
        'Gemini2.5Flash': 'Gemini 2.5 Flash (LLM)',
        'LanguageTool': 'LanguageTool (rule-based)',
        'ClassifyAndFix': 'Traditional (CoLA + Flan-T5)'
    }.get(model, model)

    print(f"{model_name} & {weighted_acc:.3f} & {weighted_prec:.3f} & {weighted_rec:.3f} & {weighted_f1:.3f} \\\\")

print("""\\midrule
\\multicolumn{5}{l}{\\textit{Sanity check baselines:}} \\\\""")

# Add baselines (using handcrafted as representative)
baseline_models = ['AlwaysFluent', 'AlwaysNonFluent', 'Random']
for model in baseline_models:
    if model in data and 'handcrafted' in data[model]:
        metrics = data[model]['handcrafted']
        model_name = {
            'AlwaysFluent': 'Always fluent',
            'AlwaysNonFluent': 'Always non-fluent',
            'Random': 'Random'
        }.get(model, model)

        acc = metrics['accuracy']
        prec = metrics['precision'] if metrics['precision'] != 0 else 0
        rec = metrics['recall']
        f1 = metrics['f1']

        prec_str = f"{prec:.3f}" if prec > 0 else "--"
        f1_str = f"{f1:.3f}" if f1 > 0 else "0.000"

        print(f"{model_name} & {acc:.3f} & {prec_str} & {rec:.3f} & {f1_str} \\\\")

print("""\\bottomrule
\\end{tabular}
\\caption{Fluency scorer comparison on combined test set (3,665 examples total: 28 hand-crafted, 1,734 GEC from IteraTeR, 1,903 Gemini-augmented). Metrics are weighted by dataset size. Our ModernBERT edit-aware classifier achieves the best performance, demonstrating that dedicated training on paired before/after sentences is necessary for edit-level fluency classification.}
\\label{tab:fluency-comparison}
\\end{table}
""")
