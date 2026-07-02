"""
Compare Isolation Forest approaches using binary predictions and classification metrics.

This script:
1. Trains each ISO approach (ISO-only, ISO+HMM, HMM-as-ISO-feature)
2. Gets binary predictions for all sources (human, coedit, llama, gemini)
3. Computes F1, precision, recall, accuracy
4. Creates correlation matrix showing agreement between approaches
5. Visualizes confusion matrices
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
from scipy.stats import pearsonr
import sys
import logging

# Import functions from compute_hmm_isolation_scores
sys.path.insert(0, str(Path(__file__).parent))
from compute_hmm_isolation_scores import (
    train_models, compute_scores, load_predictions
)
from transformers import AutoTokenizer
from multiprocessing import cpu_count

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
script_dir = Path(__file__).parent

# Use all available CPU workers
num_workers = max(1, cpu_count() - 1)
logger.info(f"Using {num_workers} workers")

print("="*80)
print("ISOLATION FOREST COMPARISON: Binary Predictions & Classification Metrics")
print("="*80)

# Initialize tokenizer
logger.info("\nInitializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.1-8B-Instruct")

# Define file paths for train/test splits
human_train_file = script_dir / "data" / "human_with_edits_train.jsonl"
human_test_file = script_dir / "data" / "human_with_edits_test.jsonl"

# Test set: human test + all model predictions
test_sources_files = {
    'human': human_test_file,
    'coedit': script_dir / "data" / "coedit_with_edits.jsonl",
    'llama': script_dir / "data" / "llama_with_edits.jsonl",
    'gemini': script_dir / "data" / "gemini_with_edits.jsonl"
}

# Define approaches to compare
approaches = {
    'ISO-only': {'iso_only': True, 'hmm_only': False, 'hmm_as_feature_only': False},
    'ISO+HMM': {'iso_only': False, 'hmm_only': False, 'hmm_as_feature_only': False},
    'HMM-as-ISO-feature': {'iso_only': False, 'hmm_only': False, 'hmm_as_feature_only': True},
}

# Store predictions for each approach
all_predictions = {}  # approach -> source -> list of binary predictions
all_labels = {}  # source -> list of ground truth labels

print("\n" + "="*80)
print("TRAINING MODELS AND COMPUTING PREDICTIONS")
print("="*80)

for approach_name, approach_flags in approaches.items():
    print(f"\n{'='*80}")
    print(f"APPROACH: {approach_name}")
    print(f"{'='*80}")

    # Train model on TRAIN SET
    logger.info(f"Training {approach_name} on train set ({human_train_file.name})...")

    # Create model prefix based on approach name
    model_prefix = approach_name.lower().replace(' ', '_').replace('-', '_') + "_"

    hmm_model, iso_model, operation_vocab = train_models(
        str(human_train_file),
        tokenizer,
        max_examples=None,
        num_workers=num_workers,
        save_models=True,
        model_prefix=model_prefix,
        **approach_flags
    )

    # Compute predictions for TEST SET (human test + models)
    approach_predictions = {}
    for source_name, file_path in test_sources_files.items():
        logger.info(f"\nScoring {source_name} test set...")
        predictions = load_predictions(str(file_path))

        scores, binary_preds = compute_scores(
            predictions,
            source_name,
            hmm_model,
            iso_model,
            operation_vocab,
            tokenizer,
            num_workers=num_workers,
            hmm_as_feature_only=approach_flags['hmm_as_feature_only'],
            return_predictions=True
        )

        approach_predictions[source_name] = binary_preds

        # Store ground truth labels (only once)
        if source_name not in all_labels:
            # Human = 1 (normal/inlier), Models = -1 (anomaly/outlier)
            if source_name == 'human':
                all_labels[source_name] = [1] * len(binary_preds)
            else:
                all_labels[source_name] = [-1] * len(binary_preds)

    all_predictions[approach_name] = approach_predictions

print("\n" + "="*80)
print("CLASSIFICATION METRICS")
print("="*80)

# Compute metrics for each approach
results_table = []
for approach_name in approaches.keys():
    print(f"\n{approach_name}:")
    print("-" * 80)

    # Combine all predictions and labels
    all_y_true = []
    all_y_pred = []

    for source_name in test_sources_files.keys():
        y_true = all_labels[source_name]
        y_pred = all_predictions[approach_name][source_name]

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        # Per-source metrics
        acc = accuracy_score(y_true, y_pred)

        # For binary classification with labels {-1, 1}
        # We want to know: how well does it classify human (1) vs model (-1)?
        # Treating 1 (normal) as positive class
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        print(f"\n  {source_name.upper()}:")
        print(f"    Accuracy: {acc:.4f}")
        print(f"    Precision (normal): {prec:.4f}")
        print(f"    Recall (normal): {rec:.4f}")
        print(f"    F1 (normal): {f1:.4f}")

        # Show distribution of predictions
        pred_counts = np.bincount(np.array(y_pred) + 1)  # Convert -1,1 to 0,1,2
        if len(pred_counts) > 1:
            n_anomaly = pred_counts[0] if len(pred_counts) > 0 else 0
            n_normal = pred_counts[2] if len(pred_counts) > 2 else 0
        else:
            n_anomaly = 0
            n_normal = 0
        print(f"    Predicted: {n_normal} normal, {n_anomaly} anomaly")

    # Overall metrics
    print(f"\n  OVERALL (all sources combined):")
    acc_overall = accuracy_score(all_y_true, all_y_pred)
    prec_overall = precision_score(all_y_true, all_y_pred, pos_label=1, zero_division=0)
    rec_overall = recall_score(all_y_true, all_y_pred, pos_label=1, zero_division=0)
    f1_overall = f1_score(all_y_true, all_y_pred, pos_label=1, zero_division=0)

    print(f"    Accuracy: {acc_overall:.4f}")
    print(f"    Precision: {prec_overall:.4f}")
    print(f"    Recall: {rec_overall:.4f}")
    print(f"    F1: {f1_overall:.4f}")

    results_table.append({
        'approach': approach_name,
        'accuracy': acc_overall,
        'precision': prec_overall,
        'recall': rec_overall,
        'f1': f1_overall
    })

# Print summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"\n{'Approach':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-" * 80)
for row in results_table:
    print(f"{row['approach']:<25} {row['accuracy']:<12.4f} {row['precision']:<12.4f} {row['recall']:<12.4f} {row['f1']:<12.4f}")

# Compute correlation between approaches
print("\n" + "="*80)
print("APPROACH AGREEMENT ANALYSIS")
print("="*80)

# For each pair of approaches, compute agreement
approach_names = list(approaches.keys())
n_approaches = len(approach_names)

# Combine all predictions across sources
combined_preds = {}
for approach_name in approach_names:
    preds = []
    for source_name in test_sources_files.keys():
        preds.extend(all_predictions[approach_name][source_name])
    combined_preds[approach_name] = np.array(preds)

# Compute correlation matrix
corr_matrix = np.zeros((n_approaches, n_approaches))
for i, app1 in enumerate(approach_names):
    for j, app2 in enumerate(approach_names):
        if i == j:
            corr_matrix[i, j] = 1.0
        else:
            # Pearson correlation between binary predictions
            corr, _ = pearsonr(combined_preds[app1], combined_preds[app2])
            corr_matrix[i, j] = corr

print("\nCorrelation Matrix (Pearson correlation between predictions):")
print(f"\n{'':<25}", end='')
for app in approach_names:
    print(f"{app:<25}", end='')
print()
print("-" * (25 + 25 * n_approaches))
for i, app1 in enumerate(approach_names):
    print(f"{app1:<25}", end='')
    for j in range(n_approaches):
        print(f"{corr_matrix[i, j]:<25.4f}", end='')
    print()

# Agreement percentages
print("\nAgreement Percentages:")
for i, app1 in enumerate(approach_names):
    for j in range(i+1, n_approaches):
        app2 = approach_names[j]
        agreement = np.mean(combined_preds[app1] == combined_preds[app2])
        print(f"  {app1} vs {app2}: {agreement*100:.2f}%")

# Visualizations
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# 1. Create correlation heatmap
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0.5,
            xticklabels=approach_names, yticklabels=approach_names,
            vmin=0, vmax=1, square=True, cbar_kws={'label': 'Pearson Correlation'})
ax.set_title('Prediction Agreement Between ISO Approaches', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(script_dir / "outputs" / "iso_approach_correlation.png", dpi=300, bbox_inches='tight')
print("✓ Saved correlation heatmap")
plt.close()

# 2. Create confusion matrices for each approach
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, approach_name in enumerate(approach_names):
    ax = axes[idx]

    # Combine all predictions
    y_true = []
    y_pred = []
    for source_name in test_sources_files.keys():
        y_true.extend(all_labels[source_name])
        y_pred.extend(all_predictions[approach_name][source_name])

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])

    # Normalize to percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                xticklabels=['Normal (1)', 'Anomaly (-1)'],
                yticklabels=['Normal (1)', 'Anomaly (-1)'],
                cbar_kws={'label': 'Percentage (%)'})
    ax.set_title(f'{approach_name}', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.suptitle('Confusion Matrices (Percentages)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(script_dir / "outputs" / "iso_confusion_matrices.png", dpi=300, bbox_inches='tight')
print("✓ Saved confusion matrices")
plt.close()

# 3. Bar plot of metrics
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

x = np.arange(len(approach_names))
width = 0.2

metrics = ['accuracy', 'precision', 'recall', 'f1']
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

for i, metric in enumerate(metrics):
    values = [row[metric] for row in results_table]
    ax.bar(x + i * width, values, width, label=metric.capitalize(), color=colors[i], alpha=0.8)

ax.set_xlabel('Approach', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Classification Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(approach_names)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(script_dir / "outputs" / "iso_metrics_comparison.png", dpi=300, bbox_inches='tight')
print("✓ Saved metrics comparison bar plot")
plt.close()

# Save results to JSON
output = {
    'predictions': {
        approach: {
            source: preds
            for source, preds in approach_preds.items()
        }
        for approach, approach_preds in all_predictions.items()
    },
    'labels': all_labels,
    'metrics': results_table,
    'correlation_matrix': corr_matrix.tolist(),
    'approach_names': approach_names
}

with open(script_dir / "outputs" / "iso_prediction_comparison.json", 'w') as f:
    json.dump(output, f, indent=2)

print("✓ Saved results to iso_prediction_comparison.json")

print("\n" + "="*80)
print("DONE")
print("="*80)
