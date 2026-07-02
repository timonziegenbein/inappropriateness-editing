#!/bin/bash
# Run complete per-sentence analysis (editing strategy) to match per-edit analysis in paper

set -e  # Exit on error

cd "$(dirname "$0")"

echo "================================================================================"
echo "PER-SENTENCE ANALYSIS: Editing Strategy (All Edits Combined)"
echo "================================================================================"
echo
echo "This evaluates overall editing strategy across all edits in a sentence."
echo "Complements the per-edit analysis by examining sentence-level patterns."
echo
echo "================================================================================"

# 1. ISO+HMM (main model)
echo
echo "1/4: Training ISO+HMM (per-sentence)..."
python compute_hmm_isolation_scores.py \
    --per-sentence \
    --train-file data/human_with_edits_train.jsonl \
    --test-file data/human_with_edits_test.jsonl \
    --save-models \
    --model-prefix per_sentence_iso_hmm_ \
    --output-file outputs/per_sentence_iso_with_hmm_token_features.json

# 2. ISO-only
echo
echo "2/4: Training ISO-only (per-sentence)..."
python compute_hmm_isolation_scores.py \
    --per-sentence \
    --iso-only \
    --save-models \
    --model-prefix per_sentence_iso_only_ \
    --output-file outputs/per_sentence_iso_only_token_features.json

# 3. HMM-only (reuse HMM from step 1)
echo
echo "3/4: Training HMM-only (per-sentence)..."
python compute_hmm_isolation_scores.py \
    --per-sentence \
    --hmm-only \
    --load-hmm \
    --save-models \
    --model-prefix per_sentence_hmm_only_ \
    --output-file outputs/per_sentence_hmm_only_token_features.json

# 4. HMM-as-ISO-feature (reuse HMM from step 1)
echo
echo "4/4: Training HMM-as-ISO-feature (per-sentence)..."
python compute_hmm_isolation_scores.py \
    --per-sentence \
    --hmm-as-feature-only \
    --load-hmm \
    --save-models \
    --model-prefix per_sentence_hmm_as_iso_feature_ \
    --output-file outputs/per_sentence_hmm_as_iso_feature_only.json

echo
echo "================================================================================"
echo "DONE! Per-sentence analysis complete."
echo "================================================================================"
echo
echo "Generated outputs in outputs/:"
echo "  - per_sentence_iso_with_hmm_token_features.json"
echo "  - per_sentence_iso_only_token_features.json"
echo "  - per_sentence_hmm_only_token_features.json"
echo "  - per_sentence_hmm_as_iso_feature_only.json"
echo
echo "Generated models in models/:"
echo "  - per_sentence_iso_hmm_*.pkl"
echo "  - per_sentence_iso_only_*.pkl"
echo "  - per_sentence_hmm_only_*.pkl"
echo "  - per_sentence_hmm_as_iso_feature_*.pkl"
echo
echo "Next steps:"
echo "  1. Run calculate_separation_metrics.py to compare all approaches"
echo "  2. Run visualize_all_approaches.py to generate plots"
echo "  3. Run analyze_correlation.py to examine complementarity"
echo "================================================================================"
