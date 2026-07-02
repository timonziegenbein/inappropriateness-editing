#!/bin/bash
# Evaluate all v11 files with the new human-like scorer (PPO classifier)

echo "=================================="
echo "Evaluating all v11 files"
echo "=================================="

# Define all v11 model files to evaluate
models=(
    "grpo_global_sentence_no_fluency_v11"
    "grpo_global_sentence_no_human_like_v11"
    "grpo_global_sentence_no_semantic_similarity_v11"
    "grpo_global_sentence_only_fluency_v11"
    "grpo_global_sentence_only_human_like_v11"
    "grpo_global_sentence_only_semantic_similarity_v11"
)

# Process ablation models
for model in "${models[@]}"; do
    echo ""
    echo "Processing ${model}..."
    echo "-----------------------------------"

    input_file="models/generated_edits/${model}.jsonl"
    output_file="models/predictions/${model}_ppo_classifier.jsonl"

    if [ ! -f "$input_file" ]; then
        echo "WARNING: Input file not found: $input_file"
        continue
    fi

    echo "Input:  $input_file"
    echo "Output: $output_file"

    python models/evaluate_edits.py \
        --input_jsonl "$input_file" \
        --output_jsonl "$output_file"

    if [ $? -eq 0 ]; then
        echo "✓ ${model} completed successfully"
    else
        echo "✗ ${model} failed"
        exit 1
    fi
done

# Process round files (r2-r11)
for round in {2..11}; do
    echo ""
    echo "Processing round ${round}..."
    echo "-----------------------------------"

    input_file="models/generated_edits/grpo_global_sentence_v11_r${round}.jsonl"
    output_file="models/predictions/grpo_global_sentence_v11_r${round}_ppo_classifier.jsonl"

    if [ ! -f "$input_file" ]; then
        echo "WARNING: Input file not found: $input_file"
        continue
    fi

    echo "Input:  $input_file"
    echo "Output: $output_file"

    python models/evaluate_edits.py \
        --input_jsonl "$input_file" \
        --output_jsonl "$output_file"

    if [ $? -eq 0 ]; then
        echo "✓ Round ${round} completed successfully"
    else
        echo "✗ Round ${round} failed"
        exit 1
    fi
done

echo ""
echo "=================================="
echo "All v11 files completed!"
echo "=================================="
echo ""
echo "Evaluated files:"
echo "  - 6 ablation models (no_*, only_*)"
echo "  - 10 round files (r2-r11)"
echo "  - Total: 16 files"
echo "=================================="
