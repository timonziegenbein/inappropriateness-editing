# Generated Edits Directory

This directory contains the generated edits from various models in JSONL format.

## Contents (not included in submission due to size constraints)

The full outputs include:
- Base model edits (Llama-3.1-8B-Instruct without fine-tuning)
- GRPO-trained model edits with different scorer configurations
- Baseline comparisons (GPT-4, GPT-4o-mini, Gemini-2.5-pro)
- Ablation study results (models trained without specific scorers)

## Size

Total size: ~63MB

## Generation

These files are generated using `models/generate_edits.py`:

```bash
# Generate edits from trained model
python models/generate_edits.py \
    --checkpoint_root <checkpoint_path> \
    --output_jsonl models/generated_edits/<model_name>.jsonl

# Generate edits from base model
python models/generate_edits.py \
    --use_base_model_only \
    --output_jsonl models/generated_edits/base_model.jsonl
```

## Usage

These generated edits can be evaluated with different scorer configurations using:

```bash
python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/<model_name>.jsonl \
    --output_jsonl models/predictions/<model_name>_evaluated.jsonl
```

See `models/EVALUATION_WORKFLOW.md` for detailed documentation.
