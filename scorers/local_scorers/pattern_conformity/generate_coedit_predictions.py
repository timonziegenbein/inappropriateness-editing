"""
Generate predictions for the IteraTeR test dataset using the CoEdit model.

This script loads the timonziegenbein/iterater-test-with-prefixes dataset
and generates edits using the grammarly/coedit-large model.
"""

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm
import json


def load_model_and_tokenizer(model_name="grammarly/coedit-large", device="cuda"):
    """Load the CoEdit model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    if torch.cuda.is_available() and device == "cuda":
        model = model.to(device)
        print(f"Model loaded on GPU")
    else:
        print(f"Model loaded on CPU")

    return tokenizer, model


def generate_predictions(dataset, tokenizer, model, batch_size=8, max_length=256, device="cuda"):
    """Generate predictions for the entire dataset."""
    print(f"\nGenerating predictions for {len(dataset)} examples...")

    predictions = []

    # Process in batches
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]

        # Get input texts (already formatted as "prefix: sentence")
        input_texts = batch['before_sent']

        # Tokenize
        input_ids = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).input_ids

        if device == "cuda":
            input_ids = input_ids.to(device)

        # Generate (greedy decoding)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=max_length
            )

        # Decode
        edited_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Store results
        for j, edited_text in enumerate(edited_texts):
            predictions.append({
                'original_intent': batch['original_intent'][j],
                'prefix': batch['prefix'][j],
                'original_before_sent': batch['original_before_sent'][j],
                'before_sent': batch['before_sent'][j],
                'after_sent': batch['after_sent'][j],
                'coedit_prediction': edited_text
            })

    return predictions


def save_predictions(predictions, output_file):
    """Save predictions to a JSONL file."""
    print(f"\nSaving {len(predictions)} predictions to {output_file}...")

    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

    print(f"✓ Predictions saved to {output_file}")


def main():
    # Configuration
    model_name = "grammarly/coedit-large"
    dataset_name = "timonziegenbein/iterater-test-with-prefixes"
    output_file = "data/coedit_predictions.jsonl"
    batch_size = 8
    max_length = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Output: {output_file}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max length: {max_length}")
    print(f"  Device: {device}")

    # Load dataset
    print(f"\nLoading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    print(f"Loaded {len(dataset)} examples")

    # Load model
    tokenizer, model = load_model_and_tokenizer(model_name, device)

    # Generate predictions
    predictions = generate_predictions(
        dataset,
        tokenizer,
        model,
        batch_size=batch_size,
        max_length=max_length,
        device=device
    )

    # Save results
    save_predictions(predictions, output_file)

    # Print some examples
    print("\n" + "="*80)
    print("Sample predictions:")
    print("="*80)
    for i in range(min(3, len(predictions))):
        pred = predictions[i]
        print(f"\nExample {i+1}:")
        print(f"  Intent: {pred['original_intent']}")
        print(f"  Prefix: {pred['prefix']}")
        print(f"  Original: {pred['original_before_sent'][:100]}...")
        print(f"  Human edit: {pred['after_sent'][:100]}...")
        print(f"  CoEdit prediction: {pred['coedit_prediction'][:100]}...")

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
