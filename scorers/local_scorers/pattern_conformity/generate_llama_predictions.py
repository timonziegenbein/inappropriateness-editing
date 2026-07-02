"""
Generate predictions for the IteraTeR test dataset using the Llama model.

This script uses the same model and prompt format as used in GRPO training
to generate edits for comparison with human edits and CoEdit predictions.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import json
import argparse


def load_model_and_tokenizer(model_name, device="cuda"):
    """Load the Llama model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Check available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Found {num_gpus} GPU(s)")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"  # This will automatically distribute across all available GPUs
    )

    print(f"Model loaded with device_map=auto (using all available GPUs)")
    return tokenizer, model


def generate_predictions(dataset, tokenizer, model, batch_size=16, max_new_tokens=256, device="cuda"):
    """Generate predictions for the entire dataset using chat template with batching."""
    print(f"\nGenerating predictions for {len(dataset)} examples...")

    predictions = []

    # Set padding token and side for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for causal LM batching

    # Process in batches
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        batch_size_actual = len(batch['before_sent'])

        # Create messages for each example in batch
        all_messages = []
        for j in range(batch_size_actual):
            # Add instruction to only output the edited text
            user_content = f"{batch['before_sent'][j]}\n\nProvide only the edited text without any explanations or notes."

            messages = [
                {"role": "user", "content": user_content},
            ]
            all_messages.append(messages)

        # Apply chat template to all messages
        batch_inputs = []
        for messages in all_messages:
            formatted = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,  # Get text first
            )
            batch_inputs.append(formatted)

        # Tokenize the batch
        inputs = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated part (exclude the prompt)
        for j in range(batch_size_actual):
            input_length = inputs["input_ids"][j].shape[0]
            generated_text = tokenizer.decode(outputs[j][input_length:], skip_special_tokens=True)

            predictions.append({
                'original_intent': batch['original_intent'][j],
                'prefix': batch['prefix'][j],
                'original_before_sent': batch['original_before_sent'][j],
                'before_sent': batch['before_sent'][j],
                'after_sent': batch['after_sent'][j],
                'llama_prediction': generated_text.strip(),
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
    parser = argparse.ArgumentParser(description="Generate predictions using Llama model.")
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.1-8B-Instruct",
                        help="Model name to use")
    parser.add_argument("--dataset_name", type=str, default="timonziegenbein/iterater-test-with-prefixes",
                        help="HuggingFace dataset name")
    parser.add_argument("--output_file", type=str, default="data/llama_predictions.jsonl",
                        help="Output file for predictions")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    print(f"Configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Output: {args.output_file}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Device: {args.device}")

    # Load dataset
    print(f"\nLoading dataset {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, split="train")
    print(f"Loaded {len(dataset)} examples")

    # Load model
    tokenizer, model = load_model_and_tokenizer(args.model_name, args.device)

    # Generate predictions
    predictions = generate_predictions(
        dataset,
        tokenizer,
        model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device
    )

    # Save results
    save_predictions(predictions, args.output_file)

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
        print(f"  Llama prediction: {pred['llama_prediction'][:100]}...")

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
