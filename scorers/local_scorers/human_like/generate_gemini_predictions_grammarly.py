"""
Generate predictions for the grammarly/coedit test dataset using Gemini.

This script loads the test split from datasets/scorers/grammarly_coedit/test.jsonl
and uses Google's Gemini API to generate text edits based on task prefixes.
"""

import os
import json
import argparse
import asyncio
from typing import List, Dict, Any
from tqdm.asyncio import tqdm_asyncio
from pathlib import Path
from google import genai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMTextEditor:
    """
    LLM-based text editor using Google's Gemini API.
    Uses structured prompting to perform text editing tasks.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", timeout: int = 60):
        """
        Initialize the LLM text editor.

        Args:
            model_name: Name of the Gemini model to use
            timeout: Timeout in seconds for API calls
        """
        self.model_name = model_name
        self.timeout = timeout
        self.system_prompt = "You are an expert text editor. Your task is to edit text according to specific instructions while preserving the core meaning."

        # Initialize Google GenAI client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.client = genai.Client(api_key=api_key)

    async def edit_text(self, before_sent: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Edit text using the LLM based on the instruction in before_sent.

        Args:
            before_sent: The full input with prefix (e.g., "Fix grammar: sentence text")
            max_retries: Maximum number of retries on failure

        Returns:
            Dictionary containing the edited text and metadata
        """
        # Construct the editing prompt - use the same format as Llama
        editing_prompt = f"""{before_sent}

Provide only the edited text without any explanations or notes."""

        # Retry logic with exponential backoff
        retry_delay = 1  # Start with 1 second

        for attempt in range(max_retries):
            try:
                # Make API call with timeout
                response = await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=self.model_name,
                        contents=editing_prompt,
                        config={
                            "system_instruction": self.system_prompt,
                            "temperature": 0.0,  # Deterministic output
                        }
                    ),
                    timeout=self.timeout
                )

                # Get the edited text
                edited_text = response.text.strip()

                return {
                    "success": True,
                    "edited_text": edited_text,
                    "error": None
                }

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Timeout after {max_retries} attempts")
                    return {
                        "success": False,
                        "edited_text": before_sent,  # Return original on failure
                        "error": f"Timeout after {max_retries} attempts"
                    }

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Error on attempt {attempt + 1}/{max_retries}: {e}, retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Error after {max_retries} attempts: {e}")
                    return {
                        "success": False,
                        "edited_text": before_sent,  # Return original on failure
                        "error": str(e)
                    }

        # Should never reach here, but just in case
        return {
            "success": False,
            "edited_text": before_sent,
            "error": "Max retries exceeded"
        }


def load_test_dataset(test_file):
    """Load test dataset from JSONL file."""
    logger.info(f"Loading test dataset from {test_file}...")
    dataset = []
    with open(test_file, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    logger.info(f"Loaded {len(dataset)} examples")
    return dataset


async def process_example(editor: LLMTextEditor, example: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single example through the LLM."""
    result = await editor.edit_text(example['original_before_sent'])

    return {
        'prefix': example.get('prefix', ''),
        'original_before_sent': example['original_before_sent'],
        'after_sent': example['after_sent'],
        'llm_prediction': result['edited_text'],
        'success': result['success'],
        'error': result.get('error')
    }


async def generate_predictions_async(dataset: List[Dict], model_name: str, concurrency: int = 10, timeout: int = 60) -> List[Dict]:
    """Generate predictions for all examples with controlled concurrency."""
    logger.info(f"Generating predictions for {len(dataset)} examples with concurrency={concurrency}...")

    editor = LLMTextEditor(model_name=model_name, timeout=timeout)

    # Create semaphore for concurrency control
    sem = asyncio.Semaphore(concurrency)

    async def process_with_semaphore(example):
        async with sem:
            return await process_example(editor, example)

    # Process all examples with progress bar
    tasks = [process_with_semaphore(ex) for ex in dataset]
    predictions = await tqdm_asyncio.gather(*tasks, desc="Generating predictions")

    # Count successes and failures
    successes = sum(1 for p in predictions if p['success'])
    failures = len(predictions) - successes
    logger.info(f"Completed: {successes} successes, {failures} failures")

    return predictions


def save_predictions(predictions: List[Dict], output_file: str):
    """Save predictions to a JSONL file."""
    logger.info(f"Saving {len(predictions)} predictions to {output_file}...")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

    logger.info(f"✓ Predictions saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate predictions using Gemini model.")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash",
                        help="Gemini model name to use")
    parser.add_argument("--test_file", type=str, default="datasets/scorers/grammarly_coedit/test.jsonl",
                        help="Test JSONL file")
    parser.add_argument("--output_file", type=str, default="scorers/local_scorers/human_like/data/gemini_predictions.jsonl",
                        help="Output file for predictions")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Number of concurrent API calls")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout in seconds for each API call")

    args = parser.parse_args()

    print("=" * 80)
    print("GENERATING GEMINI PREDICTIONS ON GRAMMARLY/COEDIT TEST SET")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Test file: {args.test_file}")
    print(f"  Output: {args.output_file}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Timeout: {args.timeout}s")

    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        print("\n✗ Error: GEMINI_API_KEY environment variable not set")
        print("Please set it before running this script:")
        print("  export GEMINI_API_KEY='your-api-key'")
        return

    # Load dataset
    dataset = load_test_dataset(args.test_file)

    # Generate predictions
    predictions = asyncio.run(generate_predictions_async(
        dataset,
        model_name=args.model_name,
        concurrency=args.concurrency,
        timeout=args.timeout
    ))

    # Save results
    save_predictions(predictions, args.output_file)

    # Print some examples
    print("\n" + "="*80)
    print("Sample predictions:")
    print("="*80)
    for i in range(min(3, len(predictions))):
        pred = predictions[i]
        print(f"\nExample {i+1}:")
        print(f"  Prefix: {pred['prefix']}")
        print(f"  Original: {pred['original_before_sent'][:100]}...")
        print(f"  Human edit: {pred['after_sent'][:100]}...")
        print(f"  Gemini prediction: {pred['llm_prediction'][:100]}...")
        print(f"  Success: {pred['success']}")

    print("\n✓ All done!")
    print(f"\nNext steps:")
    print(f"  1. Extract edits: python scorers/local_scorers/human_like/precompute_edits.py \\")
    print(f"       --input {args.output_file} \\")
    print(f"       --output scorers/local_scorers/human_like/data/gemini_with_edits.jsonl \\")
    print(f"       --source-field llm_prediction \\")
    print(f"       --original-field original_before_sent")


if __name__ == "__main__":
    main()
