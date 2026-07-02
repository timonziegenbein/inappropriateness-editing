"""
Generate predictions for the IteraTeR test dataset using an LLM (Gemini).

This script uses Google's Gemini API to generate text edits based on task prefixes,
similar to how the fluency scorer uses LLMs for evaluation.
"""

import os
import json
import argparse
import asyncio
from typing import List, Dict, Any
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
from google import genai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMTextEditor:
    """
    LLM-based text editor using Google's Gemini API.
    Uses structured prompting to perform text editing tasks.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash-exp", timeout: int = 60):
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
                        "edited_text": sentence,  # Return original on failure
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
                        "edited_text": sentence,  # Return original on failure
                        "error": str(e)
                    }


async def generate_predictions_async(
    dataset,
    editor: LLMTextEditor,
    max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    """
    Generate predictions for the entire dataset asynchronously.

    Args:
        dataset: HuggingFace dataset with examples
        editor: LLMTextEditor instance
        max_concurrent: Maximum number of concurrent API calls

    Returns:
        List of prediction dictionaries
    """
    logger.info(f"Generating predictions for {len(dataset)} examples with max {max_concurrent} concurrent requests...")

    predictions = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_example(example):
        """Process a single example with rate limiting"""
        async with semaphore:
            result = await editor.edit_text(
                before_sent=example['before_sent']
            )

            return {
                'original_intent': example['original_intent'],
                'prefix': example['prefix'],
                'original_before_sent': example['original_before_sent'],
                'before_sent': example['before_sent'],
                'after_sent': example['after_sent'],
                'llm_prediction': result['edited_text'],
                'success': result['success'],
                'error': result.get('error')
            }

    # Create tasks for all examples
    tasks = [process_example(example) for example in dataset]

    # Process with progress bar
    results = []
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Generating predictions"):
        result = await coro
        results.append(result)

    return results


def save_predictions(predictions: List[Dict[str, Any]], output_file: str):
    """Save predictions to a JSONL file."""
    logger.info(f"Saving {len(predictions)} predictions to {output_file}...")

    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

    # Count successes and failures
    successes = sum(1 for p in predictions if p['success'])
    failures = len(predictions) - successes

    logger.info(f"✓ Predictions saved to {output_file}")
    logger.info(f"  Successful: {successes} ({successes/len(predictions)*100:.1f}%)")
    logger.info(f"  Failed: {failures} ({failures/len(predictions)*100:.1f}%)")


async def main_async(args):
    """Main async function"""
    logger.info(f"Configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Dataset: {args.dataset_name}")
    logger.info(f"  Output: {args.output_file}")
    logger.info(f"  Max concurrent requests: {args.max_concurrent}")
    logger.info(f"  Timeout: {args.timeout}s")

    # Load dataset
    logger.info(f"\nLoading dataset {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, split="train")
    logger.info(f"Loaded {len(dataset)} examples")

    # Limit examples if specified
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} examples")

    # Initialize LLM editor
    editor = LLMTextEditor(model_name=args.model_name, timeout=args.timeout)
    logger.info(f"Initialized LLM editor with model: {args.model_name}")

    # Generate predictions
    predictions = await generate_predictions_async(
        dataset,
        editor,
        max_concurrent=args.max_concurrent
    )

    # Save results
    save_predictions(predictions, args.output_file)

    # Print some examples
    logger.info("\n" + "="*80)
    logger.info("Sample predictions:")
    logger.info("="*80)
    for i in range(min(3, len(predictions))):
        pred = predictions[i]
        logger.info(f"\nExample {i+1}:")
        logger.info(f"  Intent: {pred['original_intent']}")
        logger.info(f"  Prefix: {pred['prefix']}")
        logger.info(f"  Original: {pred['original_before_sent'][:100]}...")
        logger.info(f"  Human edit: {pred['after_sent'][:100]}...")
        logger.info(f"  LLM prediction: {pred['llm_prediction'][:100]}...")
        logger.info(f"  Success: {pred['success']}")

    logger.info("\n✓ All done!")


def main():
    parser = argparse.ArgumentParser(description="Generate predictions using Gemini LLM")
    parser.add_argument("--model_name", type=str, default="gemini-2.0-flash-exp",
                        help="Gemini model name to use")
    parser.add_argument("--dataset_name", type=str, default="timonziegenbein/iterater-test-with-prefixes",
                        help="HuggingFace dataset name")
    parser.add_argument("--output_file", type=str, default="data/gemini_predictions.jsonl",
                        help="Output file for predictions")
    parser.add_argument("--max_concurrent", type=int, default=10,
                        help="Maximum number of concurrent API requests")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout in seconds for API calls")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to process (for testing)")

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable not set!")
        logger.error("Please set it with: export GEMINI_API_KEY='your-api-key'")
        return

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
