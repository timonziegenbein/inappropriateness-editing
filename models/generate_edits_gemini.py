"""
Generate edits using Google's Gemini API to create a baseline for comparison.

This script uses Gemini to generate edits for the appropriateness editing task,
producing output compatible with evaluate_edits.py for direct comparison with
GRPO-trained models.

Usage:
    python models/generate_edits_gemini.py \
        --output_jsonl models/generated_edits/gemini_baseline.jsonl \
        --model_name gemini-2.5-flash \
        --split validation \
        --concurrency 10
"""

import os
import sys
import time
import logging
import json
import re
import asyncio
import argparse
from typing import List, Dict, Any
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import spacy
from datasets import load_dataset
from google import genai

from prompts.edit_inappropriate_text import create_llm_prompt
from ops.completion_processor import process_completion

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Spacy for sentence segmentation
nlp = spacy.load("en_core_web_sm")


class GeminiEditGenerator:
    """
    Generate text edits using Google's Gemini API.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", timeout: int = 120):
        """
        Initialize the Gemini edit generator.

        Args:
            model_name: Name of the Gemini model to use
            timeout: Timeout in seconds for API calls
        """
        self.model_name = model_name
        self.timeout = timeout
        self.system_prompt = """You are an expert text editor specializing in improving the appropriateness of argumentative text.
Your task is to identify and edit inappropriate parts while preserving the author's core message and argument structure.
Always respond with valid JSON in the exact format specified."""

        # Initialize Google GenAI client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.client = genai.Client(api_key=api_key)

    async def generate_edits(self, issue: str, sentences: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Generate edits using the Gemini API.

        Args:
            issue: The topic/issue of the argument
            sentences: Formatted sentences string (e.g., "Sentence 1: ...\nSentence 2: ...")
            max_retries: Maximum number of retries on failure

        Returns:
            Dictionary containing the completion and success status
        """
        # Create prompt using the same function as GRPO training
        prompt_text = create_llm_prompt(
            issue=issue[:-1] if isinstance(issue, str) and len(issue) > 0 else issue,
            sentences=sentences,
        )

        # Retry logic with exponential backoff
        retry_delay = 1  # Start with 1 second

        for attempt in range(max_retries):
            try:
                # Make API call with timeout
                response = await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=self.model_name,
                        contents=prompt_text,
                        config={
                            "system_instruction": self.system_prompt,
                            "top_k": 1,  # Greedy decoding (deterministic)
                            "max_output_tokens": 1024,  # Match generate_edits.py (max_new_tokens=1024)
                            "response_mime_type": "application/json",  # Request JSON output
                        }
                    ),
                    timeout=self.timeout
                )

                # Get the completion
                completion = response.text.strip()

                return {
                    "success": True,
                    "completion": completion,
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
                        "completion": "",
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
                        "completion": "",
                        "error": str(e)
                    }

        # Should never reach here, but just in case
        return {
            "success": False,
            "completion": "",
            "error": "Max retries exceeded"
        }


async def process_example(
    generator: GeminiEditGenerator,
    example: Dict[str, Any],
    idx: int
) -> Dict[str, Any]:
    """Process a single example through Gemini."""
    example_start = time.time()

    issue = example.get("issue", "")
    post_id = example.get("post_id")
    argument = example.get("post_text", "")

    # Normalize whitespace
    argument = re.sub(r"(\r\n)+|\r+|\n+|\t+", " ", argument, 0, re.MULTILINE)
    argument = re.sub(r"\s\s+", " ", argument, 0, re.MULTILINE)

    # Segment sentences
    doc = nlp(argument)
    sentences = [sent.text for sent in doc.sents]
    formatted_sentences = "\n".join([f"Sentence {i+1}: {sentence}" for i, sentence in enumerate(sentences)])

    # Generate edits with Gemini
    result = await generator.generate_edits(issue, formatted_sentences)

    completion = result['completion']
    parse_ok = False
    all_edits = []

    # Parse completion
    if result['success']:
        try:
            all_edits = process_completion(completion, sentences)
            if all_edits:
                parse_ok = True
                logger.info(f"Example {idx}: Extracted {len(all_edits)} edits")
        except Exception as e:
            logger.error(f"Example {idx}: Completion processing failed: {e}")
            all_edits = []

    # Build record in the same format as generate_edits.py
    record = {
        "post_id": post_id,
        "issue": issue,
        "argument": argument,
        "sentences": sentences,
        "completion": completion,
        "edits": all_edits,
        "metadata": {
            "parse_success": parse_ok,
            "num_edits": len(all_edits),
            "generation_time": time.time() - example_start,
            "round": 1,
            "is_parse_diff": False,
            "model": generator.model_name,
            "api_success": result['success'],
            "api_error": result.get('error')
        }
    }

    return record


async def generate_all_edits_async(
    dataset: List[Dict],
    model_name: str,
    concurrency: int = 10,
    timeout: int = 120
) -> List[Dict]:
    """Generate edits for all examples with controlled concurrency."""
    logger.info(f"Generating edits for {len(dataset)} examples with concurrency={concurrency}...")

    generator = GeminiEditGenerator(model_name=model_name, timeout=timeout)

    # Create semaphore for concurrency control
    sem = asyncio.Semaphore(concurrency)

    async def process_with_semaphore(example, idx):
        async with sem:
            return await process_example(generator, example, idx)

    # Process all examples with progress bar
    tasks = [process_with_semaphore(ex, idx) for idx, ex in enumerate(dataset)]
    records = await tqdm_asyncio.gather(*tasks, desc="Generating edits")

    # Count statistics
    num_parse_success = sum(1 for r in records if r['metadata']['parse_success'])
    total_edits = sum(r['metadata']['num_edits'] for r in records)
    api_successes = sum(1 for r in records if r['metadata']['api_success'])

    logger.info(f"Completed: {api_successes}/{len(dataset)} API successes, {num_parse_success}/{len(dataset)} parsed successfully")
    logger.info(f"Total edits generated: {total_edits}")

    return records


def save_records(records: List[Dict], output_file: str):
    """Save records to a JSONL file."""
    logger.info(f"Saving {len(records)} records to {output_file}...")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"✓ Records saved to {output_file}")


def main(
    output_jsonl: str,
    model_name: str = "gemini-2.5-flash",
    split: str = "validation",
    concurrency: int = 10,
    timeout: int = 120,
    max_examples: int = None
):
    """Generate edits using Gemini and save them to a JSONL file."""
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    logger.info(f"Starting Gemini edit generation on {split} split. Output file: {output_jsonl}")

    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable not set")
        print("\n✗ Error: GEMINI_API_KEY environment variable not set")
        print("Please set it before running this script:")
        print("  export GEMINI_API_KEY='your-api-key'")
        return

    # Load dataset
    eval_dataset = load_dataset("", split=split)
    eval_dataset = eval_dataset.filter(lambda x: float(x.get("Inappropriateness", 0.0)) >= 0.5)

    num_examples = len(eval_dataset)
    logger.info(f"Loaded {split} dataset: {num_examples} examples")

    # Convert to list and optionally limit
    dataset_list = list(eval_dataset)
    if max_examples:
        dataset_list = dataset_list[:max_examples]
        logger.info(f"Limited to first {max_examples} examples for testing")

    # Generate edits
    start_all = time.time()
    records = asyncio.run(generate_all_edits_async(
        dataset_list,
        model_name=model_name,
        concurrency=concurrency,
        timeout=timeout
    ))

    # Save results
    save_records(records, output_jsonl)

    logger.info(
        f"Finished. Time={time.time() - start_all:.1f}s | "
        f"examples={len(records)} | "
        f"parse_ok={sum(1 for r in records if r['metadata']['parse_success'])}/{len(records)} | "
        f"total_edits={sum(r['metadata']['num_edits'] for r in records)}"
    )
    logger.info(f"Wrote {output_jsonl}")

    print("\n" + "="*80)
    print("✓ Generation complete!")
    print("="*80)
    print(f"\nNext step: Evaluate the edits")
    print(f"  python models/evaluate_edits.py \\")
    print(f"    --input_jsonl {output_jsonl} \\")
    print(f"    --output_jsonl models/predictions/gemini_baseline.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate edits using Gemini API as a baseline.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash",
                        help="Gemini model name to use (default: gemini-2.5-flash)")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"],
                        help="Dataset split to use (default: validation)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Number of concurrent API calls (default: 10)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout in seconds for each API call (default: 120)")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to process (for testing)")

    args = parser.parse_args()

    print("=" * 80)
    print("GENERATING EDITS USING GEMINI API (BASELINE)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Split: {args.split}")
    print(f"  Output: {args.output_jsonl}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Timeout: {args.timeout}s")
    if args.max_examples:
        print(f"  Max examples: {args.max_examples}")
    print()

    main(
        args.output_jsonl,
        args.model_name,
        args.split,
        args.concurrency,
        args.timeout,
        args.max_examples
    )
