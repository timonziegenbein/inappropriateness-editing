"""
Truncate completions to a maximum token length for fair comparison.

This script reads generated edits JSONL files and truncates the completions
to a specified token limit, then re-parses the edits.

Usage:
    python models/truncate_completions.py \
        --input_jsonl models/generated_edits/gpt5_mini_baseline.jsonl \
        --output_jsonl models/generated_edits/gpt5_mini_baseline_1024.jsonl \
        --max_tokens 1024
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tiktoken
from ops.completion_processor import process_completion

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """
    Truncate text to a maximum number of tokens.

    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens
        model: Model name for tokenizer (default: gpt-4)

    Returns:
        Truncated text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base encoding (used by GPT-4 and newer models)
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    # Truncate to max_tokens
    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoding.decode(truncated_tokens)

    logger.info(f"Truncated completion from {len(tokens)} to {max_tokens} tokens")

    return truncated_text


def process_record(record: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    """
    Process a single record by truncating the completion and re-parsing edits.

    Args:
        record: The record to process
        max_tokens: Maximum number of tokens for completion

    Returns:
        Updated record with truncated completion and re-parsed edits
    """
    completion = record.get("completion", "")
    sentences = record.get("sentences", [])
    model = record.get("metadata", {}).get("model", "gpt-4")

    # Truncate completion
    truncated_completion = truncate_to_tokens(completion, max_tokens, model)

    # Re-parse edits from truncated completion
    parse_ok = False
    all_edits = []

    if truncated_completion:
        try:
            all_edits = process_completion(truncated_completion, sentences)
            if all_edits:
                parse_ok = True
        except Exception as e:
            logger.warning(f"Failed to parse truncated completion for post_id {record.get('post_id')}: {e}")

    # Update record
    updated_record = record.copy()
    updated_record["completion"] = truncated_completion
    updated_record["edits"] = all_edits
    updated_record["metadata"] = record.get("metadata", {}).copy()
    updated_record["metadata"]["parse_success"] = parse_ok
    updated_record["metadata"]["num_edits"] = len(all_edits)
    updated_record["metadata"]["truncated_to_tokens"] = max_tokens
    updated_record["metadata"]["was_truncated"] = len(completion) > len(truncated_completion)

    return updated_record


def main(input_jsonl: str, output_jsonl: str, max_tokens: int):
    """
    Truncate completions in a JSONL file and save the results.

    Args:
        input_jsonl: Path to input JSONL file
        output_jsonl: Path to output JSONL file
        max_tokens: Maximum number of tokens for completions
    """
    logger.info(f"Processing {input_jsonl}")
    logger.info(f"Truncating completions to {max_tokens} tokens")

    # Read input file
    records = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))

    logger.info(f"Loaded {len(records)} records")

    # Process records
    updated_records = []
    num_truncated = 0
    num_parse_success = 0

    for i, record in enumerate(records):
        updated_record = process_record(record, max_tokens)
        updated_records.append(updated_record)

        if updated_record["metadata"]["was_truncated"]:
            num_truncated += 1

        if updated_record["metadata"]["parse_success"]:
            num_parse_success += 1

        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(records)} records")

    # Save output file
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for record in updated_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"✓ Saved {len(updated_records)} records to {output_jsonl}")
    logger.info(f"  Truncated: {num_truncated}/{len(records)}")
    logger.info(f"  Parse success: {num_parse_success}/{len(records)}")
    logger.info(f"  Total edits: {sum(r['metadata']['num_edits'] for r in updated_records)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Truncate completions to a maximum token length"
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens for completions (default: 1024)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TRUNCATING COMPLETIONS")
    print("=" * 80)
    print(f"\nInput: {args.input_jsonl}")
    print(f"Output: {args.output_jsonl}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    main(args.input_jsonl, args.output_jsonl, args.max_tokens)
