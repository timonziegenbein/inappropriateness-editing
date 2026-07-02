"""
Pre-compute edits for all predictions using latex diff in parallel.

This script parses edits from all prediction files and saves them to JSONL files
for faster experimentation with different scorer settings.
"""

import sys
import os
from pathlib import Path

# Go up to project root: human_like -> local_scorers -> scorers -> appropriateness-edit
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import tempfile
import shutil

from ops.latexdiff_parser import DirectLatexdiffParser, fuzzy_post_process_edits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_edits_from_texts(original_text: str, edited_text: str) -> List[Dict[str, Any]]:
    """Parse edits using latex diff."""
    temp_dir = tempfile.mkdtemp(prefix="latex_diff_")

    try:
        if original_text.strip() == edited_text.strip():
            return []

        parser = DirectLatexdiffParser()
        parsed_example = parser.parse_latex_diff(original_text, edited_text, temp_dir)

        if not parsed_example or not parsed_example.get('edit_actions'):
            return []

        parsed_example['before_revision'] = original_text
        data, _ = fuzzy_post_process_edits([parsed_example])
        edits = data.get("edits", [])

        return edits

    except Exception as e:
        logger.debug(f"Error parsing edits: {e}")
        return []
    finally:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.debug(f"Failed to cleanup temp directory: {e}")


def remove_prefix(text: str) -> str:
    """
    Remove prefix from text by splitting on ": " and taking everything after first occurrence.

    Returns: cleaned text without prefix
    """
    if ':' in text:
        parts = text.split(':', 1)
        if len(parts) == 2:
            return parts[1].strip()

    # No colon found, return full text
    return text.strip()


def process_single_prediction(pred: Dict[str, Any], source_field: str, original_field: str,
                              remove_prefix_from_original: bool = False) -> Dict[str, Any]:
    """
    Process a single prediction to extract edits.

    Args:
        pred: Prediction dictionary
        source_field: Field containing the edited/predicted text
        original_field: Field containing the original text
        remove_prefix_from_original: If True, remove prefix from original text before comparing

    Returns: Original prediction with added 'parsed_edits' field
    """
    original = pred.get(original_field, "")
    edited = pred.get(source_field, "")

    result = pred.copy()

    if not original or not edited:
        result['parsed_edits'] = []
        return result

    # Remove prefix from original if requested
    if remove_prefix_from_original:
        original = remove_prefix(original)

    # Check if there's any edit
    if original.strip() == edited.strip():
        result['parsed_edits'] = []
        return result

    # Parse edits
    edits = parse_edits_from_texts(original, edited)
    result['parsed_edits'] = edits

    # Update original_before_sent to be without prefix for consistency
    if remove_prefix_from_original:
        result['original_before_sent'] = original

    return result


def load_predictions(file_path: str) -> List[Dict[str, Any]]:
    """Load predictions from a JSONL file."""
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def save_predictions(predictions: List[Dict[str, Any]], output_path: str):
    """Save predictions to a JSONL file."""
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    logger.info(f"✓ Saved {len(predictions)} predictions to {output_path}")


def precompute_edits(input_file: str, output_file: str, source_field: str,
                     original_field: str = "original_before_sent",
                     remove_prefix: bool = False,
                     num_workers: int = 4, max_examples: Optional[int] = None):
    """Pre-compute edits for a prediction file."""

    logger.info(f"\nProcessing {input_file}")
    logger.info(f"Source field: {source_field}")
    logger.info(f"Remove prefix: {remove_prefix}")

    # Load predictions
    predictions = load_predictions(input_file)
    logger.info(f"Loaded {len(predictions)} predictions")

    if max_examples and len(predictions) > max_examples:
        predictions = predictions[:max_examples]
        logger.info(f"Limited to {max_examples} examples")

    # Process in parallel
    process_func = partial(process_single_prediction,
                          source_field=source_field,
                          original_field=original_field,
                          remove_prefix_from_original=remove_prefix)

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, predictions, chunksize=50),
            total=len(predictions),
            desc=f"Parsing edits"
        ))

    # Calculate statistics
    num_with_edits = sum(1 for r in results if r.get('parsed_edits'))
    num_edits = sum(len(r.get('parsed_edits', [])) for r in results)

    logger.info(f"  Examples with edits: {num_with_edits}/{len(results)} ({num_with_edits/len(results)*100:.1f}%)")
    logger.info(f"  Total edits parsed: {num_edits}")
    logger.info(f"  Avg edits per example: {num_edits/len(results):.2f}")

    # Save results
    save_predictions(results, output_file)


def main():
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Pre-compute edits for all predictions")
    parser.add_argument("--human-file", type=str,
                       default=str(script_dir / "data" / "human_edits.jsonl"))
    parser.add_argument("--coedit-file", type=str,
                       default=str(script_dir / "data" / "coedit_predictions.jsonl"))
    parser.add_argument("--llama-file", type=str,
                       default=str(script_dir / "data" / "llama_predictions.jsonl"))
    parser.add_argument("--gemini-file", type=str,
                       default=str(script_dir / "data" / "gemini_predictions.jsonl"))
    parser.add_argument("--output-dir", type=str, default=str(script_dir / "data"),
                       help="Directory to save pre-computed edits")
    parser.add_argument("--max-examples", type=int, default=None,
                       help="Maximum number of examples to process per source")
    parser.add_argument("--num-workers", type=int, default=max(1, cpu_count() - 1),
                       help="Number of parallel workers")
    parser.add_argument("--remove-prefix", action="store_true",
                       help="Remove prefix from original_before_sent before comparing (for grammarly/coedit predictions)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("PRE-COMPUTING EDITS FOR ALL PREDICTIONS")
    logger.info("="*80)
    logger.info(f"Using {args.num_workers} parallel workers")
    logger.info(f"Output directory: {output_dir}")

    # Process each source
    sources = []
    if os.path.exists(args.human_file):
        sources.append(("human", args.human_file, "after_sent"))
    if os.path.exists(args.coedit_file):
        sources.append(("coedit", args.coedit_file, "coedit_prediction"))
    if os.path.exists(args.llama_file):
        sources.append(("llama", args.llama_file, "llama_prediction"))
    if os.path.exists(args.gemini_file):
        sources.append(("gemini", args.gemini_file, "llm_prediction"))

    for source_name, file_path, field_name in sources:
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING {source_name.upper()}")
        logger.info(f"{'='*80}")

        output_file = output_dir / f"{source_name}_with_edits.jsonl"

        precompute_edits(
            input_file=file_path,
            output_file=str(output_file),
            source_field=field_name,
            remove_prefix=args.remove_prefix,
            num_workers=args.num_workers,
            max_examples=args.max_examples
        )

    logger.info(f"\n{'='*80}")
    logger.info("✓ ALL EDITS PRE-COMPUTED SUCCESSFULLY!")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
