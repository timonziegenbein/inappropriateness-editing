"""
Generate edits from a model and save them to a JSONL file.

This script handles the costly edit generation step (model inference) and saves
the raw edits to a file. The edits can then be evaluated with different scorer
configurations using evaluate_edits.py without regenerating them.

Usage:
    # Generate edits from a trained model
    python models/generate_edits.py --checkpoint_root <checkpoint_path> --output_jsonl <output_file.jsonl>

    # Generate edits from base model
    python models/generate_edits.py --use_base_model_only --output_jsonl <output_file.jsonl>

    # Parse existing diffs (e.g., from human edits)
    python models/generate_edits.py --parse_diff --model_name rewrite_40a_60ss --output_jsonl <output_file.jsonl>

    # Multi-round editing: apply perfect edits and generate new ones
    python models/generate_edits.py --checkpoint_root <checkpoint_path> --output_jsonl <output_file.jsonl> \
        --input_evaluated_jsonl <evaluated_file.jsonl> --round 2
"""

import os
import sys
import time
import logging
import re
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import spacy
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer

from prompts.edit_inappropriate_text import create_llm_prompt
from ops.completion_processor import process_completion
from ops.latexdiff_parser import DirectLatexdiffParser, fuzzy_post_process_edits
from ops.edit_applier import apply_edits_to_argument

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_BASE = "meta-llama/Llama-3.1-8B-Instruct"

# Spacy for sentence segmentation
nlp = spacy.load("en_core_web_sm")


def load_generation_model(checkpoint_root: str, use_base_model_only: bool = False) -> tuple:
    """Load the model for generating edits."""
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.1-8B-Instruct', padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        peft_type="LORA",
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    training_args = GRPOConfig(
        output_dir="./temp_output",
        per_device_train_batch_size=2,
        log_completions=True,
        max_completion_length=1024,
        max_prompt_length=2048,
        scale_rewards=False,
        gradient_accumulation_steps=8,
        optim="paged_adamw_8bit",
        bf16=True,
        label_names=[],
        use_vllm=False,
        vllm_mode="colocate",
        loss_type="dr_grpo",
        mask_truncated_completions=True,
        reward_weights=[0.5, 0.5],
        use_cpu=not torch.cuda.is_available(),
    )

    empty_reward = lambda *args, **kwargs: 0.0

    trainer = GRPOTrainer(
        model=MODEL_BASE,
        reward_funcs=[empty_reward, empty_reward],
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    if not use_base_model_only:
        trainer.model.load_adapter(checkpoint_root, adapter_name="default")

    return trainer, tokenizer


def load_evaluated_data(input_evaluated_jsonl: str) -> Dict[str, Dict]:
    """Load evaluated data and create a mapping from post_id to data."""
    logger.info(f"Loading evaluated data from {input_evaluated_jsonl}")
    evaluated_data = {}
    with open(input_evaluated_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            post_id = data.get('post_id')
            if post_id:
                evaluated_data[post_id] = data
    logger.info(f"Loaded {len(evaluated_data)} evaluated examples")
    return evaluated_data


def main(
    checkpoint_root: str,
    output_jsonl: str,
    use_base_model_only: bool = False,
    parse_diff: bool = False,
    model_name: str = None,
    split: str = "validation",
    input_evaluated_jsonl: str = None,
    round_number: int = 1
):
    """Generate edits and save them to a JSONL file."""
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    logger.info(f"Starting edit generation on {split} split. Output file: {output_jsonl}")

    # Multi-round editing setup
    evaluated_data = None
    if input_evaluated_jsonl:
        logger.info(f"Multi-round editing enabled (Round {round_number})")
        evaluated_data = load_evaluated_data(input_evaluated_jsonl)
    else:
        logger.info(f"Single-round editing (Round {round_number})")

    if not parse_diff:
        model_load_start = time.time()
        trainer, tokenizer = load_generation_model(checkpoint_root, use_base_model_only)
        logger.info(f"Loaded generation model and tokenizer in {time.time() - model_load_start:.1f}s")

    # Load dataset
    eval_dataset = load_dataset("timonziegenbein/appropriateness-corpus", split=split)
    eval_dataset = eval_dataset.filter(lambda x: float(x.get("Inappropriateness", 0.0)) >= 0.5)

    if parse_diff:
        # Load study data for diff parsing
        df1 = pd.read_csv('/mnt/home/tziegenb/appropriateness-feedback/src/annotation-interface/appropriateness-study-abs/data/study_edits_part1.csv')
        df2 = pd.read_csv('/mnt/home/tziegenb/appropriateness-feedback/src/annotation-interface/appropriateness-study-abs/data/study_edits_part2.csv')
        df = pd.concat([df1, df2], ignore_index=True)
        eval_ids = set(eval_dataset['post_id'])
        df = df[df['id'].isin(eval_ids)].sort_values(by=['id']).reset_index(drop=True)
        eval_dataset = Dataset.from_pandas(df)

    num_examples = len(eval_dataset)
    logger.info(f"Loaded {split} dataset: {num_examples} examples")

    # Statistics
    num_parse_success = 0
    total_edits = 0

    start_all = time.time()
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for idx, example in enumerate(eval_dataset):
            example_start = time.time()
            issue = example.get("issue", "")
            post_id = example.get("post_id" if not parse_diff else "id")

            if parse_diff:
                argument = example.get("source", "")
                argument = re.sub(r"(\r\n)+|\r+|\n+|\t+", " ", argument, 0, re.MULTILINE)
                argument = re.sub(r"\s\s+", " ", argument, 0, re.MULTILINE)
                rewritten_argument = example.get(model_name, "")
                rewritten_argument = re.sub(r"(\r\n)+|\r+|\n+|\t+", " ", rewritten_argument, 0, re.MULTILINE)
                rewritten_argument = re.sub(r"\s\s+", " ", rewritten_argument, 0, re.MULTILINE)
                completion = rewritten_argument
            else:
                # Model generation mode
                original_argument = example.get("post_text", "")
                original_argument = re.sub(r"(\r\n)+|\r+|\n+|\t+", " ", original_argument, 0, re.MULTILINE)
                original_argument = re.sub(r"\s\s+", " ", original_argument, 0, re.MULTILINE)

                # Multi-round editing: start with refined argument from previous round
                num_previous_perfect = 0
                if evaluated_data and post_id in evaluated_data:
                    previous_round_data = evaluated_data[post_id]
                    argument = previous_round_data.get("argument_after_edits", original_argument)
                    previous_round_edits = previous_round_data.get("edits", [])
                    num_previous_perfect = sum(1 for e in previous_round_edits if e.get("rewards", {}).get("perfect") == 1.0)
                    logger.info(f"Example {idx}: Multi-round - starting from refined argument (previous round had {num_previous_perfect} perfect edits)")
                else:
                    # First round: use original argument
                    argument = original_argument

            # Segment sentences
            doc = nlp(argument)
            sentences = [sent.text for sent in doc.sents]
            formatted_sentences = "\n".join([f"Sentence {i+1}: {sentence}" for i, sentence in enumerate(sentences)])

            # For parse_diff mode, also segment the rewritten text
            if parse_diff:
                rewritten_doc = nlp(rewritten_argument)
                rewritten_sentences = [sent.text for sent in rewritten_doc.sents]

            if not parse_diff:
                prompt_text = create_llm_prompt(
                    issue=issue[:-1] if isinstance(issue, str) and len(issue) > 0 else issue,
                    sentences=formatted_sentences,
                )
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                logger.info(f"Model input: {prompt}")

                inputs = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)

                prediction_output = trainer.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    #do_sample=True,
                    #temperature=0.8,
                    #top_p=0.9,
                )

                completion = tokenizer.decode(
                    prediction_output[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                logger.info(f"Model completion: {completion}")

            # Parse completion
            parse_ok = False
            all_edits = []

            try:
                if parse_diff:
                    if argument.strip() == rewritten_argument.strip():
                        logger.warning(f"Example {idx}: Source and rewritten text are identical, skipping")
                        all_edits = []
                    else:
                        logger.info(f"Example {idx}: Parsing latex diff sentence by sentence")
                        parser = DirectLatexdiffParser()

                        # Check if sentence counts match
                        if len(sentences) != len(rewritten_sentences):
                            logger.warning(f"Example {idx}: Sentence count mismatch (source: {len(sentences)}, rewritten: {len(rewritten_sentences)}). Processing with alignment.")

                        # Process each sentence pair
                        total_edit_actions = 0
                        for sent_idx in range(min(len(sentences), len(rewritten_sentences))):
                            source_sent = sentences[sent_idx]
                            rewritten_sent = rewritten_sentences[sent_idx]

                            if source_sent.strip() == rewritten_sent.strip():
                                logger.debug(f"Example {idx}, Sentence {sent_idx + 1}: No changes, skipping")
                                continue

                            logger.debug(f"Example {idx}, Sentence {sent_idx + 1}: Parsing diff")
                            parsed_example = parser.parse_latex_diff(source_sent, rewritten_sent, "./temp_output")
                            num_actions = len(parsed_example.get('edit_actions', []))
                            total_edit_actions += num_actions
                            logger.debug(f"LaTeX diff returned {num_actions} edit actions for sentence {sent_idx + 1}")

                            parsed_example['before_revision'] = source_sent
                            data, _ = fuzzy_post_process_edits([parsed_example])
                            sentence_edits = data.get("edits", [])

                            # Add sentence_id to each edit (1-indexed to match the format)
                            for edit in sentence_edits:
                                edit['sentence_id'] = sent_idx + 1

                            all_edits.extend(sentence_edits)
                            logger.debug(f"After fuzzy post-processing: {len(sentence_edits)} edits extracted for sentence {sent_idx + 1}")

                        logger.info(f"LaTeX diff returned {total_edit_actions} total edit actions across {min(len(sentences), len(rewritten_sentences))} sentences")
                        logger.info(f"After fuzzy post-processing: {len(all_edits)} total edits extracted")
                else:
                    all_edits = process_completion(completion, sentences)

                if all_edits:
                    parse_ok = True
                    logger.info(f"Extracted {len(all_edits)} edits")

            except Exception as e:
                import traceback
                logger.error(f"Completion processing failed on example {idx}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                all_edits = []

            if parse_ok:
                num_parse_success += 1

            total_edits += len(all_edits)

            # Save record with raw edits
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
                    "round": round_number,
                    "is_parse_diff": parse_diff,
                }
            }

            # Add multi-round metadata if applicable
            if evaluated_data and post_id in evaluated_data and not parse_diff:
                record["metadata"]["is_multi_round"] = True
                record["metadata"]["previous_round_perfect_edits"] = num_previous_perfect
                # Store original argument for reference
                record["original_argument"] = original_argument
            else:
                record["metadata"]["is_multi_round"] = False

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Progress logging
            if idx < 3 or (idx + 1) % 50 == 0 or (idx + 1) == num_examples:
                elapsed = time.time() - example_start
                logger.info(
                    f"[{idx + 1}/{num_examples}] gen={elapsed:.1f}s edits={len(all_edits)} | issue='{issue[:100]}'"
                )

    logger.info(
        f"Finished. Time={time.time() - start_all:.1f}s | parse_ok={num_parse_success}/{num_examples} | total_edits={total_edits}"
    )
    logger.info(f"Wrote {output_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate edits from a model and save them to JSONL.")
    parser.add_argument("--checkpoint_root", type=str, help="Path to the checkpoint directory.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--use_base_model_only", action="store_true", help="Use the base model without LoRA.")
    parser.add_argument("--parse_diff", action="store_true", help="Parse diffs instead of generating edits.")
    parser.add_argument("--model_name", type=str, help="The name of the model to evaluate from the dataframe (for parse_diff mode).")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"], help="Dataset split to use (default: validation)")
    parser.add_argument("--input_evaluated_jsonl", type=str, help="Path to evaluated JSONL file from previous round (for multi-round editing).")
    parser.add_argument("--round", type=int, default=1, help="Round number for multi-round editing (default: 1)")
    args = parser.parse_args()

    if not args.use_base_model_only and not args.checkpoint_root and not args.parse_diff:
        parser.error("--checkpoint_root is required unless --use_base_model_only or --parse_diff is specified.")

    if args.input_evaluated_jsonl and args.parse_diff:
        parser.error("--input_evaluated_jsonl cannot be used with --parse_diff (multi-round editing only applies to model generation).")

    main(
        args.checkpoint_root,
        args.output_jsonl,
        args.use_base_model_only,
        args.parse_diff,
        args.model_name,
        args.split,
        args.input_evaluated_jsonl,
        args.round
    )
