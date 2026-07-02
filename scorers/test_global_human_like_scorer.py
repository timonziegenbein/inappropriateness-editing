"""
Interactive test script for the Global Human-Like Scorer.

This script allows you to test the global human-like scorer with example arguments
and edits, showing all intermediate processing steps.

Usage:
    # Interactive mode: select file from models/generated_edits/ and cycle through examples
    python scorers/test_global_human_like_scorer.py

    # Directly specify a file to cycle through
    python scorers/test_global_human_like_scorer.py --input_jsonl models/generated_edits/grpo_global_sentence_v11.jsonl

    # Start from a specific example index
    python scorers/test_global_human_like_scorer.py --input_jsonl <path> --example_index 5

    # Limit number of examples to show
    python scorers/test_global_human_like_scorer.py --input_jsonl <path> --max_examples 10

    # Use predefined example
    python scorers/test_global_human_like_scorer.py --predefined 0

    # Use custom model path and threshold
    python scorers/test_global_human_like_scorer.py --input_jsonl <path> --model_path <path> --threshold 2.5
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from scorers.global_scorers.human_like.global_human_like_scorer import GlobalHumanLikeScorer

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_section(title: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}\n")


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*60}{Colors.ENDC}")


def visualize_tokenization(scorer: GlobalHumanLikeScorer, text: str):
    """Visualize how the text is tokenized."""
    print_subsection("Step 1: Tokenization")

    encoding = scorer.tokenizer(text, return_offsets_mapping=True)
    tokens = scorer.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offsets = encoding['offset_mapping']

    print(f"Original text ({len(text)} chars):")
    print(f"{Colors.BLUE}{text}{Colors.ENDC}\n")

    print(f"Tokens ({len(tokens)} total):")
    for i, (token, offset) in enumerate(zip(tokens, offsets)):
        start, end = offset
        original_text = text[start:end] if start < len(text) and end <= len(text) else ""
        print(f"  [{i:3d}] {token:20s} -> chars[{start:4d}:{end:4d}] = '{original_text}'")

    return tokens, offsets


def visualize_edit_mapping(scorer: GlobalHumanLikeScorer, text: str, tokens: List[str],
                          offsets: List[tuple], edit: Dict, edit_idx: int):
    """Visualize how an edit maps to tokens."""
    print_subsection(f"Edit {edit_idx + 1} Mapping")

    inappropriate_part = edit.get('inappropriate_part', '')
    rewritten_part = edit.get('rewritten_part', '')
    reason = edit.get('reason', 'N/A')

    print(f"Reason: {Colors.YELLOW}{reason}{Colors.ENDC}")
    print(f"Inappropriate part: {Colors.RED}{inappropriate_part}{Colors.ENDC}")
    print(f"Rewritten part: {Colors.GREEN}{rewritten_part}{Colors.ENDC}\n")

    # Find in text
    start_char = text.find(inappropriate_part)
    if start_char == -1:
        print(f"{Colors.RED}ERROR: Could not find inappropriate part in text!{Colors.ENDC}")
        return None, None, None

    end_char = start_char + len(inappropriate_part)
    print(f"Character range: [{start_char}:{end_char}]\n")

    # Find token indices
    token_start_index = -1
    token_end_index = -1
    affected_tokens = []

    for i, offset in enumerate(offsets):
        token_start, token_end = offset
        if start_char < token_end and end_char > token_start:
            if token_start_index == -1:
                token_start_index = i
            token_end_index = i
            affected_tokens.append((i, tokens[i], offset))

    if token_start_index == -1:
        print(f"{Colors.RED}ERROR: Could not map to tokens!{Colors.ENDC}")
        return None, None, None

    print(f"Affected tokens: [{token_start_index}:{token_end_index+1}]")
    print("Token details:")
    for i, token, offset in affected_tokens:
        print(f"  [{i:3d}] {token}")

    # Tokenize the rewritten part
    after_edit_tokens = scorer.tokenizer.tokenize(rewritten_part) if rewritten_part else []
    print(f"\nRewritten part tokens ({len(after_edit_tokens)}):")
    for i, token in enumerate(after_edit_tokens):
        print(f"  [{i:3d}] {token}")

    return token_start_index, token_end_index, after_edit_tokens


def visualize_edit_sequence(scorer: GlobalHumanLikeScorer, text: str, edits: List[Dict]):
    """Visualize the complete edit sequence generation."""
    print_subsection("Step 2: Generate Document-Level Edit Sequence")

    encoding = scorer.tokenizer(text, return_offsets_mapping=True)
    tokens = scorer.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offsets = encoding['offset_mapping']

    # Show edit mappings
    for i, edit in enumerate(edits):
        visualize_edit_mapping(scorer, text, tokens, offsets, edit, i)

    # Generate the full sequence
    sequence = scorer._generate_document_edit_sequence(text, edits)

    print_subsection("Generated Edit Operation Sequence")
    print(f"Sequence length: {len(sequence)}")

    # Count operation types
    operation_counts = {}
    for op in sequence:
        operation_counts[op] = operation_counts.get(op, 0) + 1

    print("\nOperation distribution:")
    for op, count in sorted(operation_counts.items(), key=lambda x: -x[1]):
        percentage = (count / len(sequence)) * 100
        color = Colors.BLUE if op == 'keep' else Colors.RED if op == 'del' else Colors.GREEN if op == 'add' else Colors.YELLOW
        print(f"  {color}{op:15s}{Colors.ENDC}: {count:4d} ({percentage:5.1f}%)")

    # Show sequence in chunks
    print("\nEdit sequence (showing first 200 tokens):")
    chunk_size = 50
    for i in range(0, min(200, len(sequence)), chunk_size):
        chunk = sequence[i:i+chunk_size]
        colored_chunk = []
        for op in chunk:
            if op == 'keep':
                colored_chunk.append(f"{Colors.BLUE}K{Colors.ENDC}")
            elif op == 'del':
                colored_chunk.append(f"{Colors.RED}D{Colors.ENDC}")
            elif op == 'add':
                colored_chunk.append(f"{Colors.GREEN}A{Colors.ENDC}")
            elif op == 'replace':
                colored_chunk.append(f"{Colors.YELLOW}R{Colors.ENDC}")
            elif op == 'keep-in-edit':
                colored_chunk.append(f"{Colors.CYAN}E{Colors.ENDC}")
            else:
                colored_chunk.append("?")
        print(f"  [{i:4d}-{i+len(chunk):4d}] {''.join(colored_chunk)}")

    if len(sequence) > 200:
        print(f"\n  ... ({len(sequence) - 200} more tokens)")

    print(f"\n{Colors.CYAN}Legend: K=keep, D=del, A=add, R=replace, E=keep-in-edit{Colors.ENDC}")

    return sequence


def visualize_perplexity_calculation(scorer: GlobalHumanLikeScorer, sequence: List[str]):
    """Visualize the perplexity calculation."""
    print_subsection("Step 3: Calculate Perplexity")

    # Show vocab mapping
    print("Vocabulary mapping:")
    from scorers.global_scorers.human_like.global_human_like_scorer import hl_vocab
    for token, idx in sorted(hl_vocab.items(), key=lambda x: x[1]):
        print(f"  {token:15s} -> {idx}")

    # Convert to integers
    sequence_as_int = [hl_vocab.get(token, 0) for token in sequence]
    print(f"\nSequence as integers (first 50):")
    print(f"  {sequence_as_int[:50]}")
    if len(sequence_as_int) > 50:
        print(f"  ... ({len(sequence_as_int) - 50} more)")

    # Calculate perplexity
    print(f"\nCalculating perplexity...")
    perplexity = scorer._calculate_perplexity_for_sequence(sequence)

    print(f"\n{Colors.BOLD}Perplexity: {Colors.CYAN}{perplexity:.4f}{Colors.ENDC}")

    return perplexity


def test_global_human_like_scorer(argument: str, edits: List[Dict],
                                  model_path: str = None, threshold: float = None):
    """Test the global human-like scorer with detailed output."""
    print_section("Global Human-Like Scorer Test")

    # Initialize scorer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scorer_kwargs = {"device": device}
    if model_path:
        scorer_kwargs["model_path"] = model_path
    if threshold:
        scorer_kwargs["threshold"] = threshold

    scorer = GlobalHumanLikeScorer(**scorer_kwargs)
    print(f"Model path: {scorer.model_path}")
    print(f"Threshold: {scorer.threshold}")
    print(f"Max length: {scorer.max_len}")

    # Show input
    print_section("Input Data")
    print(f"Argument ({len(argument)} chars):")
    print(f"{Colors.BLUE}{argument}{Colors.ENDC}\n")
    print(f"Number of edits: {len(edits)}")
    for i, edit in enumerate(edits):
        print(f"\nEdit {i+1}:")
        print(f"  Reason: {edit.get('reason', 'N/A')}")
        print(f"  Inappropriate: '{edit.get('inappropriate_part', '')}'")
        print(f"  Rewritten: '{edit.get('rewritten_part', '')}'")
        print(f"  Sentence ID: {edit.get('sentence_id', 'N/A')}")

    # Step 1: Tokenization
    print_section("Processing Steps")
    tokens, offsets = visualize_tokenization(scorer, argument)

    # Step 2: Edit sequence
    sequence = visualize_edit_sequence(scorer, argument, edits)

    # Step 3: Perplexity
    perplexity = visualize_perplexity_calculation(scorer, sequence)

    # Final result
    print_section("Final Result")
    binary_score, actual_perplexity = scorer.calculate_global_human_likeness(argument, edits)

    print(f"Perplexity: {Colors.CYAN}{actual_perplexity:.4f}{Colors.ENDC}")
    print(f"Threshold: {Colors.YELLOW}{scorer.threshold:.4f}{Colors.ENDC}")

    if binary_score == 1.0:
        print(f"Binary Score: {Colors.GREEN}{Colors.BOLD}{binary_score:.1f} (PASS){Colors.ENDC}")
        print(f"\n{Colors.GREEN}✓ The edit pattern is HUMAN-LIKE at the document level{Colors.ENDC}")
    else:
        print(f"Binary Score: {Colors.RED}{Colors.BOLD}{binary_score:.1f} (FAIL){Colors.ENDC}")
        print(f"\n{Colors.RED}✗ The edit pattern is NOT HUMAN-LIKE at the document level{Colors.ENDC}")

    margin = actual_perplexity - scorer.threshold
    print(f"\nMargin: {Colors.CYAN}{margin:+.4f}{Colors.ENDC} (negative is better)")


def load_example_from_jsonl(filepath: str, index: int = 0):
    """Load an example from a generated edits JSONL file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == index:
                data = json.loads(line)
                argument = data.get('argument', '')
                edits = data.get('edits', [])
                return argument, edits
    raise IndexError(f"Example index {index} not found in {filepath}")


def get_predefined_examples():
    """Get predefined test examples."""
    examples = []

    # Example 1: Few edits (should be human-like)
    examples.append({
        "name": "Few edits (likely human-like)",
        "argument": "The policy is terrible and stupid. It will never work in practice. We need to consider alternative approaches that are more sensible.",
        "edits": [
            {
                "inappropriate_part": "terrible and stupid",
                "rewritten_part": "poorly designed",
                "reason": "Toxic Emotions",
                "sentence_id": 1
            },
            {
                "inappropriate_part": "never",
                "rewritten_part": "may not",
                "reason": "Toxic Emotions",
                "sentence_id": 2
            }
        ]
    })

    # Example 2: Many edits (likely not human-like)
    examples.append({
        "name": "Many edits (likely not human-like)",
        "argument": "The policy is bad. It is wrong. Nobody likes it. Everyone hates it. It will fail.",
        "edits": [
            {
                "inappropriate_part": "bad",
                "rewritten_part": "suboptimal",
                "reason": "Toxic Emotions",
                "sentence_id": 1
            },
            {
                "inappropriate_part": "wrong",
                "rewritten_part": "flawed",
                "reason": "Toxic Emotions",
                "sentence_id": 2
            },
            {
                "inappropriate_part": "Nobody likes it",
                "rewritten_part": "It has limited support",
                "reason": "Toxic Emotions",
                "sentence_id": 3
            },
            {
                "inappropriate_part": "Everyone hates it",
                "rewritten_part": "It faces significant opposition",
                "reason": "Toxic Emotions",
                "sentence_id": 4
            },
            {
                "inappropriate_part": "fail",
                "rewritten_part": "face challenges",
                "reason": "Toxic Emotions",
                "sentence_id": 5
            }
        ]
    })

    # Example 3: No edits
    examples.append({
        "name": "No edits (edge case)",
        "argument": "This is a well-written argument that requires no changes.",
        "edits": []
    })

    return examples


def cycle_through_file(filepath: str, model_path: str = None, threshold: float = None,
                       start_index: int = 0, max_examples: int = None):
    """Cycle through examples from a JSONL file interactively."""
    # Count total examples
    with open(filepath, 'r', encoding='utf-8') as f:
        total_examples = sum(1 for _ in f)

    print(f"{Colors.BOLD}{Colors.CYAN}File: {filepath}{Colors.ENDC}")
    print(f"{Colors.CYAN}Total examples: {total_examples}{Colors.ENDC}")
    print(f"{Colors.CYAN}Starting from index: {start_index}{Colors.ENDC}\n")

    examples_shown = 0
    current_index = start_index

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < start_index:
                continue

            if max_examples and examples_shown >= max_examples:
                break

            try:
                data = json.loads(line)
                argument = data.get('argument', '')
                edits = data.get('edits', [])

                print(f"\n\n{Colors.BOLD}{Colors.HEADER}")
                print("=" * 100)
                print(f"EXAMPLE {i} (shown {examples_shown + 1})")
                print("=" * 100)
                print(Colors.ENDC)

                test_global_human_like_scorer(argument, edits, model_path, threshold)

                examples_shown += 1
                current_index = i + 1

                # Ask user what to do next
                if current_index < total_examples and (not max_examples or examples_shown < max_examples):
                    print(f"\n{Colors.YELLOW}Options:{Colors.ENDC}")
                    print(f"  {Colors.YELLOW}[Enter]{Colors.ENDC} - Next example")
                    print(f"  {Colors.YELLOW}[q]{Colors.ENDC} - Quit")
                    print(f"  {Colors.YELLOW}[s <N>]{Colors.ENDC} - Skip to example N")
                    choice = input(f"{Colors.YELLOW}Choice: {Colors.ENDC}").strip().lower()

                    if choice == 'q':
                        print(f"\n{Colors.GREEN}Exiting...{Colors.ENDC}")
                        break
                    elif choice.startswith('s '):
                        try:
                            skip_to = int(choice.split()[1])
                            if 0 <= skip_to < total_examples:
                                current_index = skip_to
                                # Re-open file and continue from that position
                                with open(filepath, 'r', encoding='utf-8') as f2:
                                    for j, line2 in enumerate(f2):
                                        if j < skip_to:
                                            continue
                                        if j == skip_to:
                                            data = json.loads(line2)
                                            argument = data.get('argument', '')
                                            edits = data.get('edits', [])
                                            print(f"\n\n{Colors.BOLD}{Colors.HEADER}")
                                            print("=" * 100)
                                            print(f"EXAMPLE {j}")
                                            print("=" * 100)
                                            print(Colors.ENDC)
                                            test_global_human_like_scorer(argument, edits, model_path, threshold)
                                            current_index = j + 1
                                            break
                            else:
                                print(f"{Colors.RED}Invalid index. Continuing...{Colors.ENDC}")
                        except (ValueError, IndexError):
                            print(f"{Colors.RED}Invalid input. Continuing...{Colors.ENDC}")

            except json.JSONDecodeError as e:
                print(f"{Colors.RED}Error parsing line {i}: {e}{Colors.ENDC}")
                continue
            except Exception as e:
                print(f"{Colors.RED}Error processing example {i}: {e}{Colors.ENDC}")
                continue

    print(f"\n{Colors.GREEN}Finished! Showed {examples_shown} examples.{Colors.ENDC}")


def discover_generated_edits_files():
    """Find all JSONL files in the generated_edits directory."""
    import glob
    pattern = "models/generated_edits/*.jsonl"
    files = sorted(glob.glob(pattern))
    return files


def interactive_file_selection():
    """Let user interactively select a file."""
    files = discover_generated_edits_files()

    if not files:
        print(f"{Colors.RED}No JSONL files found in models/generated_edits/{Colors.ENDC}")
        return None

    print(f"{Colors.BOLD}{Colors.CYAN}Available generated edits files:{Colors.ENDC}\n")
    for i, filepath in enumerate(files):
        filename = Path(filepath).name
        print(f"  {Colors.YELLOW}[{i}]{Colors.ENDC} {filename}")

    print(f"\n{Colors.YELLOW}Select a file (0-{len(files)-1}) or [q] to quit: {Colors.ENDC}", end='')
    choice = input().strip().lower()

    if choice == 'q':
        return None

    try:
        index = int(choice)
        if 0 <= index < len(files):
            return files[index]
        else:
            print(f"{Colors.RED}Invalid selection{Colors.ENDC}")
            return None
    except ValueError:
        print(f"{Colors.RED}Invalid input{Colors.ENDC}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Interactive test for Global Human-Like Scorer")
    parser.add_argument("--input_jsonl", type=str, help="Path to generated edits JSONL file")
    parser.add_argument("--example_index", type=int, default=0, help="Starting index for cycling through examples (default: 0)")
    parser.add_argument("--max_examples", type=int, help="Maximum number of examples to show")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, help="Perplexity threshold")
    parser.add_argument("--predefined", type=int, help="Use predefined example (0, 1, or 2)")
    parser.add_argument("--interactive", action="store_true", help="Interactively select file from models/generated_edits/")
    args = parser.parse_args()

    if args.predefined is not None:
        examples = get_predefined_examples()
        if 0 <= args.predefined < len(examples):
            example = examples[args.predefined]
            print(f"Using predefined example: {example['name']}")
            test_global_human_like_scorer(example['argument'], example['edits'], args.model_path, args.threshold)
        else:
            print(f"Error: Predefined example index must be 0-{len(examples)-1}")
    elif args.interactive:
        filepath = interactive_file_selection()
        if filepath:
            cycle_through_file(filepath, args.model_path, args.threshold, args.example_index, args.max_examples)
    elif args.input_jsonl:
        print(f"Cycling through examples from {args.input_jsonl}...")
        cycle_through_file(args.input_jsonl, args.model_path, args.threshold, args.example_index, args.max_examples)
    else:
        # Default: show available files and let user select
        print(f"{Colors.BOLD}{Colors.HEADER}Global Human-Like Scorer Interactive Test{Colors.ENDC}\n")
        filepath = interactive_file_selection()
        if filepath:
            cycle_through_file(filepath, args.model_path, args.threshold, args.example_index, args.max_examples)


if __name__ == "__main__":
    main()
