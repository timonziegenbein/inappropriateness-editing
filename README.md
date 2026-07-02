# Teaching LLMs Human-Like Editing of Inappropriate Argumentation via Reinforcement Learning

This repository contains the code and experimental setup for training and evaluating reinforcement learning models for identifying and editing inappropriate text in argumentative writing.

## Paper

**"Teaching LLMs Human-Like Editing of Inappropriate Argumentation via Reinforcement Learning"** (Ziegenbein et al., ACL 2026)
- Available at: https://aclanthology.org/2026.acl-long.1789/

## Table of Contents

- [Paper](#paper)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Training](#training)

## Project Overview

This project implements a reinforcement learning system using GRPO (Generative Reinforcement Policy Optimization) to fine-tune large language models (Llama-3.1-8B-Instruct) for the task of identifying and editing inappropriate text while preserving the author's core message.

The system analyzes argumentative text and identifies inappropriate parts based on four dimensions:
- **Toxic Emotions**: Deceptive or overly intense emotional appeals
- **Missing Commitment**: Lack of seriousness or openness to other arguments
- **Missing Intelligibility**: Unclear meaning, irrelevance, or confusing reasoning
- **Other Reasons**: Severe orthographic errors or other issues

### Key Features

- **Dual Reward System**: Combines local (edit-level) and global (document-level) rewards with 80/20 weighting
- **Multi-component Scorers**: Semantic similarity, fluency, and pattern conformity evaluation
- **Two-step Evaluation Pipeline**: Separate generation and scoring for efficient ablation studies
- **Comprehensive Annotation Interface**: Django-based web application for human evaluation studies

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd appropriateness-edit
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
# Core dependencies for training and evaluation
pip install torch transformers datasets accelerate peft vllm
pip install unsloth  # For efficient LoRA training
pip install spacy wandb weave json-repair

# Download spaCy model for sentence tokenization
python -m spacy download en_core_web_sm

# Install Google EmbeddingGemma for semantic similarity scoring
# See scorers/local_scorers/semantic_similarity/README.md for details
```

### 4. Download required models

```bash
python models/download_models.py
```

**Note**: Some model files and large data files are not included in this submission due to size constraints (200MB limit). The full models can be downloaded using the script above or accessed from HuggingFace.

### 5. Set up environment variables

```bash
# HuggingFace token for model access
export HF_TOKEN="your-huggingface-token"

# Optional: For baseline comparisons
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"

# Optional: For experiment tracking
export WANDB_API_KEY="your-wandb-key"
```

## Quick Start

### Training a Model

Train the GRPO model with all scorers enabled:

```bash
python models/grpo.py \
    --model_name unsloth/Llama-3.1-8B-Instruct \
    --output_dir models/trained/my_model \
    --use_semantic_similarity \
    --use_fluency \
    --use_pattern_conformity
```

For detailed training options, see `models/grpo.py --help` and the [Training](#training) section below.

### Evaluating a Trained Model

We recommend the two-step evaluation workflow for efficiency:

**Step 1: Generate edits (run once)**

```bash
python models/generate_edits.py \
    --checkpoint_root models/trained/my_model \
    --output_jsonl models/generated_edits/my_model.jsonl
```

**Step 2: Evaluate with different scorer configurations**

```bash
# Evaluate with all scorers
python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/my_model.jsonl \
    --output_jsonl models/predictions/my_model_all.jsonl

# Ablation: disable pattern conformity scorer
python models/evaluate_edits.py \
    --input_jsonl models/generated_edits/my_model.jsonl \
    --output_jsonl models/predictions/my_model_no_hl.jsonl \
    --disable_pattern_conformity
```

See `models/EVALUATION_WORKFLOW.md` for complete documentation.

## Training

### Basic Training

Train with default configuration (all scorers enabled):

```bash
python models/grpo.py \
    --model_name unsloth/Llama-3.1-8B-Instruct \
    --output_dir models/trained/my_model
```

### Custom Scorer Configuration

Enable specific scorers:

```bash
python models/grpo.py \
    --model_name unsloth/Llama-3.1-8B-Instruct \
    --output_dir models/trained/my_model \
    --use_semantic_similarity \
    --use_fluency \
    --use_pattern_conformity
```

### Resume from Checkpoint

```bash
python models/grpo.py \
    --model_name unsloth/Llama-3.1-8B-Instruct \
    --output_dir models/trained/my_model \
    --resume_from_checkpoint models/trained/my_model/checkpoint-1000 \
    --wandb_run_id <previous-run-id>
```

