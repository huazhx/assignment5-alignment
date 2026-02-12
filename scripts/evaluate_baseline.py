#!/usr/bin/env python3
"""
Evaluate Qwen 2.5 Math 1.5B zero-shot performance on MATH dataset.

This script:
1. Loads MATH validation examples
2. Formats them using r1_zero prompt
3. Generates outputs using vLLM
4. Calculates evaluation metrics
5. Serializes results to disk
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm import LLM, SamplingParams

from configs.config import settings
from core.dataset.processor import DataProcessor
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from core.pipeline.eval_vllm import evaluate_with_ground_truth


def main() -> None:
    """Run zero-shot baseline evaluation on MATH dataset."""

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices

    print("Loading MATH validation data...")
    processor = DataProcessor(
        data_dir=settings.datasets_dir,
        eval_file=settings.eval_file,
        r1_prompt_file=settings.r1_zero_prompt_file,
    )

    # Convert to r1_zero format
    examples = processor.convert_r1_zero_format()
    print(f"Loaded {len(examples)} examples")

    # Initialize vLLM model
    print(f"Loading model from {settings.model}...")
    llm = LLM(
        model=str(settings.model),
        gpu_memory_utilization=0.6,
    )

    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        stop=settings.stop,
    )

    # Create output directory
    output_dir = settings.outputs_dir / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.jsonl"

    print("Generating responses and evaluating...")
    results, metrics = evaluate_with_ground_truth(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        examples=examples,
        eval_sampling_params=sampling_params,
        output_file=output_file,
    )

    print("\n=== Evaluation Results ===")
    print(f"Total examples: {metrics['total']}")
    print(f"Format accuracy: {metrics['format_accuracy']:.2%}")
    print(f"Answer accuracy: {metrics['answer_accuracy']:.2%}")
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
