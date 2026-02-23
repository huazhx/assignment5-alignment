"""
Evaluation pipeline for vLLM models.

This module provides functionality to evaluate language models on MATH dataset
using vLLM for efficient inference and compute evaluation metrics.
"""

import json
from pathlib import Path
from typing import Callable

from vllm import LLM, SamplingParams



def evaluate_vllm(
    generator,
    reward_fn: Callable[[str, str], dict[str, float]],
    examples: list[dict],
    output_file: Path | None = None,
) -> list[dict]:
    """
    Evaluate a language model on examples with ground truth answers.

    Args:
        generator: The Generator instance to evaluate
        reward_fn: Function that takes (response, ground_truth) and returns reward dict
        examples: List of dicts with 'prompt' and 'expected_answer' keys
        output_file: Optional path to save results as JSONL

    Returns:
        List of result dictionaries with prompts, generations, scores, and metrics
    """
    prompts = [ex["prompt"] for ex in examples]
    results = []

    # Generate outputs for all prompts
    outputs = generator.generate(prompts)

    # Compute metrics
    format_correct = 0
    answer_correct = 0
    total = len(examples)

    for i, (example, output) in enumerate(zip(examples, outputs)):
        generated_text = output
        ground_truth = example["expected_answer"]

        # Compute reward scores
        reward_dict = reward_fn(generated_text, ground_truth)

        result = {
            "id": i,
            "prompt": example["prompt"],
            "generation": generated_text,
            "expected_answer": ground_truth,
            **reward_dict,
        }
        results.append(result)

        # Track metrics
        format_correct += reward_dict.get("format_reward", 0)
        answer_correct += reward_dict.get("answer_reward", 0)

    # Compute aggregate metrics
    metrics = {
        "total": total,
        "format_accuracy": format_correct / total if total > 0 else 0,
        "answer_accuracy": answer_correct / total if total > 0 else 0,
    }

    # Save results to disk if output file is specified
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_file = output_file
        with open(results_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Save metrics summary
        metrics_file = output_file.parent / f"{output_file.stem}_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Results saved to {results_file}")
        print(f"Metrics saved to {metrics_file}")
        print(f"Format accuracy: {metrics['format_accuracy']:.2%}")
        print(f"Answer accuracy: {metrics['answer_accuracy']:.2%}")

    return results, metrics


# Alias for compatibility
evaluate_with_ground_truth = evaluate_vllm
