"""
Evaluation metrics module for computing model performance on MATH dataset.

This module provides functionality to compute evaluation metrics including
format accuracy and answer accuracy.
"""

from pathlib import Path
import json
from typing import Callable

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


class Evaluator:
    """
    Evaluate model outputs on MATH dataset.

    Reads evaluation data, computes metrics, and serializes results.
    """

    def __init__(self, reward_fn: Callable[[str, str], dict[str, float]] | None = None):
        """
        Initialize the Evaluator.

        Args:
            reward_fn: Optional custom reward function. Defaults to r1_zero_reward_fn.
        """
        self.reward_fn = reward_fn or r1_zero_reward_fn

    def compute_metrics(
        self, results: list[dict], ground_truths: list[str]
    ) -> dict[str, float]:
        """
        Compute evaluation metrics from results.

        Args:
            results: List of result dicts containing 'generation' key
            ground_truths: List of ground truth answers

        Returns:
            Dictionary with format_accuracy and answer_accuracy
        """
        total = len(results)
        if total == 0:
            return {"format_accuracy": 0.0, "answer_accuracy": 0.0, "total": 0}

        format_correct = 0
        answer_correct = 0

        for result, gt in zip(results, ground_truths):
            generation = result.get("generation", "")
            reward_dict = self.reward_fn(generation, gt)

            format_correct += reward_dict.get("format_reward", 0)
            answer_correct += reward_dict.get("answer_reward", 0)

        return {
            "total": total,
            "format_accuracy": format_correct / total,
            "answer_accuracy": answer_correct / total,
        }

    def evaluate_generations(
        self,
        generations: list[str],
        ground_truths: list[str],
        output_file: Path | None = None,
    ) -> tuple[list[dict], dict[str, float]]:
        """
        Evaluate generations against ground truths.

        Args:
            generations: List of model generations
            ground_truths: List of ground truth answers
            output_file: Optional path to save results

        Returns:
            Tuple of (results_with_scores, metrics)
        """
        results = []

        for i, (gen, gt) in enumerate(zip(generations, ground_truths)):
            reward_dict = self.reward_fn(gen, gt)

            result = {
                "id": i,
                "generation": gen,
                "expected_answer": gt,
                **reward_dict,
            }
            results.append(result)

        metrics = self._compute_aggregate_metrics(results)

        if output_file is not None:
            self._save_results(results, metrics, output_file)

        return results, metrics

    def _compute_aggregate_metrics(self, results: list[dict]) -> dict[str, float]:
        """Compute aggregate metrics from scored results."""
        total = len(results)
        if total == 0:
            return {"total": 0, "format_accuracy": 0.0, "answer_accuracy": 0.0}

        format_correct = sum(r.get("format_reward", 0) for r in results)
        answer_correct = sum(r.get("answer_reward", 0) for r in results)

        return {
            "total": total,
            "format_accuracy": format_correct / total,
            "answer_accuracy": answer_correct / total,
        }

    def _save_results(
        self, results: list[dict], metrics: dict[str, float], output_file: Path
    ) -> None:
        """Save results and metrics to disk."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Save metrics
        metrics_file = output_file.parent / f"{output_file.stem}_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Results saved to {output_file}")
        print(f"Metrics saved to {metrics_file}")
        print(f"Format accuracy: {metrics['format_accuracy']:.2%}")
        print(f"Answer accuracy: {metrics['answer_accuracy']:.2%}")
