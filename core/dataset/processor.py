"""
Data processing module for loading and formatting MATH dataset examples.

This module provides functionality to load MATH dataset validation examples
and format them as string prompts using the r1_zero prompt template.
"""

import json
import pathlib


class DataProcessor:
    """Processor for MATH dataset evaluation examples."""

    def __init__(
        self,
        data_dir: pathlib.Path,
        eval_file: pathlib.Path,
        r1_prompt_file: pathlib.Path,
    ):
        """
        Initialize the DataProcessor.

        Args:
            data_dir: Directory containing dataset files
            eval_file: Path to the evaluation data file (JSON or JSONL)
            r1_prompt_file: Path to the r1_zero prompt template file
        """
        self.data_dir = data_dir
        self.eval_file = eval_file
        self.r1_prompt_file = r1_prompt_file

    def _load_json_data(self, file_path: pathlib.Path) -> list[dict]:
        """
        Load JSON data from a file.

        Handles both JSON arrays and JSONL formats.

        Args:
            file_path: Path to the data file

        Returns:
            List of data dictionaries
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

            # Try loading as JSON array first
            try:
                data = json.loads(content)
                return data
            except json.JSONDecodeError:
                # Try loading as JSONL
                f.seek(0)
                data = [json.loads(line) for line in f]
                return data

    def _preprocess_data(self, data: list[dict]) -> list[dict]:
        """
        Preprocess raw data entries.

        Ensures problem and expected_answer fields are strings.

        Args:
            data: Raw data list

        Returns:
            Preprocessed data list
        """
        sanitized_data = []
        for entry in data:
            entry = entry.copy()

            # Handle list problems
            if isinstance(entry.get("problem"), list):
                entry["problem"] = " ".join(entry["problem"])

            # Handle list answers
            if isinstance(entry.get("expected_answer"), list):
                str_answers = [str(e) for e in entry["expected_answer"]]
                entry["expected_answer"] = ",".join(str_answers)

            sanitized_data.append(entry)

        return sanitized_data

    def convert_r1_zero_format(self) -> list[dict]:
        """
        Convert evaluation data to r1_zero prompt format.

        Returns:
            List of dicts with 'prompt' and 'expected_answer' keys
        """
        raw_data = self._load_json_data(self.eval_file)
        processed_data = self._preprocess_data(raw_data)
        r1_template = self.r1_prompt_file.read_text(encoding="utf-8")

        r1_formatted_data = []
        for entry in processed_data:
            # Use {question} as the placeholder (matching the r1_zero.prompt format)
            problem = entry.get("problem", "")
            prompt = r1_template.replace("{question}", problem)

            r1_formatted_data.append(
                {
                    "prompt": prompt,
                    "expected_answer": entry["expected_answer"],
                }
            )

        return r1_formatted_data

    def convert_r1_zero_format_file(self, output_path: pathlib.Path) -> None:
        """
        Convert and save r1_zero formatted data to a file.

        Args:
            output_path: Path to save the formatted data
        """
        r1_data = self.convert_r1_zero_format()
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for entry in r1_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Saved {len(r1_data)} formatted examples to {output_path}")
