import pathlib
import json 
from configs.config import settings


class DataProcessor:
    def __init__(self):
        self.data_dir = settings.datasets_dir
        self.eval_file = settings.eval_file
        self.r1_prompt_file = settings.r1_zero_prompt_file

    def _load_json_data(self, file_path: pathlib.Path) -> list[dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def _preprocess_data(self, data: list[dict]) -> list[dict]:

        sanitized_data = []
        for entry in data:
            if isinstance(entry['problem'], list):
                entry['problem'] = " ".join(entry['problem'])
                print(entry['problem'])
            if isinstance(entry['expected_answer'], list):
                str_answers = [str(e) for e in entry['expected_answer']]
                entry['expected_answer'] = ",".join(str_answers)

            sanitized_data.append(entry)
        return sanitized_data
    
    def convert_r1_zero_format(self) -> list[dict]:
        raw_data = self._load_json_data(self.eval_file)
        processed_data = self._preprocess_data(raw_data)
        r1_template = self.r1_prompt_file.read_text(encoding='utf-8')
        r1_formatted_data = []
        for entry in processed_data:
            prompt = r1_template.replace("{problem}", entry['problem'])
            r1_formatted_data.append({
                "prompt": prompt,
                "expected_answer": entry['expected_answer']
            })
        return r1_formatted_data
    

    def convert_r1_zero_format_file(self, output_path: pathlib.Path):
        r1_data = self.convert_r1_zero_format()
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in r1_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    