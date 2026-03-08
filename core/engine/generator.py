import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class Generator:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 4096, stop: list[str] = None):
        self.llm = LLM(model=model, gpu_memory_utilization=0.6)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    def generate(self, prompts: list[str]) -> list[str]:
        results = self.llm.generate(prompts, self.sampling_params)
        return [self.tokenizer.decode(result.outputs[0].token_ids) for result in results]
    
    def generate_from_messages_list(self, messages_list: list[list[dict]]) -> list[str]:
        """
        Generate responses from a list of message histories.

        Each message history is a list of dicts with 'role' and 'content' keys.

        Args:
            messages: List of message histories (list of dicts)

        Returns:
            List of generated responses
        """
        # Convert messages to prompt strings
        prompts = [self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + '<think>' for messages in messages_list]

        return self.generate(prompts)

