import os
from vllm import LLM, SamplingParams


class Generator:
    def __init__(self, temperature: float, max_tokens: int, stop: list[str], model: str):
        self.llm = LLM(model=model)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    def generate(self, prompts: list[str]) -> list[str]:
        responses = self.llm.generate(prompts, self.sampling_params)
        return [response.outputs[0].text for response in responses]

