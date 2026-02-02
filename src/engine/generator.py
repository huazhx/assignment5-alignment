import os
from configs.config import settings
from vllm import LLM, SamplingParams

os.environ['CUDA_VISIBLE_DEVICES'] = settings.cuda_visible_devices

class Generator:
    def __init__(self):
        self.model = LLM.load(
            settings.model,
            sampling_params=SamplingParams(
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                stop=settings.stop,
                include_input=settings.include_input
            )
        )

    def generate(self, prompt: str) -> str:
        response = self.model.generate([prompt])
        return response[0].outputs[0].text
    
    def batch_generate(self, prompts: list[str]) -> list[str]:
        responses = self.model.generate(prompts)
        return [response.outputs[0].text for response in responses]
    