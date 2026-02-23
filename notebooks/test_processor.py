import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import pathlib
    import sys
    import os
    pathlib.Path(__file__).resolve().parent.parent
    project_root = pathlib.Path("/home/xuzhenhua/git/assignment5-alignment")
    sys.path.insert(0, str(project_root))
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    return


@app.cell
def _():
    from configs.config_t import settings
    from core.dataset.processor import DataProcessor

    processor = DataProcessor(
            data_dir=settings.datasets_dir,
            eval_file=settings.eval_file,
            r1_prompt_file=settings.r1_zero_prompt_file,
        )

    messages_list = processor.convert_r1_zero_format_messages()
    return messages_list, settings


@app.cell
def _(messages_list):
    messages_list[0]
    return


@app.cell
def _(messages_list, settings):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(settings.model)

    prompts = [tokenizer.apply_chat_template(messages_list[i], tokenize=False, add_generation_prompt=True) + '<think>' for i in range(5)]
    print(prompts[0])
    return (prompts,)


@app.cell
def _():
    # # from core.engine.generator import Generator
    # # generator = Generator(model=str(settings.model))
    # from vllm import LLM, SamplingParams

    # llm = LLM(model=str(settings.model), gpu_memory_utilization=0.6)
    # sampling_params = SamplingParams(
    #              temperature=0.6,
    #              max_tokens=4096,
    #              stop=["</answer>"],
    #          )
    # tokenized_prompts = [tokenizer.apply_chat_template(messages_list[i], tokenize=True, add_generation_prompt=True) + tokenizer.tokenize("<think>") for i in range(5)]
    # print(tokenized_prompts[0])
    return


@app.cell
def _(settings):
    from core.engine.generator import Generator
    generator = Generator(model=str(settings.model), temperature=0.8, max_tokens=4096, stop=["</answer>"])
    return (generator,)


@app.cell
def _(generator, prompts):
    results = generator.generate(prompts)
    print(results[0])
    return (results,)


@app.cell
def _(results):
    for result in results:
        print(result)
        print("========================================")
    return


if __name__ == "__main__":
    app.run()
