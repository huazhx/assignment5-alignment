import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import pathlib
    from configs.config import settings

    BASE_DATA_DIR = pathlib.Path('/home/xuzhenhua/git/assignment5-alignment/datasets')
    BASE_OUTPUT_DIR = pathlib.Path('/home/xuzhenhua/git/assignment5-alignment/outputs/')
    EVAL_FILE    = BASE_DATA_DIR / 'sft-reason' / 'val.jsonl'  


    # 1. 在导入 torch 之前设置 GPU 环境
    os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices
    return BASE_OUTPUT_DIR, EVAL_FILE, pathlib


@app.cell
def _(EVAL_FILE):
    import json
    from datasets import Dataset

    with open(EVAL_FILE, "r") as f:
        data = json.load(f)

    # Sanitize: Ensure everything is a string (or your expected type)
    for entry in data:
        if isinstance(entry['problem'], list):
            entry['problem'] = " ".join(entry['problem'])
            print(entry['problem'])
        if isinstance(entry['expected_answer'], list):
            str_answers = [str(e) for e in entry['expected_answer']]
            entry['expected_answer'] = ",".join(str_answers)
            # entry['expected_answer'] = ",".join(entry['expected_answer'])

    ds = Dataset.from_list(data)
    return ds, json


@app.cell
def _(ds):
    ds[0]
    return


@app.cell
def _():
    from attr.filters import include
    from jupyterlab.semver import inc
    from vllm import LLM, SamplingParams

    # Sample prompts
    prompts = [
        "What is the capital of France?",
        "Solve for x: 2x + 3 = 7.",
        "Who wrote 'To Kill a Mockingbird'?"
    ]

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    llm = LLM(model="/home/xuzhenhua/models/Qwen2.5-Math-1.5B")
    return llm, sampling_params


@app.cell
def _(ds):
    ds_sample = ds.select(range(3))
    return (ds_sample,)


@app.cell
def _(ds_sample, pathlib):
    # Format prompts using cs336_alignment/prompts/r1_zero.prompt
    # NOTE: use .replace instead of .format because math questions can contain braces like { }.
    PROMPT_FILE = pathlib.Path('/home/xuzhenhua/git/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt')
    r1_template = PROMPT_FILE.read_text()

    def render_r1_zero(question: str) -> str:
        return r1_template.replace('{question}', question.strip())

    r1_math_prompts = [render_r1_zero(item['problem']) for item in ds_sample]
    r1_math_prompts[0]
    return (r1_math_prompts,)


@app.cell
def _(r1_math_prompts):
    r1_math_prompts
    return


@app.cell
def _(llm, r1_math_prompts, sampling_params):
    outputs = llm.generate(r1_math_prompts, sampling_params)

    for i, output in enumerate(outputs):
        print(f"Prompt: {r1_math_prompts[i]}")
        print(f"Generated Response: {output.outputs[0].text}")
        print("-" * 50)
    return (outputs,)


@app.cell
def _(ds_sample, outputs, r1_math_prompts):
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
    for i_a_w, output_a_w in enumerate(outputs):
        generated_text_w = output_a_w.outputs[0].text
        expected_answer_w = ds_sample[i_a_w]['expected_answer']
        reward_w = r1_zero_reward_fn(generated_text_w, expected_answer_w)
        print(f"Prompt: {r1_math_prompts[i_a_w]}")
        print(f"Generated Response: {generated_text_w}")
        print(f"Expected Answer: {expected_answer_w}")
        print(f"Reward: {reward_w}")
        print("-" * 50)
    return (r1_zero_reward_fn,)


@app.cell
def _(
    BASE_OUTPUT_DIR,
    ds_sample,
    json,
    outputs,
    r1_math_prompts,
    r1_zero_reward_fn,
):
    # save prompts, outputs, metrics into a jsonl file
    OUTPUT_FILE = BASE_OUTPUT_DIR / 'r1_zero_eval_outputs.jsonl'
    with open(OUTPUT_FILE, "w") as f_out:
        for i_a_w, output_a_w in enumerate(outputs):
            generated_text_w = output_a_w.outputs[0].text
            expected_answer_w = ds_sample[i_a_w]['expected_answer']
            reward_w = r1_zero_reward_fn(generated_text_w, expected_answer_w)
            record = {
                "prompt": r1_math_prompts[i_a_w],
                "generated_response": generated_text_w,
                "expected_answer": expected_answer_w,
                "reward": reward_w
            }
            f_out.write(json.dumps(record) + "\n")
    return


if __name__ == "__main__":
    app.run()
