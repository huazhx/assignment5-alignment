import marimo

__generated_with = "0.19.7"
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
    import core
    core.__file__
    return


@app.cell
def _():
    # Import Generator immediately after modifying sys.path
    from core.engine.generator import Generator

    # Initialize the Generator
    generator = Generator(
        temperature=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        model="/home/xuzhenhua/models/Qwen2.5-Math-1.5B"
    )
    return (generator,)


@app.cell
def _(generator):
    # Test prompts
    prompts = [
        """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
    User: Two fair 6-sided dice are rolled.  What is the probability the sum rolled is 9?
    Assistant: <think>""",
    ]

    outputs = generator.generate(prompts)
    return outputs, prompts


@app.cell
def _(outputs, prompts):
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}")
        print(f"Response: {output}")
        print("-" * 50)
    return


if __name__ == "__main__":
    app.run()
