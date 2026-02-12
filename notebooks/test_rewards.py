import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import importlib
    import cs336_alignment.drgrpo_grader
    import inspect

    # Force reload to get latest changes
    importlib.reload(cs336_alignment.drgrpo_grader)
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    # Debug: print actual source code of function
    print("=== FUNCTION SOURCE ===")
    print(inspect.getsource(r1_zero_reward_fn)[:500])
    print("=== END SOURCE ===")
    return (r1_zero_reward_fn,)


@app.cell
def _(r1_zero_reward_fn):
    # Test 1: Correct answer with <answer> tags
    response1 = "Let me solve this step by step.\n</think> <answer>4</answer>"
    ground_truth1 = "4"

    # Debug: check what's actually in the string
    print("=== DEBUG ===")
    print(f"Response repr: {repr(response1)}")
    print(f"Has ' <answer>': {" <answer>" in response1}")
    print(f"Has '</answer>': {'</answer>' in response1}")
    print(f"Has '<answer>': {'<answer>' in response1}")
    print("=== END DEBUG ===")
    result1 = r1_zero_reward_fn(response1, ground_truth1)

    print("Test 1: Correct answer with <answer> tags")
    print(f"Response: {response1}")
    print(f"Ground truth: {ground_truth1}")
    print(f"Result: {result1}")
    print("-" * 50)
    return


@app.cell
def _(r1_zero_reward_fn):
    # Test 2: Wrong answer with <answer> tags
    response2 = "Let me solve this step by step.\n </think> <answer>5</answer>"
    ground_truth2 = "4"
    result2 = r1_zero_reward_fn(response2, ground_truth2)

    print("Test 2: Wrong answer with <answer> tags")
    print(f"Response: {response2}")
    print(f"Ground truth: {ground_truth2}")
    print(f"Result: {result2}")
    print("-" * 50)
    return


@app.cell
def _(r1_zero_reward_fn):
    # Test 3: Missing <answer> tags (no format reward)
    response3 = "The answer is 4."
    ground_truth3 = "4"
    result3 = r1_zero_reward_fn(response3, ground_truth3)

    print("Test 3: Missing <answer> tags (format penalty)")
    print(f"Response: {response3}")
    print(f"Ground truth: {ground_truth3}")
    print(f"Result: {result3}")
    print("-" * 50)
    return


@app.cell
def _(r1_zero_reward_fn):
    # Test 4: With \boxed format (math LaTeX)
    response4 = "Let me solve step by step.\n <answer>\\boxed{42}</answer>"
    ground_truth4 = "42"
    result4 = r1_zero_reward_fn(response4, ground_truth4)

    print("Test 4: With \\boxed format (LaTeX math)")
    print(f"Response: {response4}")
    print(f"Ground truth: {ground_truth4}")
    print(f"Result: {result4}")
    print("-" * 50)
    return


@app.cell
def _(r1_zero_reward_fn):
    # Test 5: Multiple ground truths (list) - matches any
    response5 = "Solving the equation.\n <answer>\\boxed{x=2}</answer>"
    ground_truth5 = ["x=2", "2", "x = 2"]  # Multiple valid answers
    result5 = r1_zero_reward_fn(response5, ground_truth5)

    print("Test 5: Multiple ground truths (matches any)")
    print(f"Response: {response5}")
    print(f"Ground truth: {ground_truth5}")
    print(f"Result: {result5}")
    print("-" * 50)
    return


@app.cell
def _(r1_zero_reward_fn):
    # Test 6: Complex math problem
    response6 = """
    Let me solve: 2x + 3 = 7
    First, subtract 3 from both sides: 2x = 4
    Then divide by 2: x = 2
     <answer>\\boxed{x=2}</answer>
    """
    ground_truth6 = "x=2"
    result6 = r1_zero_reward_fn(response6, ground_truth6)

    print("Test 6: Complex math problem with solution")
    print(f"Response: {response6.strip()}")
    print(f"Ground truth: {ground_truth6}")
    print(f"Result: {result6}")
    print("-" * 50)
    return


@app.cell
def _(r1_zero_reward_fn):

    # For a batch of responses
    responses = ["<answer>\\boxed{5}</answer>", "wrong answer"]
    ground_truths = ["5", "10"]

    for response, gt in zip(responses, ground_truths):
        reward = r1_zero_reward_fn(response, gt)
        print(f"Reward: {reward['reward']}")
    return


if __name__ == "__main__":
    app.run()
