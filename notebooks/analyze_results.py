import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import json
    from collections import defaultdict

    # Load results
    results = []
    with open('../outputs/baseline/results.jsonl', 'r') as f:
        for line in f:
            results.append(json.loads(line))

    # Categorize results
    categories = {
        'correct_both': [],
        'format_1_answer_0': [],
        'format_0_answer_0': []
    }

    for result in results:
        format_reward = result.get('format_reward', 0)
        answer_reward = result.get('answer_reward', 0)
    
        if format_reward == 1 and answer_reward == 1:
            categories['correct_both'].append(result)
        elif format_reward == 1 and answer_reward == 0:
            categories['format_1_answer_0'].append(result)
        elif format_reward == 0 and answer_reward == 0:
            categories['format_0_answer_0'].append(result)

    # Print counts
    print(f"(1) Correct with both rewards = 1: {len(categories['correct_both'])}")
    print(f"(2) Format reward = 1, Answer reward = 0: {len(categories['format_1_answer_0'])}")
    print(f"(3) Format reward = 0, Answer reward = 0: {len(categories['format_0_answer_0'])}")

    # Examine format_reward = 0 cases
    print("\n--- Format Reward = 0 Cases (first 10) ---")
    for i, case in enumerate(categories['format_0_answer_0'][:10]):
        print(f"\nCase {i+1}:")
        print(f"Q: {case.get('prompt', '')}")
        print(f"A: {case.get('generation', '')}")
        print(f"GT: {case.get('expected_answer', '')}")

    # Examine format_reward = 1, answer_reward = 0 cases
    print("\n--- Format = 1, Answer = 0 Cases (first 10) ---")
    for i, case in enumerate(categories['format_1_answer_0'][:10]):
        print(f"\nCase {i+1}:")
        print(f"Q: {case.get('prompt', '')}")
        print(f"A: {case.get('generation', '')}")
        print(f"GT: {case.get('expected_answer', '')}")
    return


if __name__ == "__main__":
    app.run()
