import json
import logging
import re
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from matplotlib import colors

from arcprize.config import DATA_PATHS_MAP, MODEL, SUBMISSION_FILE_NAME

logging.basicConfig(level=logging.INFO)


#####################
# I/O
#####################


def load_data(dataset: str):
    if dataset not in DATA_PATHS_MAP:
        raise ValueError("Invalid dataset. Choose from 'train', 'eval', or 'test'.")

    data = {}
    with open(DATA_PATHS_MAP[dataset], "r") as f:
        text = f.read()
        data[dataset] = json.loads(text)
        logging.info(f"Loaded {len(data[dataset])} lines of {dataset} data")

    if dataset in ["train", "eval"]:
        solutions_key = f"{dataset}_solutions"
        with open(DATA_PATHS_MAP[solutions_key], "r") as f:
            text = f.read()
            data[solutions_key] = json.loads(text)
            logging.info(
                f"Loaded {len(data[solutions_key])} lines of {solutions_key} data"
            )

    return data


def json_task_to_string(task: dict) -> str:
    train_tasks = task["train"]
    test_task = task["test"]

    final_output = ""
    final_output = "Training Examples\n"

    for i, task in enumerate(train_tasks):
        final_output += f"Example {i + 1}: Input\n["
        for row in task["input"]:
            final_output += f"\n{str(row)},"

        final_output += "]\n\n"
        final_output += f"Example {i + 1}: Output\n["

        for row in task["output"]:
            final_output += f"\n{str(row)},"

        final_output += "]\n\n"

    final_output += "Test\n["
    for row in test_task[0]["input"]:
        final_output += f"\n{str(row)}"

    final_output += "]"

    return final_output


def create_submission_file(submission, file_name="submission.json"):
    """
    Save a submission file to the specified file name
    """

    def convert_ndarray(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(file_name, "w") as file:
        json.dump(submission, file, default=convert_ndarray)

    print(f"Submission saved to {file_name}")


#####################
# AI
#####################


def count_string_tokens(prompt: str, model: MODEL) -> int:
    """
    Returns the number of tokens in a (prompt or completion) text string.

    Args:
        prompt (str): The text string
        model_name (str): The name of the encoding to use. (e.g., "gpt-3.5-turbo")

    Returns:
        int: The number of tokens in the text string.
    """
    model = model.lower()
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logging.warning("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(prompt))


def generate_user_prompt(input_string, user_prompt):
    return user_prompt.replace("{input_string}", json_task_to_string(input_string))


def extract_reasoning_from_response(response):
    """Extracts <reasoning> tag from response"""
    pattern = r"<reasoning>(.*?)</reasoning>"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        raise ValueError("No reasoning found in response.")

    return matches[0]


def exec_response(response, input_matrix):
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        raise ValueError("No Python code found in response.")

    code = matches[0] + "\nresult = apply_transformation(input_matrix)"

    # Define a function to dynamically import modules
    def dynamic_import(module_name):
        components = module_name.split(".")
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod

    # List of modules that might be needed
    required_modules = {
        "np": "numpy",
        "scipy": "scipy",
        "Counter": "collections.Counter",
        # Add more modules here as needed
    }

    # Dynamically import the required modules
    global_scope = {
        name: dynamic_import(module) for name, module in required_modules.items()
    }
    local_scope = {"input_matrix": input_matrix}

    try:
        exec(code, global_scope, local_scope)
        result = local_scope.get("result", None)
        return result, code
    except Exception as e:
        raise ValueError(f"Failed to run the code: {e}")


def score_submission(solutions) -> Tuple[float, int]:
    # source: https://www.kaggle.com/code/gregkamradt/using-frontier-models-on-arc-agi-via-langchain?scriptVersionId=190345835

    """
    submission_file_name: str, the file name of your submission file
    solutions: dict, the ground truth solutions you'd like to test against

    Read a submission from file, score it, then return the score
    """
    print(f"Scoring {SUBMISSION_FILE_NAME}\n")

    # Open your submission file
    with open(SUBMISSION_FILE_NAME, "r") as file:
        submission = json.load(file)

    total_score = 0
    total_tasks = 0

    # Loop through each task in your submission to grade it
    for task_id, task_submission in submission.items():
        total_tasks += 1
        task_score = 0
        num_pairs = len(task_submission)

        # Go through each task. Most will only have 1
        for pair_index, pair_attempts in enumerate(task_submission):
            pair_correct = False

            # Look at both of your attempts
            for attempt_key, attempt in pair_attempts.items():
                try:
                    if attempt == solutions[task_id][pair_index]:
                        logging.info(
                            f"Task Id {task_id} pair {pair_index+1} {attempt_key} matches solution"
                        )
                        pair_correct = True
                        break
                except IndexError as e:
                    logging.info(
                        f"IndexError: {e} for task_id {task_id} and pair_index {pair_index}"
                    )
                    logging.info(f"solutions[task_id]: {solutions.get(task_id)}")
                    logging.info(f"task_submission: {task_submission}")
                    raise

            if pair_correct:
                task_score += 1

        task_score /= num_pairs
        total_score += task_score

        logging.info(f"Task {task_id} score: {task_score}")

    return {"total_score": total_score, "total_tasks_scored": total_tasks}


#####################
# plotting
#####################

cmap = colors.ListedColormap(
    [
        "#000000",
        "#0074D9",
        "#FF4136",
        "#2ECC40",
        "#FFDC00",
        "#AAAAAA",
        "#F012BE",
        "#FF851B",
        "#7FDBFF",
        "#870C25",
    ]
)
norm = colors.Normalize(vmin=0, vmax=9)


def plot_task(task, task_solutions, i, t):
    """Plots the train and test pairs of a specified task,
    using same color scheme as the ARC app"""

    num_train = len(task["train"])
    num_test = len(task["test"])

    w = num_train + num_test
    fig, axs = plt.subplots(2, w, figsize=(3 * w, 3 * 2))
    plt.suptitle(f"Set #{i+1}, {t}:", fontsize=20, fontweight="bold", y=1)

    for j in range(num_train):
        plot_one(task, axs[0, j], j, "train", "input")
        plot_one(task, axs[1, j], j, "train", "output")

    plot_one(task, axs[0, j + 1], 0, "test", "input")

    answer = task_solutions
    input_matrix = answer

    axs[1, j + 1].imshow(input_matrix, cmap=cmap, norm=norm)
    axs[1, j + 1].grid(True, which="both", color="lightgrey", linewidth=0.5)
    axs[1, j + 1].set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])
    axs[1, j + 1].set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    axs[1, j + 1].set_xticklabels([])
    axs[1, j + 1].set_yticklabels([])
    axs[1, j + 1].set_title("TEST OUTPUT", color="green", fontweight="bold")

    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor("black")  # substitute 'k' for black
    fig.patch.set_facecolor("#dddddd")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_one(task, ax, i, train_or_test, input_or_output):
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which="both", color="lightgrey", linewidth=0.5)

    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])
    ax.set_title(train_or_test + " " + input_or_output, fontweight="bold")


def plot_prediction(data_samples, solutions, results, dataset):
    """Plots the test input, expected output, and predicted output."""
    for sample_key, result in results.items():
        if sample_key not in data_samples:
            logging.error(f"Sample key {sample_key} not found in data_samples")
            continue

        test_input = data_samples[sample_key]["test"][0]["input"]
        if dataset == "train":
            expected_output = solutions[sample_key][0]
        elif dataset == "eval":
            expected_output = solutions[sample_key][0]
        else:
            expected_output = None

        predicted_output_1 = result[0]["attempt_1"]
        predicted_output_2 = result[0]["attempt_2"]

        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        plt.suptitle(f"Sample {sample_key}", fontsize=20, fontweight="bold")

        # Plot test input
        plot_matrix(axs[0], test_input, title="Test Input")
        # Plot expected output if available
        if expected_output is not None:
            plot_matrix(axs[1], expected_output, title="Expected Output")
        # Plot predicted outputs
        plot_matrix(axs[2], predicted_output_1, title="Predicted Output 1")
        plot_matrix(axs[3], predicted_output_2, title="Predicted Output 2")

        plt.tight_layout()
        # Save the plot
        plt.savefig(f"output/sample_{sample_key}.png")
        plt.close(fig)


def plot_matrix(ax, matrix, title=""):
    """Helper function to plot a single matrix with title."""
    ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, which="both", color="lightgrey", linewidth=0.5)
    ax.set_xticks([x - 0.5 for x in range(1 + len(matrix[0]))])
    ax.set_yticks([x - 0.5 for x in range(1 + len(matrix))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
