import argparse
import asyncio
import json
import logging
import os
import random

from dotenv import load_dotenv
from openai import AsyncOpenAI

from arcprize.config import (
    BASE_RESPONSE,
    MODEL,
    RETRY_ATTEMPTS,
    SUBMISSION_FILE_NAME,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    USER_PROMPT_1,
    USER_PROMPT_2,
)
from arcprize.helpers import (
    create_submission_file,
    exec_response,
    extract_reasoning_from_response,
    generate_user_prompt,
    load_data,
    plot_prediction,
    score_submission,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("arcprize.log"), logging.StreamHandler()],
)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI()


async def fix_code_with_llm(broken_code, error_message, sample):
    fix_user_prompt = f"""
    The following code has an error:
    ```python
    {broken_code}
    ```
    The error message is:
    {error_message}

    Reason carefully and fix your algorithm so it matches the expected output.

    Here is the input again:
    {sample["test"][0]["input"]}
    
    Use the following template for your algorithm:

    ```python
    import numpy as np

    # Your thought process
    def apply_transformation(input_matrix):
        # perform transformation
        ...
        return output_matrix
    ```

    respond with only the reasoning and the fixed code.
    """

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": fix_user_prompt},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    fixed_code = response.choices[0].message.content
    reasoning = extract_reasoning_from_response(fixed_code)
    logging.info(f"Reasoning: {reasoning}")
    return fixed_code


def execute_code(code, input_data):
    try:
        output, executed_code = exec_response(code, input_data)
        return output, executed_code, None
    except Exception as e:
        return None, None, str(e)


async def handle_code_execution(resp, sample):
    input_data = sample["test"][0]["input"]
    output, code, error = execute_code(resp, input_data)

    if error:
        logging.error(f"Initial code execution failed: {error}")
        fixed_code = await fix_code_with_llm(resp, error, sample)
        output, code, error = execute_code(fixed_code, input_data)

    return output, code, error


async def get_task_prediction(
    sample,
    retry_attempts=RETRY_ATTEMPTS,
    system_prompt=SYSTEM_PROMPT,
    user_prompt=USER_PROMPT_1,
):
    user_prompt = generate_user_prompt(sample, user_prompt)

    for attempt in range(retry_attempts):
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            resp = response.choices[0].message.content
            reasoning = extract_reasoning_from_response(resp)
            logging.info(f"Reasoning: {reasoning}")

            output, code, error = await handle_code_execution(resp, sample)

            if error:
                logging.error(f"Attempt {attempt + 1} failed with error: {error}")
                continue

            task_out_dim = (
                len(sample["train"][0]["output"]),
                len(sample["train"][0]["output"][0]),
            )
            pred_out_dim = (len(output), len(output[0]))

            if task_out_dim == pred_out_dim:
                logging.info(f"Task output dimension: {task_out_dim}")
                logging.info(f"Predicted output dimension: {pred_out_dim}")
                return output
            else:
                logging.error(
                    f"Output dimension mismatch: Expected {task_out_dim}, Got {pred_out_dim}"
                )
                fixed_code = await fix_code_with_llm(
                    resp,
                    f"Output dimension mismatch: Expected {task_out_dim}, Got {pred_out_dim}",
                    sample,
                )
                output, code, error = execute_code(
                    fixed_code, sample["test"][0]["input"]
                )

                if not error and task_out_dim == (len(output), len(output[0])):
                    return output

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed with error: {e}")

    logging.error("Failed to get correct output dimensions after multiple attempts")
    return BASE_RESPONSE


async def main(dataset, n_samples):
    data = load_data(dataset)
    results = {}

    if dataset == "train":
        data_samples = data["train"]
        solutions = data["train_solutions"]
    elif dataset == "eval":
        data_samples = data["eval"]
        solutions = data["eval_solutions"]
    elif dataset == "test":
        data_samples = data["test"]
        solutions = None
    else:
        raise ValueError("Invalid dataset. Choose from 'train', 'eval', or 'test'.")

    if n_samples:
        sampled_items = random.sample(list(data_samples.items()), n_samples)
    else:
        sampled_items = list(data_samples.items())

    semaphore = asyncio.Semaphore(10)

    async def process_task(task_id, sample):
        async with semaphore:
            logging.info(f"Predicting attempt for #{task_id}")
            try:
                output1 = await get_task_prediction(sample, user_prompt=USER_PROMPT_1)
                output2 = await get_task_prediction(sample, user_prompt=USER_PROMPT_2)

                results[task_id] = [{"attempt_1": output1, "attempt_2": output2}]
            except Exception as e:
                logging.error(f"Error for task {task_id}: {e}")

    tasks = [process_task(task_id, sample) for task_id, sample in sampled_items]
    await asyncio.gather(*tasks)

    logging.info("saving results...\n")

    if results is None:
        logging.error("No results found.")
        return

    plot_prediction(data_samples, solutions, results, dataset)
    create_submission_file(results, SUBMISSION_FILE_NAME)

    if solutions:
        score_result = score_submission(solutions)
        logging.info(
            f"Final score: {score_result['total_score']} of {score_result['total_tasks_scored']} ({round(score_result['total_score']/score_result['total_tasks_scored'] * 100, 2)}%)"
        )
        with open("score.json", "w") as f:
            json.dump(score_result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARC Prize predictions.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["train", "eval", "test"],
        required=True,
        help="Dataset to run predictions on (train, eval, test).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help="Number of samples to run predictions on. If not provided, run on the entire dataset.",
    )
    args = parser.parse_args()

    asyncio.run(main(args.dataset, args.n_samples))
