# Arc Prize

code for [ARC Prize 2024](https://www.kaggle.com/competitions/arc-prize-2024/code?competitionId=67357&sortBy=voteCount&excludeNonAccessedDatasources=true)

[leaderboard](https://arcprize.org/leaderboard) for ARC prize

## results

| Dataset | Score         | Time Taken | cost |
| ------- | ------------- | ---------- | ---- |
| Train   | (38/400) 9.5% | 75m        | $200 |

## improvements

- output dimensions not matching
- import errors ('bfs') and missing variables in code
- prefix caching

## run

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/arcprize.git
   cd arcprize
   ```

2. **Install dependencies:**

   ```sh
   poetry install
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

You can run the code for different datasets (`train`, `eval`, `test`) and specify the number of samples to process.

### Train Dataset

To run predictions on the training dataset:

```sh
poetry run python arcprize/main.py --dataset train --n_samples 10
poetry run python arcprize/main.py --dataset eval --n_samples 5
poetry run python arcprize/main.py --dataset test
```

## Code Reference

The main script for running predictions is located in `arcprize/main.py`:

The configuration settings are defined in `arcprize/config.py`:

Helper functions are implemented in `arcprize/helpers.py`:

## links

### existing approaches

- [Getting 50% (SoTA) on ARC-AGI with GPT-4o](https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt)
  - [LessWrong post](https://www.lesswrong.com/posts/Rdwui3wHxCeKb7feK/getting-50-sota-on-arc-agi-with-gpt-4o)
  - [video](https://www.youtube.com/watch?v=z9j3wB1RRGA)

### code references

- [ARC 2024: Starter notebook + EDA](https://www.kaggle.com/code/allegich/arc-2024-starter-notebook-eda)
- [ARC Prize ChatGPT4o writes solution algorithms](https://www.kaggle.com/code/millernicholas/arc-prize-chatgpt4o-writes-solution-algorithms/notebook)

### prefix caching

- [Implementation — vLLM](https://docs.vllm.ai/en/latest/automatic_prefix_caching/details.html)

### interviews

- [Francois Chollet - LLMs won’t lead to AGI - $1,000,000 Prize to find true solution](https://www.youtube.com/watch?v=UakqL6Pj9xo)
