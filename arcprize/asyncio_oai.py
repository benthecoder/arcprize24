import asyncio
import os
from timeit import default_timer as timer

from dotenv import load_dotenv
from openai import AsyncOpenAI


async def call_openai(client, id, model, theme, answers):
    print(f"Generating a story about {theme} using {id}.")
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Generate a one-paragraph story about a {theme}.",
            },
        ],
    )
    answers.append(response.choices[0].message.content)
    print(f"Generated a story about {theme} using {id}.")


async def main():
    load_dotenv()
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    themes = ["dog", "cat", "chicken", "tiger"]
    model1 = "gpt-4o"
    model2 = "gpt-4o"
    id1 = "client-1"
    id2 = "client-2"

    answers_s = []
    start = timer()
    # Run sequentially
    for i in range(0, len(themes), 2):
        start_step = timer()
        await call_openai(client, id1, model1, themes[i], answers_s)
        await call_openai(client, id2, model2, themes[i + 1], answers_s)
        end_step = timer()
        print(
            f"Finished generating stories about a {themes[i]} and a {themes[i+1]} sequentially in {end_step - start_step:.2f} seconds."
        )
    end = timer()
    print(f"Generated stories sequentially in {end - start:.2f} seconds.")

    answers_c = []
    start = timer()
    for i in range(0, len(themes), 2):
        start_step = timer()
        async with asyncio.TaskGroup() as tg:
            print(
                f"Started generating stories about {themes[i]} and {themes[i+1]} concurrently."
            )
            tg.create_task(call_openai(client, id1, model1, themes[i], answers_c))
            tg.create_task(call_openai(client, id2, model2, themes[i + 1], answers_c))
        end_step = timer()
        print(
            f"Finished generating stories about a {themes[i]} and a {themes[i+1]} concurrently in {end_step - start_step:.2f} seconds."
        )
    end = timer()
    print(f"Generated stories concurrently in {end - start:.2f} seconds.\n\n")


if __name__ == "__main__":
    asyncio.run(main())
