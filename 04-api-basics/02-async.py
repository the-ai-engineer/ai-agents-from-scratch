from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import nest_asyncio
import time

nest_asyncio.apply()
load_dotenv()

async_client = AsyncOpenAI()

prompts = ["What is Python?", "What is JavaScript?", "What is Rust?"]


async def run_parallel():
    """Run requests in parallel"""
    start_time = time.perf_counter()

    tasks = [
        async_client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            temperature=0,
        )
        for prompt in prompts
    ]

    responses = await asyncio.gather(*tasks)

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print("=== PARALLEL EXECUTION ===")
    for prompt, response in zip(prompts, responses):
        print(f"\n{prompt}")
        print(f"→ {response.output_text[:100]}...")

    print(f"\n⏱️  Parallel time: {elapsed:.2f} seconds")
    return elapsed


async def run_sequential():
    """Run requests one by one"""
    start_time = time.perf_counter()

    responses = []
    for prompt in prompts:
        response = await async_client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            temperature=0,
        )
        responses.append(response)

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print("\n=== SEQUENTIAL EXECUTION ===")
    for prompt, response in zip(prompts, responses):
        print(f"\n{prompt}")
        print(f"→ {response.output_text[:100]}...")

    print(f"\n⏱️  Sequential time: {elapsed:.2f} seconds")
    return elapsed


async def main():
    # Run both and compare
    parallel_time = await run_parallel()
    sequential_time = await run_sequential()

    print("\n" + "=" * 40)
    print(f"Speedup: {sequential_time / parallel_time:.2f}x faster")
    print(f"Time saved: {sequential_time - parallel_time:.2f} seconds")


# Run the comparison
if __name__ == "__main__":
    asyncio.run(main())
