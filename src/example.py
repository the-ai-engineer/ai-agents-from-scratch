"""
Example showing how to use the Agent class with tools.
"""
from src.agent import Agent
import asyncio


async def main():
    # Define a tool
    def get_weather(city: str) -> str:
        """Get the weather for a given city."""
        return f"The weather in {city} is 21°C."

    # Create agent with system prompt
    agent = Agent(
        model="gpt-4o-mini",
        system_prompt=(
            "You are a helpful assistant with access to tools. "
            "Use these tools when needed to answer user questions accurately. "
            "Always cite which tool you used to get information."
        ),
    )

    # First try without tools - agent won't be able to answer
    print("=== Without tools ===")
    response1 = await agent.run("What's the weather in Tokyo?")
    print(response1)
    print()

    # Now add the tool
    agent.add_tool(get_weather)

    # Reset conversation and try again with tools
    agent.reset()
    print("=== With tools ===")
    response2 = await agent.run("What's the weather in Tokyo?")
    print(response2)


if __name__ == "__main__":
    asyncio.run(main())
