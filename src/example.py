from agent import Agent

import logging

logger = logging.getLogger(__name__)


# Create agent with system prompt
agent = Agent(
    model="gpt-4o-mini",
    system_prompt=(
        "You are a helpful assistant with access to tools. "
        "Use these tools when needed to answer user questions accurately. "
        "Always cite which tool you used to get information."
    ),
    max_iterations=10,
)

agent.chat("What's the weather in Tokyo?")


def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    logger.info(f"Getting weather for {city}")
    return f"The weather in {city} is 21°C."


agent.register_tool(get_weather)

agent.chat("What's the weather in Tokyo?")
