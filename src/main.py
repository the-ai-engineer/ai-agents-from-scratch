from agent import Agent
from tool import Tool

agent = Agent()
agent.register_tool(Tool.from_function(get_weather))
agent.chat("What's the weather in Tokyo?")
