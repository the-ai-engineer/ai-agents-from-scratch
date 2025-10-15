"""
Quick verification script to test Responses API integration.
This replicates the failing integration test pattern.

Run with: uv run python verify_responses_api.py
"""
from src.agent_sync import AgentSync
from dotenv import load_dotenv
import os

load_dotenv()


def test_multiple_tool_calls():
    """
    Replicates: tests/test_integration.py::TestRealAPIToolCalling::test_multiple_tool_calls
    """
    def get_weather() -> str:
        """Get the current weather."""
        return "sunny"

    def get_temperature() -> str:
        """Get the current temperature."""
        return "72 degrees"

    agent = AgentSync(system_prompt="Be concise.")
    agent.add_tools(get_weather, get_temperature)

    print("Testing multiple tool calls...")
    print(f"API Key set: {bool(os.getenv('OPENAI_API_KEY'))}")
    print()

    try:
        response = agent.run("What's the weather and temperature?")

        print("✅ Success!")
        print(f"Response: {response}")
        print()

        # Verify assertions
        assert "sunny" in response.lower(), "Expected 'sunny' in response"
        assert "72" in response, "Expected '72' in response"

        print("✅ All assertions passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {type(e).__name__}")
        print(f"Message: {str(e)[:200]}")

        if "401" in str(e) or "Incorrect API key" in str(e):
            print()
            print("This is an authentication error. To fix:")
            print("1. Get a valid API key from: https://platform.openai.com/account/api-keys")
            print("2. Update .env file: OPENAI_API_KEY=sk-proj-your-key-here")
            print("3. Run again: uv run python verify_responses_api.py")

        return False


if __name__ == "__main__":
    print("="*60)
    print("Responses API Verification")
    print("="*60)
    print()

    success = test_multiple_tool_calls()

    print()
    print("="*60)
    if success:
        print("✅ Responses API migration is working correctly!")
    else:
        print("⚠️  Update your API key to test with real OpenAI API")
    print("="*60)
