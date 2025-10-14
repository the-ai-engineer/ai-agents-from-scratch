"""
Test script to verify your development environment is configured correctly.

Run from project root:
    uv run python 03-development-setup/test_setup.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# Load environment variables
load_dotenv()

print("="* 60)
print("Testing Development Environment Setup")
print("="* 60)

##=================================================##
## Test 1: Check Python version
##=================================================##

import sys
print(f"\n✓ Python version: {sys.version.split()[0]}")
if sys.version_info < (3, 10):
    print("⚠ Warning: Python 3.10+ recommended")

##=================================================##
## Test 2: Check API key is configured
##=================================================##

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"✓ API key found: {api_key[:7]}...{api_key[-4:]}")
else:
    print("✗ OPENAI_API_KEY not found in environment")
    print("  Create a .env file with: OPENAI_API_KEY=sk-your-key")
    sys.exit(1)

##=================================================##
## Test 3: Test API connection
##=================================================##

print("\nTesting OpenAI API connection...")
try:
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model="gpt-4o-mini",
        input="Say 'Hello, AI Engineer!' if you can read this.",
        temperature=0
    )
    print(f"✓ API working: {response.output_text}")
except Exception as e:
    print(f"✗ API error: {e}")
    sys.exit(1)

##=================================================##
## Test 4: Test tokenization
##=================================================##

print("\nTesting tokenization...")
try:
    text = "This is a test message for tokenization."
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = len(encoding.encode(text))
    print(f"✓ Tokenization working")
    print(f"  Text: '{text}'")
    print(f"  Tokens: {tokens}")
except Exception as e:
    print(f"✗ Tokenization error: {e}")
    sys.exit(1)

##=================================================##
## Summary
##=================================================##

print("\n" + "="* 60)
print("✓ All tests passed! Your environment is ready.")
print("="* 60)
print("\nNext steps:")
print("  1. Head to lesson 04-api-basics")
print("  2. Make your first API call")
print("  3. Start building!")
