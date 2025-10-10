from openai import OpenAI, APIError, RateLimitError, APIConnectionError

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure retries when creating the client
client = OpenAI(
    max_retries=3,  # Number of retries for failed requests
    timeout=60.0,  # Timeout for requests
)

try:
    response = client.responses.create(
        model="gpt-4o-mini", input="Write a haiku about AI."
    )
    logger.info(response.output_text)
except RateLimitError:
    logger.info("Rate limit exceeded. Retrying...")
    pass
except APIConnectionError:
    logger.info("Network error. Retrying...")
    pass
except APIError as e:
    logger.info(f"API error: {e}")
    pass

# INFO:httpx:HTTP Request: POST https://api.openai.com/v1/responses "HTTP/1.1 503 Service Unavailable"
# INFO:openai._base_client:Retrying request to /responses in 0.430739 seconds
# INFO:httpx:HTTP Request: POST https://api.openai.com/v1/responses "HTTP/1.1 200 OK"

# INFO:__main__:Silent circuits hum,
# In the depths of thought they dream,
# Mind of metal blooms.
