# OpenAIIntegration

`OpenAIIntegration` is a Python class that provides a robust, high-level wrapper around the OpenAI API, including synchronous and asynchronous clients. It handles completions, embeddings, fine-tuning, assistants, threads, messages, and image generation, with built-in retry logic and token counting utilities.

---

## Features

- **Completion API**: Synchronous and asynchronous completions with optional retry handling.
- **Embeddings**: Generate embeddings for text with retry support.
- **Token Counting**: Calculate token usage for messages compatible with various OpenAI models.
- **File Management**: Upload and delete files for fine-tuning or other purposes.
- **Fine-Tuning Jobs**: Start fine-tuning jobs for custom models.
- **Assistants & Threads**: Create, manage, and delete assistants and threads for structured multi-turn conversations.
- **Message Handling**: Create and retrieve messages in threads.
- **Run Management**: Create and retrieve assistant runs.
- **Image Generation**: Generate images via DALL-E with base64 decoding.
- **Retries & Robustness**: Uses exponential backoff and tenacity for handling connection errors, rate limits, and timeouts.

---

## Installation

Ensure you have the required dependencies installed:

```bash
pip install openai tenacity tiktoken


## Usage
```
from openai_integration import OpenAIIntegration

# Initialize integration
ai = OpenAIIntegration()

# Check connection
ai.check_openai_connection()

# Get a chat completion
completion = ai.completion_with_retry(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(completion)

# Generate embeddings
embedding = ai.embedding_with_retry(
    model="text-embedding-3-small",
    input="OpenAI API integration"
)
print(embedding)

# Count tokens
tokens = ai.num_tokens_for_message("Hello world")
print(f"Number of tokens: {tokens}")

# Upload a file for fine-tuning
with open("data.jsonl", "rb") as f:
    file_info = ai.upload_file(f, purpose="fine-tune")
print(file_info)

# Generate an image
image_bytes = ai.generate_image(
    model="dall-e-3",
    prompt="A futuristic city skyline",
    size="1024x1024",
    quality="high",
    style="realistic"
)
with open("output.png", "wb") as f:
    f.write(image_bytes)
```