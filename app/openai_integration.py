from typing import Optional, Any, Literal
import openai
import tenacity
import traceback
import tiktoken
import json
import base64
from openai import OpenAI, AsyncOpenAI
from pprint import pformat

from models.llm_message import LLMMessage
from app.utils.logger import get_logger
from app.core.config import settings

OpenAIUploadFilePurpose = Literal[
    "assistants",
    "batch",
    "fine-tune",
    "vision",
    "user_data",
    "evals"
]

MODELS = {
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o-mini",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4-0314",
    "gpt-4-32k-0314",
    "gpt-4-0613",
    "gpt-4-32k-0613",
}

MessageInput = (
    dict[str, Any]
    | tuple[str, str | dict[str, Any]]
    | LLMMessage
)

class OpenAIIntegration:
    logger = get_logger("OpenAIIntegration")
    completion_timeout = 60
    embedding_timeout = 10

    track_completion_times = False
    track_embedding_times = False

    def __init__(self):
        self.config = settings
        self.enabled = False
        self.client: Optional[OpenAI] = None
        self.async_client: Optional[AsyncOpenAI] = None

        if self.config.OPENAI_API_KEY is not None:
            self.client = OpenAI(api_key=self.config.api_key, max_retries=0)
            self.async_client = AsyncOpenAI(api_key=self.config.api_key, max_retries=0)
            self.enabled = True
        else:
            self.enabled = False

        try:
            self.tokenizer_encoding = tiktoken.encoding_for_model("gpt-4.1-nano")
        except KeyError:
            self.tokenizer_encoding = tiktoken.get_encoding("cl100k_base")

    def check_openai_connection(self):
        if not self.async_client:
            raise RuntimeError("OpenAI async client not initialized")
        models = self.client.models.list()
        # Some SDKs return an object with .data; guard either way:
        count = len(getattr(models, "data", models))
        self.logger.info(f"Connected. Models: {count}")

    def completion(self, *args, **kwargs):
        """
        This function gets a completion from openai, without any retries.
        """
        try:
            result = self.client.chat.completions.create(*args, **kwargs)
            if "stream" in kwargs and kwargs["stream"]:
                # TODO: We shouldn't be ignoring completion usage just because we are streaming
                return result
            else:
                return  result
        except openai.APITimeoutError as e:
            raise e
        except Exception as e:
            self.logger.warning(f"Warning, a error fetching completion for input {pformat(args)}, "
                                f"{pformat(kwargs, width=120, compact=True)}:\n{traceback.format_exception(e)}. "
                                f"Will retry if attempts not exhausted.")

    @tenacity.retry(
        wait=tenacity.wait_exponential_jitter(initial=2, exp_base=2),
        retry_error_callback=lambda state: OpenAIIntegration.logger.warning(
            f"Warning, error received while fetching completion for prompt {pformat(state.args)}, "
            f"{pformat(state.kwargs)}:\n{traceback.format_exception(state.outcome.exception())}. "
            f"Will retry if attempts not exhausted."
        ),
        stop=tenacity.stop_after_attempt(8),
        retry=tenacity.retry_if_exception_type(
            (
                openai.APIConnectionError,
                openai.InternalServerError,
                openai.RateLimitError,
                openai.APITimeoutError,
            )
        ),
    )
    def completion_with_retry(self, *args, **kwargs):
        """
        This function handles retries for the openai.Completion.create() function, allowing us
        to smoothly handle situations where there is connection errors or rate limiting errors
        """
        kwargs["timeout"] = self.completion_timeout

        return self.completion(*args, **kwargs)

    def parse_completion(self, *args, **kwargs):
        """
        This function gets a completion using a structured response.
        """
        try:
            result = self.client.beta.chat.completions.parse(*args, **kwargs)

            if "stream" in kwargs and kwargs["stream"]:
                # TODO: We shouldn't be ignoring completion usage just because we are streaming
                return result
            else:
                return result
        except openai.APITimeoutError as e:
            # Pass through without logging it. This is a common error seen when using OpenAI APIs.
            raise e
        except Exception as e:
            self.logger.warning(f"Warning, a error fetching completion for input {pformat(args)}, "
                                f"{pformat(kwargs, width=120, compact=True)}:\n{traceback.format_exception(e)}. "
                                f"Will retry if attempts not exhausted.")


    @tenacity.retry(
        wait=tenacity.wait_exponential_jitter(initial=2, exp_base=2),
        retry_error_callback=lambda state: OpenAIIntegration.logger.warning(
            f"Warning, error received while parsing completion for prompt {pformat(state.args)}, "
            f"{pformat(state.kwargs)}:\n{traceback.format_exception(state.outcome.exception())}. "
            f"Will retry if attempts not exhausted."
        ),
        stop=tenacity.stop_after_attempt(8),
        retry=tenacity.retry_if_exception_type(
            (
                openai.APIConnectionError,
                openai.InternalServerError,
                openai.RateLimitError,
                openai.APITimeoutError,
            )
        ),
    )
    def parse_completion_with_retry(self, *args, **kwargs):
        """
        This function handles retries for the openai.Completion.create() function, allowing us
        to smoothly handle situations where there is connection errors or rate limiting errors
        """
        kwargs["timeout"] = self.completion_timeout

        return self.parse_completion(*args, **kwargs)

    def embedding(self, *args, **kwargs):
        """
        This function handles fetches an embedding, without any retries
        """
        try:
            embedding = self.client.embeddings.create(*args, **kwargs)
            return embedding
        except openai.APITimeoutError as e:
            raise e
        except Exception as e:
            self.logger.warning(f"Warning, a error fetching embedding for text {pformat(args)}, "
                                f"{pformat(kwargs, width=120, compact=True)}:\n{traceback.format_exception(e)}. "
                                f"Will retry if attempts not exhausted.")


    @tenacity.retry(
        wait=tenacity.wait_exponential_jitter(initial=2, exp_base=2),
        retry_error_callback=lambda state: OpenAIIntegration.logger.warning(
            f"Warning, error received while fetching embedding for text {pformat(state.args)}, "
            f"{pformat(state.kwargs)}:\n{traceback.format_exception(state.outcome.exception())}. "
            f"Will retry if attempts not exhausted."
        ),
        stop=tenacity.stop_after_attempt(8),
        retry=tenacity.retry_if_exception_type(
            (
                openai.APIConnectionError,
                openai.InternalServerError,
                openai.RateLimitError,
                openai.APITimeoutError,
            )
        ),
    )
    def embedding_with_retry(self, *args, **kwargs):
        """
        This function handles retries for the openai.Embedding.create() function, allowing us
        to smoothly handle situations where there is connection errors or rate limiting errors
        """
        kwargs["timeout"] = self.embedding_timeout

        return self.embedding(*args, **kwargs)

    def num_tokens_for_message(self, message: Optional[list | dict | str]) -> int:
        """
        Returns the number of tokens used by a message. This uses a method originally described on the OpenAI
        website here, although they don't say its exact and that it may change as the models evolve.

        This method is also capable of taking in None as a message, and will return 1 token for that.

        If message is a dict or a list (such as with tools or some such), they will be encoded into JSON
        and the tokens will be counted for that.
        """
        if message is None:
            return 1
        elif isinstance(message, dict) or isinstance(message, list):
            message = json.dumps(message)

        # 4 tokens are just used by default on every chat completion, no matter what.
        num_tokens = 4
        num_tokens += len(self.tokenizer_encoding.encode(message))

        return num_tokens

    def upload_file(self, file_stream, purpose: OpenAIUploadFilePurpose = "fine-tune"):
        return self.client.files.create(file=file_stream, purpose=purpose)

    def delete_file(self, file_id):
        return self.client.files.delete(file_id=file_id)

    def start_fine_tuning_job(self, training_file_id, model_name="gpt-4.1-nano"):
        return self.client.fine_tuning.jobs.create(training_file=training_file_id, model=model_name)

    def create_assistant(self, **params):
        return self.client.beta.assistants.create(**params)

    def delete_assistant(self, assistant_id):
        return self.client.beta.assistants.delete(assistant_id)

    def create_thread(self):
        return self.client.beta.threads.create()

    def delete_thread(self, thread_id):
        # Quick hack for tests. Don't attempt to send a delete request for any test thread its.
        if thread_id.startswith("test_"):
            return None
        return self.client.beta.threads.delete(thread_id)

    def create_message(self, thread_id, role, content):
        return self.client.beta.threads.messages.create(thread_id=thread_id, role=role, content=content)

    def create_run(self, thread_id, assistant_id, **kwargs):
        return self.client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id, **kwargs)

    def retrieve_run(self, run_id, thread_id):
        return self.client.beta.threads.runs.retrieve(run_id=run_id, thread_id=thread_id)

    def retrieve_messages(self, **params):
        return self.client.beta.threads.messages.list(**params)

    def num_tokens_from_messages(self, messages: MessageInput | list[MessageInput], model: str = "gpt-4-1106-preview") -> int:
        # Accept single message
        if not isinstance(messages, list):
            messages = [messages]

        # Normalize into OpenAI-style dicts
        normalized = []
        for i, msg in enumerate(messages):
            if isinstance(msg, LLMMessage):  # Pydantic v2
                msg = msg.model_dump()

            # Accept (role, content) or (role, name, content)
            if isinstance(msg, tuple):
                if len(msg) == 2:
                    role, content = msg
                    msg = {"role": role, "content": content}
                elif len(msg) == 3:
                    role, name, content = msg
                    msg = {"role": role, "name": name, "content": content}
                else:
                    raise TypeError(f"Message #{i}: unsupported tuple len={len(msg)}")

            if not isinstance(msg, dict):
                raise TypeError(f"Message #{i}: expected dict/LLMMessage/tuple, got {type(msg).__name__}")

            # Ensure required keys (allow tool/func shapes)
            if "role" not in msg or "content" not in msg:
                if "tool_calls" in msg or "function_call" in msg:
                    msg.setdefault("role", "assistant")
                    msg.setdefault("content", "")
                else:
                    raise ValueError(f"Message #{i}: missing 'role'/'content': keys={list(msg.keys())}")

            normalized.append(msg)

        messages = normalized

        # --- encoding selection (your existing logic) ---
        try:
            encoding = tiktoken.encoding_for_model(model_name=model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        if model in MODELS:
            token_per_message, tokens_per_name = 3, 1
        elif model == "gpt-3.5-turbo-0301":
            token_per_message, tokens_per_name = 4, -1
        elif "gpt-3.5-turbo" in model:
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            # If you keep this redirect, it will re-enter this function with dicts already normalized
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")
        elif "gpt-5" in model:
            token_per_message, tokens_per_name = 3, 1
        else:
            raise NotImplementedError(
                f"num_tokens_from_messages not implemented for {model}. See chatml.md."
            )

        # --- robust counting ---
        num_tokens = 0
        for m in messages:
            num_tokens += token_per_message
            for k, v in m.items():
                if isinstance(v, (list, dict)):
                    # Safer JSON dump in case any inner object isn’t natively serializable
                    num_tokens += len(encoding.encode(json.dumps(v, default=str)))
                elif v is None:
                    num_tokens += 1
                else:
                    if not isinstance(v, str):
                        v = str(v)
                    num_tokens += len(encoding.encode(v))
                    if k == "name":
                        num_tokens += tokens_per_name

        num_tokens += 3  # reply priming
        return num_tokens

    @tenacity.retry(
        wait=tenacity.wait_exponential_jitter(initial=2, exp_base=2),
        retry_error_callback=lambda state: OpenAIIntegration.logger.warning(
            f"Warning, error received while generating image {pformat(state.args)}, "
            f"{pformat(state.kwargs)}:\n{traceback.format_exception(state.outcome.exception())}. "
            f"Will retry if attempts not exhausted."
        ),
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception_type(
            (
                openai.APIConnectionError,
                openai.InternalServerError,
                openai.RateLimitError,
                openai.APITimeoutError,
            )
        ),
    )
    def generate_image(self, model, prompt, size, quality, style) -> bytes:
        """Generates an image using DALL-E."""
        response = self.client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            response_format="b64_json",
            n=1,
        )

        # Convert the response data into a bytes object
        image_data_b64 = response.data[0].b64_json
        image_data_bytes = base64.b64decode(image_data_b64)

        return image_data_bytes
