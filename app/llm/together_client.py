# together_client.py: The "Together AI" Client for LLM Inference
#
# This sends prompts to Together AI's cloud hosted LLMs and then returns generated text.
# Together AI is a service that hosts open source models with an
# OpenAI compatible API.
#
# The production model is "Qwen3-Next 80B-A3B-Instruct" (MoE: 80B total parameters, 3B
# active per token). This model was selected against Llama 3.3 70B because of higher citation recall,
# faster speed, and less cost.

"""This is the Together AI client for the LLM inference via an OpenAIcompatible API"""

# import libraries
from __future__ import annotations
import os
import time
from openai import OpenAI


class TogetherClient:
    """This class wraps the Together AI API

    It handles the API key validation, the client initialisation, and the prompt
    sending and returns generated text as well as performance metrics

    Usage:
    client = TogetherClient()
    if client.is_available():
    text, latency, in_tok, out_tok = client.generate(
    model="Qwen/Qwen3-Next-80B-A3B-Instruct",
    system_prompt="You are a regulatory compliance assistant...",
    user_prompt="What does GDPR Article 17 require?",
    )
    """

    # This is the API base URL for Together AI which replaces OpenAI's default endpoint
    BASE_URL = "https://api.together.xyz/v1"

    def __init__(self) -> None:
        # initialized on the first call to "generate()"
        self._client: OpenAI | None = None

    def is_available(self) -> bool:
        """This returns true if the TOGETHER_API_KEY is set and not empty"""
        return bool(os.getenv("TOGETHER_API_KEY"))

    def _get_client(self) -> OpenAI:
        """This returns OpenAI client and creates it on the first use

        Raises:
        ValueError: if TOGETHER_API_KEY isn't set
        """
        if self._client is None:
            key = os.getenv("TOGETHER_API_KEY", "")
            if not key:
                raise ValueError("TOGETHER_API_KEY not set")
            self._client = OpenAI(api_key=key, base_url=self.BASE_URL)
        return self._client

    def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[str, float, int, int]:
        """This sends the prompt to the LLM and returns the response generated

        Args:
        model: the Together AI model identifier (e.g., "Qwen/Qwen3-Next-80B-A3B-Instruct")
        system_prompt: this is the "system prompt" that sets the model's behaviour for the session
        user_prompt: user query
        max_tokens: the maximum amount of tokens to generate (default is 1024)
        temperature: the output randomness: 0.0 is the default for
                     reproducible answers for this project

        Returns:
        a tuple of (response_text, latency_ms, input_tokens, output_tokens)
        """
        client = self._get_client()

        start = time.perf_counter()  # "perf_counter" more accurate than "time.time()"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        # or "" protects against the content being None
        text = response.choices[0].message.content or ""

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return text, latency_ms, input_tokens, output_tokens
