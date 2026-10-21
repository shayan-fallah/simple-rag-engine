from typing import Any, Dict, List, Type, TypeVar

from anthropic import Anthropic
from openai import OpenAI
from pydantic import BaseModel

from config.settings import get_settings

T = TypeVar("T", bound=BaseModel)


class LLMFactory:
    """Get a Pydantic object back from OpenAI or Anthropic.

    Both APIs now support structured outputs natively: the response is
    constrained to the model's JSON schema, so no extra library is needed.
    """

    def __init__(self, provider: str):
        settings = get_settings()
        if provider not in ("openai", "anthropic"):
            raise ValueError(f"Unsupported LLM provider: {provider}")
        self.provider = provider
        self.settings = getattr(settings, provider)
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        if self.provider == "openai":
            # Also reads OPENAI_BASE_URL, for OpenAI-compatible servers.
            return OpenAI(api_key=self.settings.api_key, max_retries=self.settings.max_retries)
        return Anthropic(api_key=self.settings.api_key, max_retries=self.settings.max_retries)

    def create_completion(
        self, response_model: Type[T], messages: List[Dict[str, str]], **kwargs
    ) -> T:
        model = kwargs.get("model", self.settings.default_model)
        temperature = kwargs.get("temperature", self.settings.temperature)
        max_tokens = kwargs.get("max_tokens", self.settings.max_tokens)

        if self.provider == "openai":
            return self._openai_completion(response_model, messages, model, temperature, max_tokens)
        return self._anthropic_completion(response_model, messages, model, temperature, max_tokens)

    def _openai_completion(self, response_model, messages, model, temperature, max_tokens):
        params: Dict[str, Any] = {
            "model": model,
            "input": messages,  # the Responses API accepts system/user/assistant roles
            "text_format": response_model,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens

        response = self.client.responses.parse(**params)
        if response.output_parsed is None:
            raise RuntimeError(f"OpenAI returned no parsed output (status: {response.status})")
        return response.output_parsed

    def _anthropic_completion(self, response_model, messages, model, temperature, max_tokens):
        # Anthropic takes the system prompt as a separate parameter.
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]

        params: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": chat,
            "output_format": response_model,
            # If the model declines a request, the API retries it on a fallback model.
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }
        if temperature is not None:
            params["temperature"] = temperature

        response = self.client.beta.messages.parse(**params)
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise RuntimeError(
                f"Anthropic returned no parsed output (stop_reason: {response.stop_reason})"
            )
        return response.parsed_output
