"""Unified model interface — local HuggingFace + remote API models.

All models implement the same :class:`ModelBackend` protocol, so the smoke
test runner and benchmark harness can treat them identically:

    from silp.bench.models import get_model
    model = get_model("smollm-360m")
    response = model.generate("Decode this SILP payload: cancel(flight)")

Local models (CPU):
    - SmolLM-360M
    - Qwen2.5-0.5B
    - TinyLlama-1.1B

Remote models (API):
    - gpt-4o-mini (OpenAI)
    - claude-3.5-sonnet (Anthropic)
    - gemini-pro (Google)

Configuration is loaded from environment variables (.env file).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Model registry ────────────────────────────────────────────────────

_LOCAL_MODELS: dict[str, dict[str, str]] = {
    "smollm-360m": {
        "hf_id": "HuggingFaceTB/SmolLM-360M",
        "description": "SmolLM 360M (CPU, ~720MB)",
    },
    "qwen2.5-0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B",
        "description": "Qwen2.5 0.5B (CPU, ~1GB)",
    },
    "tinyllama-1.1b": {
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "TinyLlama 1.1B Chat (CPU, ~2.2GB)",
    },
}

_API_MODELS: dict[str, dict[str, str]] = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "description": "OpenAI GPT-4o-mini",
    },
    "claude-3.5-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-3-5-sonnet-20241022",
        "description": "Anthropic Claude 3.5 Sonnet",
    },
    "gemini-pro": {
        "provider": "google",
        "model_id": "gemini-pro",
        "description": "Google Gemini Pro",
    },
}


# ── Data structures ───────────────────────────────────────────────────


@dataclass
class GenerationConfig:
    """Parameters for text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.0  # 0 = deterministic (greedy)
    top_p: float = 1.0
    do_sample: bool = False
    timeout: float = 60.0  # seconds (API models)


@dataclass
class ModelResponse:
    """Unified response from any model backend."""

    text: str
    model: str
    backend: str  # "local" or "api"
    raw: object = None  # provider-specific raw response
    elapsed: float = 0.0
    error: Optional[str] = None


# ── Abstract backend ──────────────────────────────────────────────────


class ModelBackend(ABC):
    """Abstract base for all model backends."""

    name: str
    backend_type: str  # "local" or "api"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> ModelResponse:
        """Generate text from a prompt."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} type={self.backend_type!r}>"


# ── Local HuggingFace backend ─────────────────────────────────────────


class LocalHFBackend(ModelBackend):
    """Local HuggingFace model backend (CPU inference)."""

    backend_type = "local"

    def __init__(self, name: str, hf_id: str) -> None:
        self.name = name
        self.hf_id = hf_id
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Lazy-load model and tokenizer on first use."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "torch/transformers not installed. "
                "Run: pip install torch transformers"
            ) from exc

        print(f"  [load] {self.name} ({self.hf_id})...", file=__import__("sys").stderr)
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            torch_dtype=torch.float32,  # CPU: always float32
            device_map="cpu",
        )
        self._model.eval()
        print(f"  [load] {self.name} ready", file=__import__("sys").stderr)

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> ModelResponse:
        import time

        self._load()
        config = config or GenerationConfig()

        try:
            import torch

            inputs = self._tokenizer(prompt, return_tensors="pt")
            t0 = time.time()
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature if config.do_sample else 1.0,
                    top_p=config.top_p,
                    do_sample=config.do_sample,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            elapsed = time.time() - t0

            # Extract only the new tokens (exclude the prompt)
            input_len = inputs["input_ids"].shape[1]
            new_tokens = outputs[0][input_len:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            return ModelResponse(
                text=text,
                model=self.name,
                backend="local",
                elapsed=elapsed,
            )
        except Exception as exc:
            return ModelResponse(
                text="",
                model=self.name,
                backend="local",
                error=str(exc),
            )


# ── OpenAI backend ────────────────────────────────────────────────────


class OpenAIBackend(ModelBackend):
    """OpenAI API backend."""

    backend_type = "api"

    def __init__(self, name: str, model_id: str) -> None:
        self.name = name
        self.model_id = model_id
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai not installed. Run: pip install openai") from exc

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Copy .env.example to .env.")

        base_url = os.environ.get("OPENAI_BASE_URL") or None
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> ModelResponse:
        import time

        config = config or GenerationConfig()
        try:
            client = self._get_client()
            t0 = time.time()
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                timeout=config.timeout,
            )
            elapsed = time.time() - t0
            return ModelResponse(
                text=response.choices[0].message.content.strip(),
                model=self.name,
                backend="api",
                raw=response,
                elapsed=elapsed,
            )
        except Exception as exc:
            return ModelResponse(
                text="",
                model=self.name,
                backend="api",
                error=str(exc),
            )


# ── Anthropic backend ─────────────────────────────────────────────────


class AnthropicBackend(ModelBackend):
    """Anthropic API backend."""

    backend_type = "api"

    def __init__(self, name: str, model_id: str) -> None:
        self.name = name
        self.model_id = model_id
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic not installed. Run: pip install anthropic"
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set.")

        self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> ModelResponse:
        import time

        config = config or GenerationConfig()
        try:
            client = self._get_client()
            t0 = time.time()
            response = client.messages.create(
                model=self.model_id,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.time() - t0
            text = response.content[0].text.strip() if response.content else ""
            return ModelResponse(
                text=text,
                model=self.name,
                backend="api",
                raw=response,
                elapsed=elapsed,
            )
        except Exception as exc:
            return ModelResponse(
                text="",
                model=self.name,
                backend="api",
                error=str(exc),
            )


# ── Google Gemini backend ─────────────────────────────────────────────


class GeminiBackend(ModelBackend):
    """Google Gemini API backend."""

    backend_type = "api"

    def __init__(self, name: str, model_id: str) -> None:
        self.name = name
        self.model_id = model_id
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "google-generativeai not installed. "
                "Run: pip install google-generativeai"
            ) from exc

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set.")

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(self.model_id)
        return self._model

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> ModelResponse:
        import time

        config = config or GenerationConfig()
        try:
            model = self._get_model()
            t0 = time.time()
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": config.max_new_tokens,
                    "temperature": config.temperature,
                },
            )
            elapsed = time.time() - t0
            text = response.text.strip() if hasattr(response, "text") else ""
            return ModelResponse(
                text=text,
                model=self.name,
                backend="api",
                raw=response,
                elapsed=elapsed,
            )
        except Exception as exc:
            return ModelResponse(
                text="",
                model=self.name,
                backend="api",
                error=str(exc),
            )


# ── Factory ───────────────────────────────────────────────────────────


def get_model(name: str) -> ModelBackend:
    """Look up and instantiate a model backend by name.

    Available models:
        Local:  smollm-360m, qwen2.5-0.5b, tinyllama-1.1b
        API:    gpt-4o-mini, claude-3.5-sonnet, gemini-pro
    """
    if name in _LOCAL_MODELS:
        info = _LOCAL_MODELS[name]
        return LocalHFBackend(name, info["hf_id"])

    if name in _API_MODELS:
        info = _API_MODELS[name]
        provider = info["provider"]
        model_id = info["model_id"]
        if provider == "openai":
            return OpenAIBackend(name, model_id)
        elif provider == "anthropic":
            return AnthropicBackend(name, model_id)
        elif provider == "google":
            return GeminiBackend(name, model_id)

    available = ", ".join(list(_LOCAL_MODELS) + list(_API_MODELS))
    raise KeyError(f"Unknown model {name!r}. Available: {available}")


def list_models() -> dict[str, dict[str, str]]:
    """Return all registered models (local + API)."""
    return {**_LOCAL_MODELS, **_API_MODELS}


def load_env() -> None:
    """Load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parents[3] / ".env"
        load_dotenv(env_path)
    except ImportError:
        pass  # dotenv optional; user can export vars manually
