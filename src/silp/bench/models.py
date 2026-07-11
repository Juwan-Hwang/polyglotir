"""Unified model interface — local HuggingFace + OpenAI-compatible API models.

All models implement the same :class:`ModelBackend` protocol, so the smoke
test runner and benchmark harness can treat them identically:

    from silp.bench.models import get_model
    model = get_model("deepseek-v3.2")
    response = model.generate("Decode this SILP payload: cancel(flight)")

Model definitions live in ``data/metadata/model_configs.json`` — the single
source of truth.  Add or remove models there and the code picks them up
automatically.

Three backend types:
    - **LocalHFBackend**  — HuggingFace CPU inference (SmolLM, Qwen, TinyLlama)
    - **OpenAIBackend**   — any OpenAI-compatible endpoint (local proxy, official API)
    - **AnthropicBackend**— Anthropic native API
    - **GeminiBackend**   — Google Gemini native API

The OpenAI backend reads ``OPENAI_BASE_URL`` from the environment, so it
works with any OpenAI-compatible proxy (e.g. ``http://localhost:8787/v1``).
"""

from __future__ import annotations

import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _ROOT / "data" / "metadata" / "model_configs.json"


# ── Config loading ────────────────────────────────────────────────────


def _load_config() -> dict[str, dict[str, dict[str, str]]]:
    """Load model definitions from the JSON config file.

    Returns a dict with two keys: ``"local"`` and ``"api"``, each mapping
    model names to their configuration.
    """
    if not _CONFIG_PATH.exists():
        return {"local": {}, "api": {}}
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "local": data.get("local", {}),
        "api": data.get("api", {}),
    }


_CONFIG = _load_config()
_LOCAL_MODELS = _CONFIG["local"]
_API_MODELS = _CONFIG["api"]


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

        print(f"  [load] {self.name} ({self.hf_id})...", file=sys.stderr)
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            torch_dtype=torch.float32,  # CPU: always float32
            device_map="cpu",
        )
        self._model.eval()
        print(f"  [load] {self.name} ready", file=sys.stderr)

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


# ── OpenAI-compatible backend ─────────────────────────────────────────


class OpenAIBackend(ModelBackend):
    """OpenAI-compatible API backend.

    Works with:
    - The official OpenAI API (api.openai.com)
    - Any OpenAI-compatible proxy (set ``OPENAI_BASE_URL``)

    The proxy at ``http://localhost:8787/v1`` is configured via ``.env``.
    """

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

        api_key = os.environ.get("OPENAI_API_KEY", "dummy")
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

            # Use streaming to handle proxies that always return SSE format.
            # Collect all chunks, then assemble the final text.
            chunks: list[str] = []
            stream = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.max_new_tokens,
                temperature=config.temperature if config.temperature > 0 else 0,
                timeout=config.timeout,
                stream=True,
            )
            for event in stream:
                if event.choices and event.choices[0].delta.content:
                    chunks.append(event.choices[0].delta.content)

            elapsed = time.time() - t0
            text = "".join(chunks).strip()
            return ModelResponse(
                text=text,
                model=self.name,
                backend="api",
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


# ── Provider → backend class mapping ──────────────────────────────────

_BACKENDS: dict[str, type[ModelBackend]] = {
    "local": LocalHFBackend,
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "google": GeminiBackend,
}


# ── Factory ───────────────────────────────────────────────────────────


def get_model(name: str) -> ModelBackend:
    """Look up and instantiate a model backend by name.

    Model names are defined in ``data/metadata/model_configs.json``.
    Use :func:`list_models` to see all available names.
    """
    if name in _LOCAL_MODELS:
        info = _LOCAL_MODELS[name]
        return LocalHFBackend(name, info["hf_id"])

    if name in _API_MODELS:
        info = _API_MODELS[name]
        provider = info["provider"]
        backend_cls = _BACKENDS.get(provider)
        if backend_cls is None:
            raise KeyError(f"Unknown provider {provider!r} for model {name!r}")
        return backend_cls(name, info["model_id"])

    available = ", ".join(list(_LOCAL_MODELS) + list(_API_MODELS))
    raise KeyError(f"Unknown model {name!r}. Available: {available}")


def list_models() -> dict[str, dict[str, str]]:
    """Return all registered models (local + API) from config."""
    return {**_LOCAL_MODELS, **_API_MODELS}


def list_model_names() -> list[str]:
    """Return sorted list of all model names."""
    return sorted(list(_LOCAL_MODELS) + list(_API_MODELS))


def get_model_family(name: str) -> str:
    """Return the model family (e.g. 'glm', 'kimi', 'deepseek')."""
    if name in _LOCAL_MODELS:
        return _LOCAL_MODELS[name].get("family", "unknown")
    if name in _API_MODELS:
        return _API_MODELS[name].get("family", "unknown")
    return "unknown"


def discover_proxy_models(base_url: str | None = None) -> list[str]:
    """Discover models available on an OpenAI-compatible proxy.

    Queries the ``/v1/models`` endpoint and returns a list of model IDs.
    Useful for verifying that the proxy is reachable and seeing what's
    available without manually checking the config.
    """
    import urllib.request

    base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
    if not base_url:
        base_url = "http://localhost:8787/v1"

    # Ensure the URL ends with /v1/models
    url = base_url.rstrip("/")
    if not url.endswith("/models"):
        if not url.endswith("/v1"):
            url += "/v1"
        url += "/models"

    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return [m["id"] for m in data.get("data", [])]
    except Exception as exc:
        print(f"  [warn] could not discover proxy models: {exc}", file=sys.stderr)
        return []


def load_env() -> None:
    """Load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        env_path = _ROOT / ".env"
        load_dotenv(env_path)
    except ImportError:
        pass  # dotenv optional; user can export vars manually
