"""LLMLingua-2 frontend — prompt-compression baseline.

Per spec §4 Phase 2: "基线含自然语言原文、LLMLingua-2、纯 JSON"

This frontend compresses the natural-language rendering of the IR using
Microsoft's LLMLingua-2, a trained prompt-compression model. It serves as
the **compression-tool baseline** — the thing SILP is NOT (SILP is a
protocol layer, not a compression tool).

The comparison is illuminating:

- ``code`` / ``json`` frontends: structurally compact by design, no ML needed.
- ``llmlingua2``: ML-based compression of natural language, optimizes for
  token reduction but may lose semantic precision (especially negation logic).
- ``natural``: uncompressed natural language control.

If SILP's structured frontends achieve higher cross-model comprehension
*and* lower token variance than LLMLingua-2, that is evidence that
"shared syntax prior > compression" for inter-agent communication.

The LLMLingua-2 model is loaded lazily on first ``compile()`` call.
The default ``rate`` (compression ratio) is 0.5, matching the paper's
recommended setting. This is configurable via the ``rate`` parameter.
"""

from __future__ import annotations

from ..ir.schema import SilpIR
from .base import Frontend
from .natural import NaturalFrontend


class LLMLingua2Frontend(Frontend):
    """LLMLingua-2 compressed natural-language baseline.

    Uses the ``llmlingua`` package (Microsoft) to compress the natural-
    language rendering of the IR. The compressed text is what the model
    sees — measuring whether compression hurts cross-model comprehension
    compared to structured frontends.
    """

    name = "llmlingua2"

    def __init__(self, rate: float = 0.5) -> None:
        self.rate = rate
        self._natural = NaturalFrontend()
        self._compressor = None

    def _get_compressor(self):
        """Lazy-load the LLMLingua-2 prompt compressor."""
        if self._compressor is not None:
            return self._compressor
        try:
            from llmlingua import PromptCompressor
        except ImportError as exc:
            raise ImportError(
                "llmlingua not installed. Run: pip install llmlingua"
            ) from exc

        # LLMLingua-2 uses a small BERT-based model for token-level
        # compression. ``use_llmlingua2=True`` enables the v2 algorithm.
        self._compressor = PromptCompressor(use_llmlingua2=True)
        return self._compressor

    def compile(self, ir: SilpIR) -> str:
        prose = self._natural.compile(ir)
        compressor = self._get_compressor()

        compressed = compressor.compress_prompt(
            prose,
            rate=self.rate,
            force_tokens=None,  # no forced tokens — pure compression
            drop_consecutive=False,
        )

        # The returned dict has a "compressed_prompt" key.
        result = compressed.get("compressed_prompt", "")
        if not result:
            # Fallback: if compression fails, use original prose
            result = prose

        return result.strip()

    def decode(self, text: str) -> SilpIR:
        """Decode is not supported — compressed text is lossy by design.

        LLMLingua-2 compression is inherently lossy (it removes tokens
        based on perplexity). The compressed text cannot be parsed back
        to a structured IR. This is itself a finding: SILP's structured
        frontends are losslessly decodable, while compression tools are not.
        """
        raise NotImplementedError(
            "LLMLingua2Frontend.decode is not supported — compression is "
            "inherently lossy and cannot be reversed. This is a key "
            "distinction from SILP's lossless structured frontends."
        )
