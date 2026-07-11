"""Natural-language-in-JSON frontend — format-isolation control baseline.

Per spec §4 Phase 2 ①: "未压缩自然语言塞进 JSON task_description"

This frontend takes the **exact same natural-language prose** produced by
:class:`NaturalFrontend` and wraps it in a JSON object:

    {"task_description": "If not rain at t+1, start hike otherwise start cards at indoor."}

**Purpose**: This is a controlled experiment. The *content* is identical
to the ``natural`` frontend (unstructured prose). The only difference is
the *container* (JSON vs. bare text). If models perform better on this
frontend than on bare ``natural``, the improvement is attributable to the
JSON wrapper alone — not to structured slots or code vocabulary.

This cleanly isolates "format contribution" from "code vocabulary prior"
when compared against the ``code`` and ``json`` frontends.
"""

from __future__ import annotations

import json

from ..ir.schema import SilpIR
from .base import Frontend
from .natural import NaturalFrontend


class NLInJSONFrontend(Frontend):
    """Natural language wrapped in JSON ``task_description`` — format control.

    The prose content is identical to :class:`NaturalFrontend`; only the
    container differs (JSON vs. bare text).
    """

    name = "nl_json"

    def __init__(self) -> None:
        self._natural = NaturalFrontend()

    def compile(self, ir: SilpIR) -> str:
        prose = self._natural.compile(ir)
        return json.dumps(
            {"task_description": prose},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def decode(self, text: str) -> SilpIR:
        """Decode is not supported — this is a control baseline.

        Like :class:`NaturalFrontend`, the content is unstructured prose
        and cannot be losslessly parsed back to IR.
        """
        raise NotImplementedError(
            "NLInJSONFrontend.decode is not supported — it is a control "
            "baseline, not a round-trip codec."
        )
