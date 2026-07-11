"""silp-frontend — Layer 2: pluggable multi-frontend surface layer.

Each frontend compiles :class:`~silp.ir.SilpIR` → surface ``str`` and decodes
surface ``str`` → :class:`~silp.ir.SilpIR`.  The code frontend is the default;
math, knowledge-reference, structure-symbol, and natural-language frontends
are pluggable alternatives.

All frontends are registered in :data:`FRONTENDS` so the CLI and bench
modules can look them up by name.
"""

from .base import Frontend
from .code import CodeFrontend
from .json_frontend import JSONFrontend
from .llmlingua import LLMLingua2Frontend
from .lock import CompileLock
from .natural import NaturalFrontend
from .nl_json import NLInJSONFrontend
from .registry import get_frontend, register_frontend, list_frontends

# ── Register built-in frontends ───────────────────────────────────────
register_frontend(CodeFrontend)
register_frontend(NaturalFrontend)
register_frontend(JSONFrontend)
register_frontend(NLInJSONFrontend)
register_frontend(LLMLingua2Frontend)

__all__ = [
    "Frontend",
    "CodeFrontend",
    "NaturalFrontend",
    "JSONFrontend",
    "NLInJSONFrontend",
    "LLMLingua2Frontend",
    "CompileLock",
    "get_frontend",
    "register_frontend",
    "list_frontends",
]
