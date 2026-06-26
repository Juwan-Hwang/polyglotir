"""Abstract frontend base — defines the compile/decode contract.

Every frontend MUST:
- ``compile(ir)`` — produce a surface string from IR (lossless).
- ``decode(text)`` — recover the IR from a surface string (lossless).
- ``name`` — unique identifier used by the CLI and benchmark harness.

The contract is: ``decode(compile(ir)) == ir`` for all valid IR.
This is tested in :mod:`tests.test_frontend`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..ir.schema import SilpIR


class Frontend(ABC):
    """Abstract base for all SILP frontends."""

    name: str = "abstract"

    @abstractmethod
    def compile(self, ir: SilpIR) -> str:
        """Compile IR → surface string."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, text: str) -> SilpIR:
        """Decode surface string → IR.

        Raises:
            ValueError: if *text* cannot be parsed by this frontend.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
