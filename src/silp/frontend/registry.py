"""Frontend registry — plug-in lookup by name."""

from __future__ import annotations

from .base import Frontend

# Global registry: name → frontend class
_REGISTRY: dict[str, type[Frontend]] = {}


def register_frontend(cls: type[Frontend]) -> type[Frontend]:
    """Register a frontend class.  Can be used as a decorator.

    >>> @register_frontend
    ... class MyFrontend(Frontend):
    ...     name = "my"
    """
    if not hasattr(cls, "name") or cls.name == "abstract":
        raise ValueError(f"{cls.__name__} must define a non-'abstract' name")
    _REGISTRY[cls.name] = cls
    return cls


def get_frontend(name: str = "code") -> Frontend:
    """Look up and instantiate a frontend by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown frontend {name!r}. Available: {available}")
    return _REGISTRY[name]()


def list_frontends() -> list[str]:
    """Return sorted list of registered frontend names."""
    return sorted(_REGISTRY)
