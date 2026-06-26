"""silp-negotiation — Layer 3: meta-protocol (handshake, session, error codes).

Phase 0.5 stub — handshake and session management will be implemented
in Phase 3.  For now, only the error-code enum is defined.
"""

from enum import Enum


class ErrorCode(str, Enum):
    """Payload-decode error codes (spec §2 Layer 3).

    These exist ONLY in the payload-decode sub-layer; they do NOT replace
    the outer MCP/A2A JSON-RPC error mechanism.
    """

    INVALID_SYNTAX = "invalid_syntax"
    UNSUPPORTED_VERSION = "unsupported_version"
    UNSUPPORTED_FRONTEND = "unsupported_frontend"
    DECODE_FAILED = "decode_failed"
    TIMEOUT = "timeout"
    SESSION_EXPIRED = "session_expired"


__all__ = ["ErrorCode"]
