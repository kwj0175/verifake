from __future__ import annotations


class Stage1UnavailableError(RuntimeError):
    """Raised when Stage1 preprocessing runtime dependencies are unavailable."""
