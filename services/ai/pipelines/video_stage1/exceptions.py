from __future__ import annotations


class Stage1UnavailableError(RuntimeError):
    """Raised when Stage1 preprocessing runtime dependencies are unavailable."""


class Stage1ExplanationError(RuntimeError):
    """Raised when Stage1 explanation generation fails."""
