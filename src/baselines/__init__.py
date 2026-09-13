"""Baseline context-management strategies (plan Phase 6).

The package owns the five modes the controlled comparison needs and the
BrainOS-free retriever Mode C/E use. It depends on :mod:`brain.retrieval_policy`
for lexical scoring but never imports the BrainOS runtime, so the modes stay
runnable in an environment where BrainOS is absent.
"""

from __future__ import annotations

from .modes import (
    DEFAULT_MODE,
    EVIDENCE_BUDGET,
    LEGACY_MODE_ALIASES,
    MODE_BRAINOS,
    MODE_BRAINOS_RAG,
    MODE_FULL_CONTEXT,
    MODE_ORDER,
    MODE_RAG,
    MODE_SLIDING_WINDOW,
    MODES,
    BaselineMode,
    mode_choices,
    mode_label,
    mode_profile,
    resolve_mode,
)
from .rag import HistoryChunk, chunk_history, retrieve_chunks

__all__ = [
    "DEFAULT_MODE",
    "EVIDENCE_BUDGET",
    "LEGACY_MODE_ALIASES",
    "MODES",
    "MODE_BRAINOS",
    "MODE_BRAINOS_RAG",
    "MODE_FULL_CONTEXT",
    "MODE_ORDER",
    "MODE_RAG",
    "MODE_SLIDING_WINDOW",
    "BaselineMode",
    "HistoryChunk",
    "chunk_history",
    "mode_choices",
    "mode_label",
    "mode_profile",
    "resolve_mode",
    "retrieve_chunks",
]
