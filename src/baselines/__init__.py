"""Baseline context-management strategies (plan Phase 6).

The package owns the five modes the controlled comparison needs and the
BrainOS-free retriever Mode C/E use. It depends on :mod:`brain.retrieval_policy`
for lexical scoring but never imports the BrainOS runtime, so the modes stay
runnable in an environment where BrainOS is absent.
"""

from __future__ import annotations

from .ablations import (
    ABLATIONS,
    EXCLUDED_ABLATIONS,
    POLICY_KNOB_FIELDS,
    TEMPORAL_SIGNAL_NAMES,
    AblationProfile,
    ablation_pairs,
    ablation_profile,
    is_ablation,
    prepare_records,
    strip_lifecycle,
    strip_temporal_signals,
)
from .modes import (
    ABLATION_NO_CONFLICT,
    ABLATION_NO_MEMORY,
    ABLATION_NO_RELEVANCE,
    ABLATION_NO_TEMPORAL,
    ABLATION_ORDER,
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
    ablation_choices,
    mode_choices,
    mode_label,
    mode_profile,
    resolve_mode,
)
from .rag import HistoryChunk, chunk_history, retrieve_chunks

__all__ = [
    "ABLATIONS",
    "ABLATION_NO_CONFLICT",
    "ABLATION_NO_MEMORY",
    "ABLATION_NO_RELEVANCE",
    "ABLATION_NO_TEMPORAL",
    "ABLATION_ORDER",
    "DEFAULT_MODE",
    "EVIDENCE_BUDGET",
    "EXCLUDED_ABLATIONS",
    "LEGACY_MODE_ALIASES",
    "MODES",
    "MODE_BRAINOS",
    "MODE_BRAINOS_RAG",
    "MODE_FULL_CONTEXT",
    "MODE_ORDER",
    "MODE_RAG",
    "MODE_SLIDING_WINDOW",
    "POLICY_KNOB_FIELDS",
    "TEMPORAL_SIGNAL_NAMES",
    "AblationProfile",
    "BaselineMode",
    "HistoryChunk",
    "ablation_choices",
    "ablation_pairs",
    "ablation_profile",
    "chunk_history",
    "is_ablation",
    "mode_choices",
    "mode_label",
    "mode_profile",
    "prepare_records",
    "resolve_mode",
    "retrieve_chunks",
    "strip_lifecycle",
    "strip_temporal_signals",
]
