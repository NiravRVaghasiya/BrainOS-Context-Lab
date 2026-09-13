# Threat model

This document is the Phase 0 threat-model entry point. The maintained security controls are documented in [`security.md`](security.md).

The primary assets are provider API keys, user conversation/memory content, session identifiers, and evaluation artifacts. The principal threats are credential leakage, cross-session data access, prompt injection through retrieved memory, accidental runaway provider usage, and misleading evaluation results caused by non-equivalent baselines.

The initial mitigations are session-only secret handling, explicit session/conversation scoping, sanitized traces and exports, retrieved-memory delimiters, cost limits, and controlled evaluation protocols.
