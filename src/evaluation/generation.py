"""Model-in-the-loop generation for evaluation runs (plan Phase 9).

Phase 7 could score retrieval without credentials, and Phase 8 stopped there on
purpose: *"QAE is meaningless while answers are scripted. Put a model in the
loop before comparing quality-adjusted efficiency across modes."* This module is
that model call, and nothing more:

```text
mode prompt (built by the context engine)
        |
        v
provider.generate(messages, **constant request parameters)
        |
        v
GenerationResult(text, latency, usage, settings fingerprint)
```

What it deliberately does **not** do:

* **No prompt construction.** The messages come from
  :mod:`brain.context_builder` through the baseline-mode replay, so a generated
  answer is scored against the exact prompt a mode produced. If this module
  built prompts, the comparison would be between two implementations.
* **No credentials.** A provider object is passed in already configured. API
  keys are read from an environment variable by
  :func:`api_key_from_environment` (for CLI runs) and are never stored on a
  :class:`ModelSpec`, a :class:`GenerationResult`, or a run artifact — a
  :class:`ModelSpec` records the *name* of the environment variable, not a value.
* **No scoring.** The answer string goes back to the Phase 7 scorer; verdicts
  stay in :mod:`evaluation.scoring` so there is exactly one place that decides
  what "correct" means.
* **No retries by default.** ``max_attempts=1``. A retry changes how many
  requests a task cost and how long it took, so it is opt-in and recorded
  (``attempts``) rather than applied silently.

The single property this module exists to protect is the plan's controlled
comparison: every mode is called with **identical** sampling parameters. Those
parameters live in one :class:`GenerationSettings` per run, and
:meth:`GenerationSettings.fingerprint` is a hash of them, so an experiment can
prove afterwards that all five modes were generated under the same settings
instead of merely asserting that they were.

Errors are failures, not exceptions: a provider that rejects a request, times
out, or returns an empty completion produces a :class:`GenerationResult` whose
``error`` is credential-redacted, so one failed request cannot destroy a run
that already spent budget (and cannot leak a key into a traceback either).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Protocol

from brain.tokenizers import TokenCounter, estimate_tokens
from providers import create_provider
from providers.base import (
    ProviderConfig,
    safe_error_message,
)

#: Identifies the generation semantics stored on every result.
GENERATION_VERSION = "generation-v1"

#: Per-message overhead added when a prompt is counted for a cost ceiling. The
#: context builder uses the same figure (``per_message_overhead``), so the
#: budget's idea of "input tokens" matches the accounting the prompt reports.
MESSAGE_OVERHEAD_TOKENS = 4

#: A credential is never a field, and the field that names where to *find* one
#: is validated as an environment variable name. That closes a real leak: a
#: caller who pastes a key into ``api_key_env`` would otherwise export it inside
#: the model spec, and the error must not echo the value back either.
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")

#: Sampling parameters a controlled comparison must hold constant. Listed
#: explicitly so the fingerprint cannot silently start ignoring a new parameter.
SAMPLING_PARAMETERS: tuple[str, ...] = (
    "model",
    "temperature",
    "max_tokens",
    "top_p",
    "seed",
)


class MissingCredentialError(RuntimeError):
    """No credential was found for a run that needs one.

    The message names the environment variable. It never contains a value: a
    missing key is reported the same way whether the variable is unset or the
    key is simply empty.
    """


@dataclass(frozen=True)
class GenerationSettings:
    """The request parameters held constant across every mode of a run."""

    model: str = ""
    temperature: float = 0.0
    max_tokens: int | None = None
    top_p: float | None = None
    seed: int | None = None
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("A model identifier is required for generation.")
        if not isinstance(self.temperature, (int, float)) or not isfinite(self.temperature):
            raise ValueError("temperature must be a finite number.")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative.")
        if self.max_tokens is not None and (
            not isinstance(self.max_tokens, int)
            or isinstance(self.max_tokens, bool)
            or self.max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer or None.")
        if self.top_p is not None and (
            not isinstance(self.top_p, (int, float))
            or not isfinite(self.top_p)
            or not 0 < float(self.top_p) <= 1
        ):
            raise ValueError("top_p must be in (0, 1] or None.")
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool)
        ):
            raise ValueError("seed must be an integer or None.")
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or not isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be greater than zero.")

    def request_kwargs(self) -> dict[str, Any]:
        """Return the keyword arguments passed to ``provider.generate``.

        Only set parameters are included, so a model that rejects ``seed`` (or
        ``top_p``) is not sent one just because the interface supports it.
        """

        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs

    def sampling_parameters(self) -> dict[str, Any]:
        """Return the parameters the controlled-comparison fingerprint covers."""

        return {name: getattr(self, name) for name in SAMPLING_PARAMETERS}

    def fingerprint(self) -> str:
        """Return a stable hash of the sampling parameters.

        Two requests share a fingerprint exactly when a controlled comparison
        would consider them generated under the same settings. Transport
        settings (``timeout_seconds``) are excluded: a timeout is a cost
        guard, not a property of the model's distribution.
        """

        payload = {
            "version": GENERATION_VERSION,
            **self.sampling_parameters(),
        }
        rendered = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "seed": self.seed,
            "timeout_seconds": self.timeout_seconds,
            "settings_fingerprint": self.fingerprint(),
        }


@dataclass(frozen=True)
class ModelSpec:
    """One model in the plan's controlled-experiment matrix.

    ``name`` is the matrix role (``"small"``, ``"medium"``, ``"local"`` …) and is
    what a cross-model report groups by. The credential is *never* a field: the
    spec records ``api_key_env``, the name of the environment variable the CLI
    reads, so a spec can be logged, exported, and diffed safely.
    """

    name: str = "primary"
    provider: str = "openai"
    model: str = ""
    temperature: float = 0.0
    max_tokens: int | None = None
    top_p: float | None = None
    seed: int | None = None
    timeout_seconds: float = 60.0
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("A model matrix role name is required.")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("A provider name is required.")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("A model identifier is required.")
        if not isinstance(self.api_key_env, str) or not _ENV_NAME_RE.fullmatch(
            self.api_key_env.strip()
        ):
            raise ValueError(
                "api_key_env must be the name of an environment variable "
                "(letters, digits, underscore), not a credential value."
            )
        # Validation for the remaining numeric fields is shared with the
        # settings object, so the two can never disagree.
        self.generation_settings()

    def generation_settings(self) -> GenerationSettings:
        """Return the frozen request parameters for this model."""

        return GenerationSettings(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            seed=self.seed,
            timeout_seconds=self.timeout_seconds,
        )

    def with_limits(self, limits: Any) -> ModelSpec:
        """Return a copy with a run's cost caps applied.

        ``limits`` is duck-typed (anything exposing ``capped``), which keeps the
        cost controls free of any dependency on this module. The caps are applied
        once, before the first request, so they cannot change mid-run and break
        the controlled comparison.
        """

        max_tokens, timeout_seconds = limits.capped(self.max_tokens, self.timeout_seconds)
        return ModelSpec(
            name=self.name,
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=max_tokens,
            top_p=self.top_p,
            seed=self.seed,
            timeout_seconds=timeout_seconds,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
            notes=self.notes,
        )

    def provider_config(self, api_key: str) -> ProviderConfig:
        """Return the session configuration for this model.

        The returned object keeps the key out of ``repr`` and out of
        :meth:`ProviderConfig.safe_dict`, which is why the adapter constructors
        accept it rather than loose arguments.
        """

        return ProviderConfig(
            provider=self.provider,
            model=self.model,
            api_key=api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready provenance. Contains no credential."""

        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "seed": self.seed,
            "timeout_seconds": self.timeout_seconds,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "notes": self.notes,
            "settings_fingerprint": self.generation_settings().fingerprint(),
        }


@dataclass(frozen=True)
class GenerationResult:
    """One provider request, normalized and safe to serialize.

    ``ok`` is false for an empty completion as well as for an error: a mode that
    produced no text cannot answer a question, and scoring it as an empty answer
    would be a fabricated result.
    """

    text: str = ""
    model: str = ""
    requested_model: str = ""
    latency_ms: float | None = None
    usage: dict[str, int] = field(default_factory=dict)
    settings_fingerprint: str = ""
    prompt_token_count: int | None = None
    attempts: int = 0
    #: Set by the cost controls when a request was refused *before* being sent
    #: (an input prompt past the run's per-request ceiling). A skipped request
    #: cost nothing and is never scored as a wrong answer.
    skipped: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return not self.skipped and not self.error and bool(self.text.strip())

    @property
    def prompt_tokens(self) -> int | None:
        reported = self.usage.get("prompt_tokens")
        if reported is not None:
            return reported
        return self.prompt_token_count

    @property
    def completion_tokens(self) -> int | None:
        return self.usage.get("completion_tokens")

    @property
    def total_tokens(self) -> int | None:
        reported = self.usage.get("total_tokens")
        if reported is not None:
            return reported
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return int(self.prompt_tokens or 0) + int(self.completion_tokens or 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "requested_model": self.requested_model,
            "latency_ms": self.latency_ms,
            "usage": dict(self.usage),
            "settings_fingerprint": self.settings_fingerprint,
            "prompt_token_count": self.prompt_token_count,
            "attempts": self.attempts,
            "skipped": self.skipped,
            "error": self.error,
            "generated": self.ok,
        }


#: Shape the baseline-mode replay and the experiment runner both consume.
Generator = Callable[[Sequence[Mapping[str, str]]], GenerationResult]


class TokenBudget(Protocol):
    """Structural seam for the cost controls (plan Phase 15).

    Defined as a protocol so this module never imports
    :mod:`evaluation.limits`: generation wraps a budget, it does not define one.
    """

    counter: TokenCounter

    def fits_prompt(self, prompt_tokens: int) -> tuple[bool, str]:
        """Return whether one prompt may be sent under the per-request cap."""

    def check_request(self, prompt_tokens: int) -> None:
        """Raise when a run-level limit is already exhausted."""

    def record_skip(self) -> None:
        """Count a request that was refused before it was sent."""

    def charge(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_text: str = "",
        usage: Mapping[str, int] | None = None,
        failed: bool = False,
    ) -> dict[str, int]:
        """Record one request against the run's budget."""


def count_messages(
    messages: Sequence[Mapping[str, str]], counter: TokenCounter = estimate_tokens
) -> int:
    """Count a prompt the way the cost ceiling counts it.

    Content plus the same per-message overhead the context builder charges, so a
    budget refusal and the prompt's own accounting agree.
    """

    total = 0
    for message in messages:
        content = "" if message is None else str(message.get("content", "") or "")
        total += counter(content) + MESSAGE_OVERHEAD_TOKENS
    return total


def api_key_from_environment(
    spec: ModelSpec, environ: Mapping[str, str] | None = None
) -> str:
    """Read a run's credential from the environment variable ``spec`` names.

    The environment is the only place a CLI run looks for a key: it keeps keys
    out of shell history and out of the process argument list, and it means no
    command line or artifact can contain one.
    """

    source = os.environ if environ is None else environ
    name = spec.api_key_env
    value = str(source.get(name, "") or "").strip()
    if not value:
        raise MissingCredentialError(
            f"No credential found in the environment variable {name}. Set it for "
            "this shell, or use the application's bring-your-own-key UI instead. "
            "Keys are never written to a run artifact."
        )
    return value


def build_provider(
    spec: ModelSpec,
    *,
    api_key: str,
    provider_factory: Callable[[ProviderConfig], Any] = create_provider,
) -> Any:
    """Build the provider for a run.

    A provider object is returned rather than a configuration, so the credential
    stays inside the adapter that needs it. Validation (unknown provider, missing
    key, bad endpoint) happens here, before any budget is spent.
    """

    return provider_factory(spec.provider_config(api_key))


def generate_answer(
    provider: Any,
    messages: Sequence[Mapping[str, str]],
    settings: GenerationSettings,
    *,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    secrets: tuple[str, ...] = (),
    max_attempts: int = 1,
    retry_delay_seconds: float = 0.0,
) -> GenerationResult:
    """Send one prompt and return a normalized, credential-safe result.

    ``secrets`` is a defence-in-depth redaction list: adapters already scrub
    their own errors, but a run that holds a key can pass it here so a provider
    that echoes a request header cannot leak through ``error``.
    """

    request_messages = [
        {
            "role": str(message.get("role", "user")),
            "content": str(message.get("content", "") or ""),
        }
        for message in messages
    ]
    kwargs = settings.request_kwargs()
    fingerprint = settings.fingerprint()
    attempts_allowed = max(1, int(max_attempts))
    latency_ms: float | None = None
    error: str | None = None

    for attempt in range(1, attempts_allowed + 1):
        started = clock()
        try:
            response = provider.generate(request_messages, **kwargs)
        except Exception as exc:  # provider SDK exceptions are deliberately opaque
            latency_ms = _elapsed_ms(clock() - started)
            error = safe_error_message(exc, secrets=secrets)
        else:
            latency_ms = _elapsed_ms(clock() - started)
            return _result_from_response(response, settings, fingerprint, latency_ms, attempt)
        if attempt < attempts_allowed:
            sleep(max(0.0, float(retry_delay_seconds)))

    return GenerationResult(
        model=settings.model,
        requested_model=settings.model,
        latency_ms=latency_ms,
        settings_fingerprint=fingerprint,
        attempts=attempts_allowed,
        error=error or "Provider request failed.",
    )


def budgeted_generator(
    provider: Any,
    settings: GenerationSettings,
    budget: TokenBudget | None = None,
    *,
    counter: TokenCounter = estimate_tokens,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    secrets: tuple[str, ...] = (),
    max_attempts: int = 1,
    retry_delay_seconds: float = 0.0,
) -> Generator:
    """Wrap a provider into the ``Generator`` a mode replay calls.

    Order of operations for every request, which is what makes the cost controls
    trustworthy:

    1. count the prompt (the counter the budget enforces with);
    2. refuse *without sending* when the prompt exceeds the per-request ceiling
       — a skip, not a failure, because nothing was spent;
    3. raise :class:`~evaluation.limits.BudgetExceeded` when the run is already
       out of requests or tokens;
    4. send, then charge the budget from the provider's reported usage when it
       exists and from the counter when it does not.

    The budget is shared by every mode of an experiment on purpose: a run-level
    limit that resets per mode would let a five-mode run spend five times the
    limit a user set.
    """

    active_counter = budget.counter if budget is not None else counter

    def generate(messages: Sequence[Mapping[str, str]]) -> GenerationResult:
        prompt_tokens = count_messages(messages, active_counter)
        if budget is not None:
            fits, reason = budget.fits_prompt(prompt_tokens)
            if not fits:
                budget.record_skip()
                return GenerationResult(
                    model=settings.model,
                    requested_model=settings.model,
                    settings_fingerprint=settings.fingerprint(),
                    prompt_token_count=prompt_tokens,
                    skipped=True,
                    error=reason,
                )
            budget.check_request(prompt_tokens)
        result = generate_answer(
            provider,
            messages,
            settings,
            clock=clock,
            sleep=sleep,
            secrets=secrets,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )
        if budget is not None:
            budget.charge(
                prompt_tokens=prompt_tokens,
                completion_text=result.text,
                usage=result.usage,
                failed=not result.ok,
            )
        if result.prompt_token_count is None:
            result = _with_prompt_count(result, prompt_tokens)
        return result

    return generate


def _result_from_response(
    response: Any,
    settings: GenerationSettings,
    fingerprint: str,
    latency_ms: float | None,
    attempts: int,
) -> GenerationResult:
    """Normalize a provider response without trusting its shape."""

    text = str(getattr(response, "text", "") or "")
    model = str(getattr(response, "model", "") or "") or settings.model
    usage = _usage_dict(getattr(response, "usage", None))
    error = None if text.strip() else "Provider returned an empty completion."
    return GenerationResult(
        text=text,
        model=model,
        requested_model=settings.model,
        latency_ms=latency_ms,
        usage=usage,
        settings_fingerprint=fingerprint,
        attempts=attempts,
        error=error,
    )


def _with_prompt_count(result: GenerationResult, prompt_tokens: int) -> GenerationResult:
    return GenerationResult(
        text=result.text,
        model=result.model,
        requested_model=result.requested_model,
        latency_ms=result.latency_ms,
        usage=dict(result.usage),
        settings_fingerprint=result.settings_fingerprint,
        prompt_token_count=prompt_tokens,
        attempts=result.attempts,
        skipped=result.skipped,
        error=result.error,
    )


def _usage_dict(usage: Any) -> dict[str, int]:
    """Keep only integer token counts from a provider's usage block."""

    if not isinstance(usage, Mapping):
        return {}
    output: dict[str, int] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(name)
        if value is None or isinstance(value, bool):
            continue
        try:
            output[name] = int(value)
        except (TypeError, ValueError):
            continue
    return output


def _elapsed_ms(seconds: float) -> float:
    """Return a non-negative latency in milliseconds, rounded to microseconds."""

    return round(max(0.0, float(seconds)) * 1000.0, 3)


__all__ = [
    "GENERATION_VERSION",
    "MESSAGE_OVERHEAD_TOKENS",
    "SAMPLING_PARAMETERS",
    "GenerationResult",
    "GenerationSettings",
    "Generator",
    "MissingCredentialError",
    "ModelSpec",
    "TokenBudget",
    "api_key_from_environment",
    "budgeted_generator",
    "build_provider",
    "count_messages",
    "generate_answer",
]
