"""Infra-only retry/backoff wrapper for the live runner.

Retries transient infrastructure failures (HTTP 5xx / 429 / timeout) only.
Model refusals or odd-but-valid responses are DATA and must never be retried
Backoff is deterministic; `sleep` is injected for testability."""

from __future__ import annotations

import time
from collections.abc import Callable

from prism.models import Message, ModelAdapter, ModelResponse, ModelSettings, ToolSpec

# Strongly billing-specific signals in a provider error message (lowercased). Kept
# narrow so a transient rate-limit ("rate limit exceeded") is never mistaken for an
# out-of-credits stop: "credit balance" (Anthropic), "insufficient_quota"/"billing"
# (OpenAI's quota message says "...check your plan and billing details"; Gemini's
# billing-disabled says "Billing account ... is disabled").
_BILLING_PHRASES = ("credit balance", "insufficient_quota", "insufficient funds", "billing")


def _http_status(exc: BaseException) -> int | None:
    """The HTTP status off an SDK exception. Anthropic/OpenAI expose `.status_code`;
    google-genai exposes `.code`. Returns None if neither is an int."""
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def is_infra_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    status = _http_status(exc)
    # 429 (rate limit) or ANY 5xx (500-504, plus 529 overloaded and other transient server
    # codes) is retryable infra. A range check avoids whack-a-mole as providers add 5xx codes.
    return status is not None and (status == 429 or status >= 500)


def is_billing_error(exc: BaseException) -> bool:
    """True for an out-of-credits / quota / payment rejection that will recur on retry.

    Detected by HTTP 402 (Payment Required) or a billing-specific phrase in the message.
    Deliberately conservative: a plain rate-limit 429 is transient infra, not a billing stop."""
    if _http_status(exc) == 402:
        return True
    message = str(exc).lower()
    return any(phrase in message for phrase in _BILLING_PHRASES)


def is_retryable_infra(exc: BaseException) -> bool:
    """The runner's default retry predicate: transient infra, but never a billing rejection
    (retrying an out-of-credits 429 just burns calls against an empty account)."""
    return is_infra_error(exc) and not is_billing_error(exc)


class RetryingAdapter:
    """Wraps a ModelAdapter; retries infra failures with deterministic exponential
    backoff. Re-raises non-infra errors immediately (they are data, not failures)."""

    name: str
    settings: ModelSettings

    def __init__(
        self,
        base: ModelAdapter,
        *,
        max_attempts: int = 3,
        is_retryable: Callable[[BaseException], bool] = is_retryable_infra,
        sleep: Callable[[float], None] = time.sleep,
        backoff_base: float = 1.0,
    ) -> None:
        self._base = base
        self.name = base.name
        self.settings = base.settings
        self._max_attempts = max_attempts
        self._is_retryable = is_retryable
        self._sleep = sleep
        self._backoff_base = backoff_base

    def complete(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelResponse:
        for attempt in range(self._max_attempts):
            try:
                return self._base.complete(system, messages, tools)
            except Exception as exc:  # noqa: BLE001 - re-raised below unless retryable
                last = attempt == self._max_attempts - 1
                if last or not self._is_retryable(exc):
                    raise
                self._sleep(self._backoff_base * (2**attempt))
        raise AssertionError("unreachable: loop returns or raises")
