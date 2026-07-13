"""Infra-only retry for the runner. Classifier + RetryingAdapter wrapper."""

from __future__ import annotations

from bench.resilience import RetryingAdapter, is_billing_error, is_infra_error
from prism.models import Message, ModelResponse, ModelSettings, Usage


class _FakeHTTPError(Exception):
    def __init__(self, status_code: int, message: str = "") -> None:
        super().__init__(message or f"http {status_code}")
        self.status_code = status_code


class _FakeGoogleError(Exception):
    """google-genai exposes the HTTP status as `.code`, not `.status_code`."""

    def __init__(self, code: int, message: str = "") -> None:
        super().__init__(message or f"google {code}")
        self.code = code


def test_5xx_and_429_are_infra() -> None:
    assert is_infra_error(_FakeHTTPError(500)) is True
    assert is_infra_error(_FakeHTTPError(503)) is True
    assert is_infra_error(_FakeHTTPError(429)) is True


def test_overloaded_529_and_other_5xx_are_infra() -> None:
    # Anthropic 529 (overloaded_error) is retryable but outside 500-504; any 5xx must count.
    assert is_infra_error(_FakeHTTPError(529)) is True
    assert is_infra_error(_FakeHTTPError(521)) is True
    assert is_infra_error(_FakeGoogleError(529)) is True
    assert is_infra_error(_FakeHTTPError(499)) is False  # 4xx (other than 429) is not infra


def test_google_style_code_attribute_is_recognized() -> None:
    # google-genai errors carry .code (not .status_code); a 502 must be retryable infra,
    # not an unrecognized error that crashes the whole run.
    assert is_infra_error(_FakeGoogleError(502)) is True
    assert is_infra_error(_FakeGoogleError(503)) is True
    assert is_infra_error(_FakeGoogleError(429)) is True
    assert is_infra_error(_FakeGoogleError(400)) is False
    assert is_billing_error(_FakeGoogleError(402, "payment required")) is True


def test_timeout_is_infra() -> None:
    assert is_infra_error(TimeoutError("slow")) is True


def test_4xx_and_plain_errors_are_not_infra() -> None:
    assert is_infra_error(_FakeHTTPError(400)) is False
    assert is_infra_error(ValueError("bad arg")) is False


def test_billing_errors_are_detected_across_providers() -> None:
    # Anthropic: 400 with a credit-balance message.
    anthropic = _FakeHTTPError(400, "Your credit balance is too low to access the Anthropic API.")
    # OpenAI: 429 whose body is the insufficient_quota / billing message.
    openai = _FakeHTTPError(429, "You exceeded your current quota, check your plan and billing.")
    # Gemini / generic: 402 Payment Required, or a billing-disabled message.
    payment_required = _FakeHTTPError(402, "payment required")
    gemini = _FakeHTTPError(403, "Billing account for the project is disabled.")
    assert is_billing_error(anthropic) is True
    assert is_billing_error(openai) is True
    assert is_billing_error(payment_required) is True
    assert is_billing_error(gemini) is True


def test_transient_limits_are_not_billing() -> None:
    # A plain rate-limit 429 or a 5xx is transient infra, NOT a billing stop.
    assert is_billing_error(_FakeHTTPError(429, "Rate limit exceeded, retry shortly.")) is False
    assert is_billing_error(_FakeHTTPError(503, "service unavailable")) is False
    assert is_billing_error(TimeoutError("slow")) is False


class _FlakyAdapter:
    """Raises `exc` for the first `fail_times` calls, then returns a response."""

    name = "fake:flaky"
    settings = ModelSettings()

    def __init__(self, exc: Exception, fail_times: int) -> None:
        self._exc = exc
        self._fail_times = fail_times
        self.calls = 0

    def complete(self, system, messages, tools):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return ModelResponse(text="ok", tool_calls=(), usage=Usage(1, 1, 0, 0), stop_reason="end")


def test_retries_infra_error_then_succeeds() -> None:
    base = _FlakyAdapter(_FakeHTTPError(503), fail_times=2)
    adapter = RetryingAdapter(base, max_attempts=3, sleep=lambda _s: None)
    resp = adapter.complete("sys", [Message(role="user", text="x")], [])
    assert resp.text == "ok"
    assert base.calls == 3


def test_gives_up_after_max_attempts_and_reraises() -> None:
    base = _FlakyAdapter(_FakeHTTPError(500), fail_times=5)
    adapter = RetryingAdapter(base, max_attempts=3, sleep=lambda _s: None)
    try:
        adapter.complete("sys", [Message(role="user", text="x")], [])
        raise AssertionError("expected the infra error to propagate")
    except _FakeHTTPError:
        pass
    assert base.calls == 3


def test_does_not_retry_non_infra_error() -> None:
    base = _FlakyAdapter(ValueError("model said no"), fail_times=5)
    adapter = RetryingAdapter(base, max_attempts=3, sleep=lambda _s: None)
    try:
        adapter.complete("sys", [Message(role="user", text="x")], [])
        raise AssertionError("expected the ValueError to propagate immediately")
    except ValueError:
        pass
    assert base.calls == 1


def test_does_not_retry_billing_error_even_on_429() -> None:
    # An out-of-credits 429 looks transient by status alone, but must NOT be retried:
    # retrying just burns calls against an empty account. It propagates on the first try.
    billing = _FakeHTTPError(429, "You exceeded your current quota; check your billing.")
    base = _FlakyAdapter(billing, fail_times=5)
    adapter = RetryingAdapter(base, max_attempts=3, sleep=lambda _s: None)
    try:
        adapter.complete("sys", [Message(role="user", text="x")], [])
        raise AssertionError("expected the billing error to propagate immediately")
    except _FakeHTTPError:
        pass
    assert base.calls == 1
