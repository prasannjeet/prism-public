"""Cost layer: token totals -> USD via a per-model price table (USD per 1M tokens)."""

from __future__ import annotations

import pytest

from bench.prices import PRICES, CostMeter, ModelPrice, cost_usd


def test_cost_usd_sums_all_four_token_kinds() -> None:
    price = ModelPrice(
        input_per_mtok=1.0,
        output_per_mtok=5.0,
        cache_read_per_mtok=0.1,
        cache_write_per_mtok=1.25,
    )
    totals = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_read_tokens": 1_000_000,
        "cache_write_tokens": 1_000_000,
    }
    # 1.0 + 5.0 + 0.1 + 1.25
    assert cost_usd(totals, price) == pytest.approx(7.35)


def test_cost_usd_scales_per_million() -> None:
    price = ModelPrice(2.0, 0.0, 0.0, 0.0)
    assert cost_usd({"input_tokens": 500_000}, price) == pytest.approx(1.0)


def test_cost_usd_missing_keys_count_as_zero() -> None:
    price = ModelPrice(1.0, 1.0, 1.0, 1.0)
    assert cost_usd({}, price) == 0.0


def test_prices_table_covers_the_locked_lineup() -> None:
    for model_id in (
        "claude-haiku-4-5",
        "gpt-5.4-mini",
        "gemini-3.1-flash-lite",
        "claude-sonnet-4-6",
    ):
        assert model_id in PRICES
        assert isinstance(PRICES[model_id], ModelPrice)


def test_cost_meter_tracks_running_spend() -> None:
    meter = CostMeter(budget_usd=10.0)
    assert meter.spent == 0.0
    meter.add(3.0)
    meter.add(2.0)
    assert meter.spent == pytest.approx(5.0)


def test_cost_meter_would_exceed_is_strict_over_budget() -> None:
    meter = CostMeter(budget_usd=10.0)
    meter.add(9.0)
    assert meter.would_exceed(estimate=2.0) is True   # 9 + 2 > 10
    assert meter.would_exceed(estimate=1.0) is False  # 9 + 1 == 10, not over
