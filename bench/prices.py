"""Per-model price table + token->USD cost + a running budget meter.

Prices are USD per 1,000,000 tokens. They are DATA, snapshotted per run to
results/<run_id>/prices.yaml, so a correction is a one-line edit + re-aggregate,
never a re-run. The values below are PROVISIONAL — verify against each provider's
published pricing before any billed run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0


# PROVISIONAL rates (USD / 1M tokens) — confirm before billing.
PRICES: dict[str, ModelPrice] = {
    "claude-haiku-4-5": ModelPrice(1.0, 5.0, 0.1, 1.25),
    "gpt-5.4-mini": ModelPrice(0.25, 2.0, 0.0, 0.0),
    "gemini-3.1-flash-lite": ModelPrice(0.1, 0.4, 0.0, 0.0),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0, 0.3, 3.75),
}


def cost_usd(totals: Mapping[str, int], price: ModelPrice) -> float:
    """Dollar cost of one conversation's token totals. Missing keys count as zero."""
    return (
        totals.get("input_tokens", 0) * price.input_per_mtok
        + totals.get("output_tokens", 0) * price.output_per_mtok
        + totals.get("cache_read_tokens", 0) * price.cache_read_per_mtok
        + totals.get("cache_write_tokens", 0) * price.cache_write_per_mtok
    ) / _PER_MILLION


def price_table_dict(prices: Mapping[str, ModelPrice]) -> dict[str, dict[str, float]]:
    """JSON/YAML-able snapshot of a price table (for results/<run_id>/prices.yaml)."""
    return {model_id: asdict(price) for model_id, price in prices.items()}


class UnknownModelPriceError(Exception):
    """A model in a billed run plan has no PRICES entry (would silently meter as $0)."""


class CostMeter:
    """Running spend across completed cells + a budget guard. The runner asks
    would_exceed(estimate) BEFORE starting a cell and stops gracefully if True."""

    def __init__(self, budget_usd: float) -> None:
        self._budget = budget_usd
        self._spent = 0.0

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def budget(self) -> float:
        return self._budget

    def add(self, cost: float) -> None:
        self._spent += cost

    def would_exceed(self, estimate: float) -> bool:
        return self._spent + estimate > self._budget
