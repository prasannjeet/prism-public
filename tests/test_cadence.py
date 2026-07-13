"""The cadence packer: default joins a beat's parts into one message; C2 emits one per part.
This is the ONLY place that branches on config (fairness invariant)."""

from __future__ import annotations

from bench.cadence import pack
from bench.scenario import Beat, Expected, Part, Scenario


def _scenario() -> Scenario:
    return Scenario(
        id="x",
        workflow="form_booking",
        category="multi_answer",
        beats=(
            Beat(
                parts=(
                    Part("a move-out cleaning", "service_type", "move_out"),
                    Part("for Jane Doe", "customer_name", "Jane Doe"),
                )
            ),
            Beat(
                intent="correct",
                parts=(Part("make it window cleaning", "service_type", "window_clean"),),
            ),
            Beat(intent="request_commit", say="yes, book it"),
        ),
        expected=Expected(final_state={}, valid_completion=("committed",)),
    )


def test_default_cadence_joins_parts_per_beat() -> None:
    assert pack(_scenario(), "C5") == [
        "a move-out cleaning, for Jane Doe",
        "make it window cleaning",
        "yes, book it",
    ]


def test_c2_cadence_one_message_per_part() -> None:
    assert pack(_scenario(), "C2") == [
        "a move-out cleaning",
        "for Jane Doe",
        "make it window cleaning",
        "yes, book it",
    ]


def test_other_configs_use_default_cadence() -> None:
    for name in ("C1", "C3", "C4"):
        assert pack(_scenario(), name) == pack(_scenario(), "C5")
