"""Render a scenario's beats into per-config user messages. C2 (next-step-only) sees one value
per turn; every other config gets the front-loaded message (a beat's parts joined). This is the
generalized, automatic form of the manual harness's stepwise_user_turns."""

from __future__ import annotations

from .scenario import Scenario


def pack(scenario: Scenario, config_name: str) -> list[str]:
    messages: list[str] = []
    for beat in scenario.beats:
        if not beat.parts:
            messages.append(beat.say)
        elif config_name == "C2":
            messages.extend(part.say for part in beat.parts)
        else:
            messages.append(", ".join(part.say for part in beat.parts))
    return messages
