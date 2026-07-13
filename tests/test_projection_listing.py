from __future__ import annotations

from bench.projection_listing import render_example


def test_example_has_tier1_header_and_active_branch() -> None:
    text = render_example()
    assert "=== WORKFLOW STRUCTURE ===" in text  # Tier 1
    assert "Active Branch: service_type" in text  # Tier 2, on the move_out branch
    assert "Complete: No" in text  # mid-form
    assert 'service_type = "move_out"' in text  # answered value rendered
