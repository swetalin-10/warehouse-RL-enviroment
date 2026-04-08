# Copyright (c) 2024. All rights reserved.
# Warehouse Dispatch Environment — Deterministic Grader.
#
# Computes a normalized score (0.0–1.0) for an episode based on
# 4 weighted metrics: fulfillment, urgency, efficiency, validity.

from __future__ import annotations


def grade_episode(
    fulfilled_count: int,
    expired_count: int,
    total_orders: int,
    fulfilled_urgent: int,
    total_urgent: int,
    invalid_actions: int,
    unnecessary_skips: int,
    unnecessary_replenishes: int,
    step_count: int,
    max_steps: int,
) -> float:
    """
    Compute a deterministic grade for a completed episode.

    The score is a weighted combination of 4 metrics (each 0.0–1.0):
      - Fulfillment Rate (40%): fulfilled / total orders
      - Urgency Score (25%):    fulfilled urgent / total urgent
      - Efficiency Score (20%): 1.0 - wasted_steps / max_steps
      - Validity Score (15%):   1.0 - invalid_actions / total_actions

    Args:
        fulfilled_count: Number of orders fulfilled
        expired_count: Number of orders that expired
        total_orders: Total number of orders in the episode
        fulfilled_urgent: Number of urgent orders fulfilled
        total_urgent: Total number of urgent orders
        invalid_actions: Number of invalid actions taken
        unnecessary_skips: Skips when pending orders existed
        unnecessary_replenishes: Replenishes when stock was above threshold
        step_count: Total steps taken in the episode
        max_steps: Maximum allowed steps

    Returns:
        Float score between 0.0 and 1.0
    """
    # --- Metric 1: Fulfillment Rate (40% weight) ---
    if total_orders == 0:
        fulfillment_rate = 1.0
    else:
        fulfillment_rate = fulfilled_count / total_orders

    # --- Metric 2: Urgency Score (25% weight) ---
    if total_urgent == 0:
        urgency_score = 1.0  # No urgent orders = perfect by default
    else:
        urgency_score = fulfilled_urgent / total_urgent

    # --- Metric 3: Efficiency Score (20% weight) ---
    # Note: invalid_actions excluded here — already penalized in validity metric
    wasted_steps = unnecessary_skips + unnecessary_replenishes
    if max_steps == 0:
        efficiency_score = 1.0
    else:
        efficiency_score = max(0.0, 1.0 - wasted_steps / max_steps)

    # --- Metric 4: Validity Score (15% weight) ---
    if step_count == 0:
        validity_score = 1.0
    else:
        validity_score = max(0.0, 1.0 - invalid_actions / step_count)

    # --- Weighted combination ---
    score = (
        0.40 * fulfillment_rate
        + 0.25 * urgency_score
        + 0.20 * efficiency_score
        + 0.15 * validity_score
    )

    # Clamp to [0.0, 1.0]
    return round(max(0.0, min(1.0, score)), 4)


def grade_label(score: float) -> str:
    """Convert a numeric score to a human-readable label."""
    if score >= 0.95:
        return "PERFECT"
    elif score >= 0.70:
        return "GOOD"
    elif score >= 0.40:
        return "PARTIAL"
    elif score >= 0.10:
        return "POOR"
    else:
        return "FAIL"
