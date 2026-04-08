#!/usr/bin/env python3
"""
Baseline Rule-Based Agent for the Warehouse Dispatch Environment.

Implements a priority-based heuristic:
1. Find the highest-priority pending order (urgent first, earliest deadline first)
2. Find a free worker in the same zone as the order's product
3. If stock is sufficient: assign. If not: replenish. If no worker: try next order.
4. If nothing works: skip.

Usage:
    # Direct (no server):
    python baseline_agent.py

    # Against server:
    python baseline_agent.py --server http://localhost:8000
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import DispatchAction
from server.warehouse_environment import WarehouseEnvironment
from server.grader import grade_episode, grade_label


def run_baseline_direct(task_id: str = "easy", verbose: bool = True) -> float:
    """
    Run the baseline agent directly against the environment (no server).

    Args:
        task_id: Difficulty level — "easy", "medium", or "hard"
        verbose: Print step-by-step output

    Returns:
        Final graded score (0.0–1.0)
    """
    env = WarehouseEnvironment()
    obs = env.reset(task_id=task_id)

    if verbose:
        print(f"\n{'='*70}")
        print(f"TASK: {task_id.upper()}")
        print(f"{'='*70}")
        print(f"[Step 0] {obs.message}")
        print(f"  Orders: {obs.pending_count} pending")

    while not obs.done:
        action = decide_action(obs)

        if verbose:
            print(f"\n[Step {obs.current_step + 1}] Action: {action.action_type}", end="")
            if action.action_type == "assign":
                print(f" | Order: {action.order_id} → Worker: {action.worker_id}")
            elif action.action_type == "replenish":
                print(f" | Product: {action.product_id}")
            else:
                print()

        obs = env.step(action)

        if verbose:
            print(f"  → {obs.message}")
            print(f"  Reward: {obs.reward:+.1f} | Pending: {obs.pending_count} | "
                  f"Fulfilled: {obs.fulfilled_count} | Expired: {obs.expired_count}")

    # Get final grade
    state = env.state
    score = grade_episode(
        fulfilled_count=state.fulfilled_count,
        expired_count=state.expired_count,
        total_orders=state.total_orders,
        fulfilled_urgent=state.fulfilled_urgent,
        total_urgent=state.total_urgent,
        invalid_actions=state.invalid_actions,
        unnecessary_skips=state.unnecessary_skips,
        unnecessary_replenishes=state.unnecessary_replenishes,
        step_count=state.step_count,
        max_steps=obs.max_steps,
    )

    if verbose:
        print(f"\n{'─'*70}")
        print(f"FINAL SCORE: {score:.4f} ({grade_label(score)})")
        print(f"  Fulfilled: {state.fulfilled_count}/{state.total_orders}")
        print(f"  Expired: {state.expired_count}")
        print(f"  Invalid actions: {state.invalid_actions}")
        print(f"  Steps used: {state.step_count}/{obs.max_steps}")
        print(f"{'─'*70}")

    return score


def decide_action(obs) -> DispatchAction:
    """
    Improved rule-based decision logic.

    Strategy (optimized for HARD mode):
    1. Always try to ASSIGN first — assignments are the only way to fulfill orders
    2. Among assignable orders, prefer: urgent > earliest deadline > zone with most pending
    3. Only REPLENISH when no assignment is possible AND stock is needed for a pending order
    4. SKIP only when truly nothing productive can be done
    """
    orders = obs.orders
    workers = obs.workers
    inventory = obs.inventory

    # Build lookup maps
    product_map = {p["product_id"]: p for p in inventory}

    # Get pending orders, sorted by priority and deadline
    pending = [o for o in orders if o["status"] == "pending"]
    if not pending:
        return DispatchAction(action_type="skip")

    pending.sort(key=lambda o: (
        0 if o["priority"] == "urgent" else 1,  # urgent first
        o["deadline_step"],                       # earliest deadline first
    ))

    # Get free workers indexed by zone
    free_workers = [
        w for w in workers
        if w["busy_until_step"] <= obs.current_step
    ]
    free_by_zone = {}
    for w in free_workers:
        if w["zone"] not in free_by_zone:
            free_by_zone[w["zone"]] = w

    # ── PHASE 1: Try to assign an order ────────────────────────────────
    # Find the best assignable order (has stock + free worker in zone)
    needs_replenish = []  # Track products that need stock for pending orders
    for order in pending:
        product = product_map.get(order["product_id"])
        if product is None:
            continue

        zone = product["zone"]
        worker = free_by_zone.get(zone)
        if worker is None:
            continue  # No free worker in this zone — try next order

        if product["stock"] >= order["quantity"]:
            # Can assign immediately — do it!
            return DispatchAction(
                action_type="assign",
                order_id=order["order_id"],
                worker_id=worker["worker_id"],
            )
        else:
            # Stock insufficient — track for potential replenishment
            needs_replenish.append(order["product_id"])

    # ── PHASE 2: Replenish only if needed ──────────────────────────────
    # Only replenish products that a pending order actually needs
    # and where stock is at or below threshold
    seen = set()
    for pid in needs_replenish:
        if pid in seen:
            continue
        seen.add(pid)
        product = product_map.get(pid)
        if product and product["stock"] <= product["reorder_threshold"]:
            return DispatchAction(action_type="replenish", product_id=pid)

    # Also check if any product is below threshold and has pending orders
    for order in pending:
        product = product_map.get(order["product_id"])
        if product and product["stock"] <= product["reorder_threshold"]:
            return DispatchAction(
                action_type="replenish",
                product_id=order["product_id"],
            )

    # ── PHASE 3: Nothing to do ─────────────────────────────────────────
    return DispatchAction(action_type="skip")


def main():
    parser = argparse.ArgumentParser(description="Baseline agent for Warehouse Dispatch")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"],
                        default="all", help="Task to run")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step output")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    scores = {}

    for task_id in tasks:
        scores[task_id] = run_baseline_direct(task_id, verbose=not args.quiet)

    if len(scores) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        for task_id, score in scores.items():
            print(f"  {task_id:8s}: {score:.4f} ({grade_label(score)})")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
