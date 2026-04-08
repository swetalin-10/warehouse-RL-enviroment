#!/usr/bin/env python3
"""
Warehouse Dispatch Environment — Evaluation Script.

Runs the baseline agent on all 3 tasks (easy, medium, hard) and prints
a clear summary of scores and metrics.

Usage:
    python evaluate.py
    python evaluate.py --task medium
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import DispatchAction
from server.warehouse_environment import WarehouseEnvironment
from server.grader import grade_episode, grade_label
from server.tasks import TASKS


def evaluate_task(task_id: str) -> dict:
    """Run baseline agent on a single task and return metrics."""
    env = WarehouseEnvironment()
    obs = env.reset(task_id=task_id)

    steps = 0
    total_reward = 0.0

    while not obs.done:
        action = decide_action(obs)
        obs = env.step(action)
        total_reward += obs.reward
        steps += 1

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

    return {
        "task_id": task_id,
        "score": score,
        "label": grade_label(score),
        "fulfilled": state.fulfilled_count,
        "total_orders": state.total_orders,
        "expired": state.expired_count,
        "invalid_actions": state.invalid_actions,
        "steps_used": state.step_count,
        "max_steps": obs.max_steps,
        "total_reward": round(total_reward, 2),
    }


def decide_action(obs) -> DispatchAction:
    """Improved baseline agent: assign-first, replenish-only-when-needed."""
    product_map = {p["product_id"]: p for p in obs.inventory}

    pending = [o for o in obs.orders if o["status"] == "pending"]
    if not pending:
        return DispatchAction(action_type="skip")

    pending.sort(key=lambda o: (
        0 if o["priority"] == "urgent" else 1,
        o["deadline_step"],
    ))

    free_workers = [w for w in obs.workers if w["busy_until_step"] <= obs.current_step]
    free_by_zone = {}
    for w in free_workers:
        if w["zone"] not in free_by_zone:
            free_by_zone[w["zone"]] = w

    # Phase 1: Try to assign
    needs_replenish = []
    for order in pending:
        product = product_map.get(order["product_id"])
        if product is None:
            continue
        zone = product["zone"]
        worker = free_by_zone.get(zone)
        if worker is None:
            continue
        if product["stock"] >= order["quantity"]:
            return DispatchAction(
                action_type="assign",
                order_id=order["order_id"],
                worker_id=worker["worker_id"],
            )
        else:
            needs_replenish.append(order["product_id"])

    # Phase 2: Replenish only if needed for a pending order
    seen = set()
    for pid in needs_replenish:
        if pid in seen:
            continue
        seen.add(pid)
        product = product_map.get(pid)
        if product and product["stock"] <= product["reorder_threshold"]:
            return DispatchAction(action_type="replenish", product_id=pid)

    for order in pending:
        product = product_map.get(order["product_id"])
        if product and product["stock"] <= product["reorder_threshold"]:
            return DispatchAction(action_type="replenish", product_id=order["product_id"])

    return DispatchAction(action_type="skip")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Warehouse Dispatch Environment")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"],
                        default="all", help="Task(s) to evaluate")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    print("=" * 60)
    print("  WAREHOUSE DISPATCH ENVIRONMENT - EVALUATION")
    print("=" * 60)

    results = []
    for task_id in tasks:
        r = evaluate_task(task_id)
        results.append(r)

        print(f"\n  Task: {task_id.upper()}")
        print(f"  {'─' * 50}")
        print(f"  Score:      {r['score']:.4f}  ({r['label']})")
        print(f"  Fulfilled:  {r['fulfilled']}/{r['total_orders']}")
        print(f"  Expired:    {r['expired']}")
        print(f"  Invalid:    {r['invalid_actions']}")
        print(f"  Steps:      {r['steps_used']}/{r['max_steps']}")
        print(f"  Reward:     {r['total_reward']:+.2f}")

    if len(results) > 1:
        avg_score = sum(r["score"] for r in results) / len(results)
        print(f"\n{'=' * 60}")
        print(f"  SUMMARY")
        print(f"{'─' * 60}")
        for r in results:
            print(f"  {r['task_id']:8s}  {r['score']:.4f}  {r['label']}")
        print(f"{'─' * 60}")
        print(f"  {'Average':8s}  {avg_score:.4f}  ({grade_label(avg_score)})")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
