#!/usr/bin/env python3
"""
Smoke test for the Warehouse Dispatch Environment.

Tests the environment directly (no HTTP server) to verify:
- reset() works for all 3 tasks
- step() handles valid and invalid actions
- Grader produces deterministic scores
- State tracking is correct

Usage:
    python test_env.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import DispatchAction
from server.warehouse_environment import WarehouseEnvironment
from server.grader import grade_episode, grade_label
from server.tasks import TASKS


def test_reset_all_tasks():
    """Test that reset() works for all 3 task definitions."""
    env = WarehouseEnvironment()
    print("TEST: reset() for all tasks")

    for task_id in ["easy", "medium", "hard"]:
        obs = env.reset(task_id=task_id)
        config = TASKS[task_id]

        assert not obs.done, f"Episode should not be done on reset for {task_id}"
        assert obs.reward == 0.0, f"Initial reward should be 0 for {task_id}"
        assert obs.current_step == 0, f"Step should be 0 on reset for {task_id}"
        assert obs.max_steps == config.max_steps
        assert obs.pending_count == config.num_orders
        assert obs.fulfilled_count == 0
        assert obs.expired_count == 0
        assert len(obs.orders) == config.num_orders
        assert len(obs.workers) == len(config.workers)
        assert len(obs.inventory) == len(config.products)

        print(f"  ✓ {task_id}: {obs.pending_count} orders, "
              f"{len(obs.workers)} workers, {len(obs.inventory)} products")

    print("  PASSED\n")


def test_valid_assign():
    """Test a valid assignment action."""
    env = WarehouseEnvironment()
    obs = env.reset(task_id="easy")
    print("TEST: valid assign action")

    # Find a pending order and matching worker
    order = obs.orders[0]
    product_id = order["product_id"]
    product_zone = None
    for p in obs.inventory:
        if p["product_id"] == product_id:
            product_zone = p["zone"]
            break

    # Find worker in same zone
    worker_id = None
    for w in obs.workers:
        if w["zone"] == product_zone:
            worker_id = w["worker_id"]
            break

    action = DispatchAction(
        action_type="assign",
        order_id=order["order_id"],
        worker_id=worker_id,
    )
    obs = env.step(action)

    assert obs.reward > 0, f"Valid assign should give positive reward, got {obs.reward}"
    assert obs.fulfilled_count == 1, "Fulfilled count should be 1"
    assert obs.current_step == 1, "Step should be 1"
    assert obs.invalid_actions == 0, "No invalid actions expected"

    print(f"  ✓ Assigned {order['order_id']} → {worker_id}, reward={obs.reward}")
    print("  PASSED\n")


def test_invalid_actions():
    """Test that invalid actions are caught correctly."""
    env = WarehouseEnvironment()
    env.reset(task_id="easy")
    print("TEST: invalid actions")

    # Invalid: assign to non-existent order
    action = DispatchAction(
        action_type="assign",
        order_id="FAKE-001",
        worker_id="W-01",
    )
    obs = env.step(action)
    assert obs.reward == -0.5, f"Invalid assign should give -0.5, got {obs.reward}"
    assert obs.invalid_actions == 1
    print(f"  ✓ Fake order: reward={obs.reward}, invalid={obs.invalid_actions}")

    # Invalid: replenish non-existent product
    action = DispatchAction(
        action_type="replenish",
        product_id="FAKE-PROD",
    )
    obs = env.step(action)
    assert obs.reward == -0.5, f"Invalid replenish should give -0.5, got {obs.reward}"
    assert obs.invalid_actions == 2
    print(f"  ✓ Fake product: reward={obs.reward}, invalid={obs.invalid_actions}")

    # Invalid: assign without required fields
    action = DispatchAction(
        action_type="assign",
    )
    obs = env.step(action)
    assert obs.reward == -0.5
    assert obs.invalid_actions == 3
    print(f"  ✓ Missing fields: reward={obs.reward}, invalid={obs.invalid_actions}")

    print("  PASSED\n")


def test_skip_action():
    """Test skip action behavior."""
    env = WarehouseEnvironment()
    env.reset(task_id="easy")
    print("TEST: skip action")

    action = DispatchAction(action_type="skip")
    obs = env.step(action)

    assert obs.reward == -0.1, f"Skip with pending orders should give -0.1, got {obs.reward}"
    print(f"  ✓ Skip with pending: reward={obs.reward}")
    print("  PASSED\n")


def test_replenish_action():
    """Test replenish action behavior."""
    env = WarehouseEnvironment()
    env.reset(task_id="medium")  # Medium has low-stock product PROD-D
    print("TEST: replenish action")

    # Replenish low-stock product
    action = DispatchAction(action_type="replenish", product_id="PROD-D")
    obs = env.step(action)
    assert obs.reward == 0.3, f"Needed replenish should give +0.3, got {obs.reward}"
    print(f"  ✓ Needed replenish (PROD-D): reward={obs.reward}")

    # Replenish well-stocked product
    action = DispatchAction(action_type="replenish", product_id="PROD-A")
    obs = env.step(action)
    assert obs.reward == -0.2, f"Unnecessary replenish should give -0.2, got {obs.reward}"
    print(f"  ✓ Unnecessary replenish (PROD-A): reward={obs.reward}")

    print("  PASSED\n")


def test_state_tracking():
    """Test that state is tracked correctly."""
    env = WarehouseEnvironment()
    env.reset(task_id="easy")
    print("TEST: state tracking")

    state = env.state
    assert state.step_count == 0
    assert state.task_id == "easy"
    assert state.total_orders == 5
    assert state.fulfilled_count == 0

    # Take a step
    env.step(DispatchAction(action_type="skip"))
    state = env.state
    assert state.step_count == 1
    print(f"  ✓ State: step={state.step_count}, task={state.task_id}, "
          f"orders={state.total_orders}")
    print("  PASSED\n")


def test_deterministic_grader():
    """Test that the grader is deterministic."""
    print("TEST: deterministic grader")

    score1 = grade_episode(3, 1, 5, 1, 2, 0, 0, 0, 10, 15)
    score2 = grade_episode(3, 1, 5, 1, 2, 0, 0, 0, 10, 15)
    assert score1 == score2, "Grader should be deterministic"

    # Perfect score
    perfect = grade_episode(5, 0, 5, 2, 2, 0, 0, 0, 5, 15)
    assert perfect == 1.0, f"Perfect episode should score 1.0, got {perfect}"

    # All failed — every metric should be near 0
    fail = grade_episode(0, 5, 5, 0, 2, 10, 5, 5, 10, 10)
    assert fail < 0.2, f"Failed episode should score < 0.2, got {fail}"

    print(f"  ✓ Deterministic: {score1} == {score2}")
    print(f"  ✓ Perfect: {perfect}")
    print(f"  ✓ Fail: {fail}")
    print("  PASSED\n")


def test_episode_completion():
    """Test that episodes actually terminate."""
    env = WarehouseEnvironment()
    print("TEST: episode completion")

    for task_id in ["easy", "medium", "hard"]:
        obs = env.reset(task_id=task_id)
        steps = 0

        while not obs.done:
            obs = env.step(DispatchAction(action_type="skip"))
            steps += 1
            assert steps <= TASKS[task_id].max_steps + 1, "Episode should not exceed max_steps"

        assert obs.done, "Episode should be done"
        print(f"  ✓ {task_id}: terminated after {steps} steps")

    print("  PASSED\n")


def main():
    print("=" * 70)
    print("WAREHOUSE DISPATCH ENVIRONMENT — SMOKE TESTS")
    print("=" * 70 + "\n")

    tests = [
        test_reset_all_tasks,
        test_valid_assign,
        test_invalid_actions,
        test_skip_action,
        test_replenish_action,
        test_state_tracking,
        test_deterministic_grader,
        test_episode_completion,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
