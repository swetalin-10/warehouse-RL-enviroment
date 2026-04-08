# Copyright (c) 2024. All rights reserved.
# Warehouse Dispatch Environment — Core Environment Logic.
#
# Implements the WarehouseEnvironment with reset(), step(), and state().
# An AI agent acts as warehouse coordinator, deciding order assignments,
# stock replenishment, and deadline management.

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Standalone fallback — use local base classes
    from models import Action, Observation, State

    class MCPEnvironment:
        """Minimal stub for standalone development."""
        def __init__(self, *args, **kwargs):
            pass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    DispatchAction,
    DispatchObservation,
    DispatchState,
    Order,
    Product,
    Worker,
)
from server.grader import grade_episode, grade_label
from server.tasks import TASKS, generate_task_data


class WarehouseEnvironment:
    """
    Warehouse Dispatch Environment.

    An AI agent acts as a warehouse coordinator making decisions about:
    - Assigning orders to workers (matching zones, checking stock)
    - Prioritizing urgent orders
    - Replenishing low stock
    - Avoiding invalid assignments
    - Handling deadlines

    Implements the OpenEnv API: reset(), step(), state.
    """

    def __init__(self):
        """Initialize the warehouse environment."""
        self._orders: List[Order] = []
        self._workers: List[Worker] = []
        self._products: List[Product] = []
        self._state = DispatchState(episode_id=str(uuid4()))
        self._max_steps: int = 15
        self._cumulative_reward: float = 0.0
        self._episode_done: bool = False
        self._last_observation: Optional[DispatchObservation] = None

    def reset(
        self,
        seed: Optional[int] = None,
        task_id: str = "easy",
        **kwargs: Any,
    ) -> DispatchObservation:
        """
        Reset the environment for a new episode.

        Args:
            seed: Optional random seed for reproducibility
            task_id: Difficulty level — "easy", "medium", or "hard"

        Returns:
            Initial observation with full warehouse state
        """
        if task_id not in TASKS:
            task_id = "easy"

        config = TASKS[task_id]
        self._orders, self._workers, self._products = generate_task_data(task_id, seed)
        self._max_steps = config.max_steps
        self._cumulative_reward = 0.0
        self._episode_done = False
        self._last_observation = None

        # Count urgent orders
        total_urgent = sum(1 for o in self._orders if o.priority == "urgent")

        self._state = DispatchState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=task_id,
            total_orders=len(self._orders),
            fulfilled_count=0,
            expired_count=0,
            invalid_actions=0,
            total_urgent=total_urgent,
            fulfilled_urgent=0,
            unnecessary_skips=0,
            unnecessary_replenishes=0,
        )

        return self._build_observation(
            reward=0.0,
            done=False,
            message=f"Warehouse ready. Task: {task_id} ({config.description}). "
                    f"{len(self._orders)} orders, {len(self._workers)} workers, "
                    f"{len(self._products)} products. Max {self._max_steps} steps.",
        )

    def step(self, action: DispatchAction, **kwargs: Any) -> DispatchObservation:
        """
        Execute one decision step in the warehouse.

        Args:
            action: The agent's decision (assign, replenish, or skip)

        Returns:
            Observation with updated warehouse state, reward, and done flag
        """
        # Guard: prevent stepping after episode is already done
        if self._episode_done:
            if self._last_observation is not None:
                return self._last_observation
            return self._build_observation(
                reward=0.0, done=True,
                message="Episode already finished. Call /reset to start a new episode.",
            )

        reward = 0.0
        message = ""

        # ── 1. VALIDATE & PROCESS ACTION ──────────────────────────────────
        if action.action_type == "assign":
            reward, message = self._handle_assign(action)
        elif action.action_type == "replenish":
            reward, message = self._handle_replenish(action)
        elif action.action_type == "skip":
            reward, message = self._handle_skip()
        else:
            reward = -0.5
            message = f"Invalid action_type: {action.action_type}"
            self._state.invalid_actions += 1

        # ── 2. ADVANCE TIME ───────────────────────────────────────────────
        self._state.step_count += 1

        # Check for expired orders
        expire_reward, expire_msg = self._check_expirations()
        reward += expire_reward
        if expire_msg:
            message += f" | {expire_msg}"

        # Free workers whose tasks are done
        self._free_workers()

        # Accumulate reward
        self._cumulative_reward += reward

        # ── 3. CHECK TERMINATION ──────────────────────────────────────────
        pending = [o for o in self._orders if o.status == "pending"]
        done = (
            self._state.step_count >= self._max_steps
            or len(pending) == 0
        )
        self._episode_done = done

        # If episode ended, compute final grade and append to message
        if done:
            score = grade_episode(
                fulfilled_count=self._state.fulfilled_count,
                expired_count=self._state.expired_count,
                total_orders=self._state.total_orders,
                fulfilled_urgent=self._state.fulfilled_urgent,
                total_urgent=self._state.total_urgent,
                invalid_actions=self._state.invalid_actions,
                unnecessary_skips=self._state.unnecessary_skips,
                unnecessary_replenishes=self._state.unnecessary_replenishes,
                step_count=self._state.step_count,
                max_steps=self._max_steps,
            )
            label = grade_label(score)
            message += (
                f" | EPISODE COMPLETE — Score: {score:.4f} ({label}). "
                f"Fulfilled: {self._state.fulfilled_count}/{self._state.total_orders}, "
                f"Expired: {self._state.expired_count}, "
                f"Invalid: {self._state.invalid_actions}."
            )

        # ── 4. RETURN OBSERVATION ─────────────────────────────────────────
        obs = self._build_observation(reward=reward, done=done, message=message)
        if done:
            self._last_observation = obs
        return obs

    # ─── ACTION HANDLERS ────────────────────────────────────────────────

    def _handle_assign(self, action: DispatchAction) -> tuple[float, str]:
        """Handle an 'assign' action."""
        # Validate required fields
        if not action.order_id or not action.worker_id:
            self._state.invalid_actions += 1
            return -0.5, "INVALID: 'assign' requires order_id and worker_id."

        # Find order
        order = self._find_order(action.order_id)
        if order is None:
            self._state.invalid_actions += 1
            return -0.5, f"INVALID: Order '{action.order_id}' not found."

        if order.status != "pending":
            self._state.invalid_actions += 1
            return -0.5, f"INVALID: Order '{action.order_id}' is '{order.status}', not pending."

        # Find worker
        worker = self._find_worker(action.worker_id)
        if worker is None:
            self._state.invalid_actions += 1
            return -0.5, f"INVALID: Worker '{action.worker_id}' not found."

        if worker.busy_until_step > self._state.step_count:
            self._state.invalid_actions += 1
            return -0.5, f"INVALID: Worker '{action.worker_id}' busy until step {worker.busy_until_step}."

        # Find product
        product = self._find_product(order.product_id)
        if product is None:
            self._state.invalid_actions += 1
            return -0.5, f"INVALID: Product '{order.product_id}' not found."

        # Check zone match
        if worker.zone != product.zone:
            self._state.invalid_actions += 1
            return (
                -0.5,
                f"INVALID: Worker '{action.worker_id}' is in zone {worker.zone}, "
                f"but product '{order.product_id}' is in zone {product.zone}.",
            )

        # Check worker capacity
        if order.quantity > worker.capacity:
            self._state.invalid_actions += 1
            return (
                -0.5,
                f"INVALID: Order needs {order.quantity} units but worker "
                f"'{worker.worker_id}' capacity is {worker.capacity}.",
            )

        # Check stock
        if product.stock < order.quantity:
            self._state.invalid_actions += 1
            return (
                -0.5,
                f"INVALID: Product '{order.product_id}' has {product.stock} units, "
                f"but order needs {order.quantity}.",
            )

        # ── All checks passed — fulfill the order ──
        order.status = "fulfilled"
        worker.current_task = order.order_id
        worker.busy_until_step = self._state.step_count + 1 + 1  # current step + processing time
        product.stock -= order.quantity
        self._state.fulfilled_count += 1

        # Calculate reward
        reward = 2.0 if order.priority == "urgent" else 1.0
        if order.priority == "urgent":
            self._state.fulfilled_urgent += 1

        # Early bonus: fulfilled well before deadline
        steps_remaining = order.deadline_step - (self._state.step_count + 1)
        if steps_remaining >= 2:
            reward += 0.5

        return reward, (
            f"ASSIGNED: Order '{order.order_id}' → Worker '{worker.worker_id}'. "
            f"Priority: {order.priority}. Reward: {reward:+.1f}"
        )

    def _handle_replenish(self, action: DispatchAction) -> tuple[float, str]:
        """Handle a 'replenish' action."""
        if not action.product_id:
            self._state.invalid_actions += 1
            return -0.5, "INVALID: 'replenish' requires product_id."

        product = self._find_product(action.product_id)
        if product is None:
            self._state.invalid_actions += 1
            return -0.5, f"INVALID: Product '{action.product_id}' not found."

        if product.stock <= product.reorder_threshold:
            product.stock += 10
            return 0.3, (
                f"REPLENISHED: Product '{action.product_id}' stock {product.stock - 10} → {product.stock}. "
                f"Needed (below threshold {product.reorder_threshold}). Reward: +0.3"
            )
        else:
            product.stock += 10
            self._state.unnecessary_replenishes += 1
            return -0.2, (
                f"REPLENISHED: Product '{action.product_id}' stock {product.stock - 10} → {product.stock}. "
                f"Unnecessary (above threshold {product.reorder_threshold}). Reward: -0.2"
            )

    def _handle_skip(self) -> tuple[float, str]:
        """Handle a 'skip' action."""
        pending = [o for o in self._orders if o.status == "pending"]
        if len(pending) > 0:
            self._state.unnecessary_skips += 1
            return -0.1, f"SKIPPED: {len(pending)} orders still pending. Reward: -0.1"
        else:
            return 0.0, "SKIPPED: No pending orders. Reward: 0.0"

    # ─── WORLD UPDATES ──────────────────────────────────────────────────

    def _check_expirations(self) -> tuple[float, str]:
        """Check and expire orders past their deadline."""
        expired_now: List[str] = []
        for order in self._orders:
            if (
                order.status == "pending"
                and order.deadline_step <= self._state.step_count
            ):
                order.status = "expired"
                self._state.expired_count += 1
                expired_now.append(order.order_id)

        if expired_now:
            penalty = -1.5 * len(expired_now)
            return penalty, f"EXPIRED: {', '.join(expired_now)} ({penalty:+.1f})"
        return 0.0, ""

    def _free_workers(self) -> None:
        """Free workers whose tasks are complete."""
        for worker in self._workers:
            if worker.busy_until_step <= self._state.step_count:
                worker.current_task = None

    # ─── LOOKUP HELPERS ──────────────────────────────────────────────────

    def _find_order(self, order_id: str) -> Optional[Order]:
        for o in self._orders:
            if o.order_id == order_id:
                return o
        return None

    def _find_worker(self, worker_id: str) -> Optional[Worker]:
        for w in self._workers:
            if w.worker_id == worker_id:
                return w
        return None

    def _find_product(self, product_id: str) -> Optional[Product]:
        for p in self._products:
            if p.product_id == product_id:
                return p
        return None

    # ─── OBSERVATION BUILDER ─────────────────────────────────────────────

    def _build_observation(
        self, reward: float, done: bool, message: str
    ) -> DispatchObservation:
        """Build a full observation from current world state."""
        pending = [o for o in self._orders if o.status == "pending"]
        fulfilled = [o for o in self._orders if o.status == "fulfilled"]
        expired = [o for o in self._orders if o.status == "expired"]

        return DispatchObservation(
            done=done,
            reward=reward,
            current_step=self._state.step_count,
            max_steps=self._max_steps,
            orders=[o.model_dump() for o in self._orders],
            workers=[w.model_dump() for w in self._workers],
            inventory=[p.model_dump() for p in self._products],
            pending_count=len(pending),
            fulfilled_count=len(fulfilled),
            expired_count=len(expired),
            invalid_actions=self._state.invalid_actions,
            message=message,
            metadata={
                "cumulative_reward": round(self._cumulative_reward, 2),
                "task_id": self._state.task_id,
            },
        )

    @property
    def state(self) -> DispatchState:
        """Get the current environment state."""
        return self._state
