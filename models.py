# Copyright (c) 2024. All rights reserved.
# Warehouse Dispatch Environment Models.
#
# Defines all typed Pydantic models for the warehouse dispatch environment:
# - Domain models: Order, Worker, Product
# - Action: DispatchAction
# - Observation: DispatchObservation
# - State: DispatchState

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Fallback for standalone development without openenv installed
    from pydantic import ConfigDict

    class Action(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        done: bool = Field(default=False)
        reward: float | None = Field(default=None)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        model_config = ConfigDict(extra="allow", validate_assignment=True)
        episode_id: Optional[str] = Field(default=None)
        step_count: int = Field(default=0, ge=0)


# =============================================================================
# Domain Models
# =============================================================================

class Order(BaseModel):
    """A customer order to be fulfilled from warehouse stock."""
    order_id: str = Field(description="Unique order identifier, e.g. 'ORD-001'")
    product_id: str = Field(description="Product to fulfill, e.g. 'PROD-A'")
    quantity: int = Field(ge=1, description="Number of units needed")
    priority: Literal["normal", "urgent"] = Field(description="Order priority level")
    deadline_step: int = Field(ge=1, description="Must be fulfilled by this step number")
    status: Literal["pending", "fulfilled", "expired"] = Field(
        default="pending", description="Current order status"
    )


class Worker(BaseModel):
    """A warehouse worker who can be assigned to fulfill orders."""
    worker_id: str = Field(description="Unique worker identifier, e.g. 'W-01'")
    name: str = Field(description="Worker name")
    zone: Literal["A", "B", "C"] = Field(description="Zone the worker operates in")
    capacity: int = Field(ge=1, description="Max units this worker can handle per task")
    current_task: Optional[str] = Field(
        default=None, description="Order ID currently assigned, or None if free"
    )
    busy_until_step: int = Field(
        default=0, description="Step number at which worker becomes free (0 = free now)"
    )


class Product(BaseModel):
    """A product stored in the warehouse inventory."""
    product_id: str = Field(description="Unique product identifier, e.g. 'PROD-A'")
    name: str = Field(description="Product name")
    zone: Literal["A", "B", "C"] = Field(description="Storage zone in the warehouse")
    stock: int = Field(ge=0, description="Current units in stock")
    reorder_threshold: int = Field(
        ge=0, description="Stock level at or below which replenishment is needed"
    )


# =============================================================================
# Action Model — What the agent decides each step
# =============================================================================

class DispatchAction(Action):
    """
    Agent action for a single decision step.

    Three action types:
    - "assign": Assign a pending order to a free worker.
      Requires order_id and worker_id.
    - "replenish": Add stock to a product (+10 units).
      Requires product_id.
    - "skip": Do nothing this step.
    """
    action_type: Literal["assign", "replenish", "skip"] = Field(
        description="Type of action: 'assign', 'replenish', or 'skip'"
    )
    order_id: Optional[str] = Field(
        default=None, description="Order to assign (required for 'assign')"
    )
    worker_id: Optional[str] = Field(
        default=None, description="Worker to assign to (required for 'assign')"
    )
    product_id: Optional[str] = Field(
        default=None, description="Product to replenish (required for 'replenish')"
    )


# =============================================================================
# Observation Model — What the agent sees after each step
# =============================================================================

class DispatchObservation(Observation):
    """
    Full observation of the warehouse state after an action is executed.
    Provides everything the agent needs to make its next decision.
    """
    current_step: int = Field(description="Current step number in the episode")
    max_steps: int = Field(description="Maximum steps allowed in this episode")
    orders: List[Dict[str, Any]] = Field(
        default_factory=list, description="All orders with current status"
    )
    workers: List[Dict[str, Any]] = Field(
        default_factory=list, description="All workers with current availability"
    )
    inventory: List[Dict[str, Any]] = Field(
        default_factory=list, description="All products with current stock levels"
    )
    pending_count: int = Field(default=0, description="Number of orders still pending")
    fulfilled_count: int = Field(default=0, description="Number of orders fulfilled so far")
    expired_count: int = Field(default=0, description="Number of orders that missed deadline")
    invalid_actions: int = Field(default=0, description="Cumulative invalid action count")
    message: str = Field(
        default="", description="Human-readable feedback about the last action"
    )


# =============================================================================
# State Model — Episode metadata
# =============================================================================

class DispatchState(State):
    """
    Internal episode state, separate from observations.
    Used for tracking and grading.
    """
    task_id: str = Field(default="easy", description="Task difficulty: 'easy', 'medium', 'hard'")
    total_orders: int = Field(default=0, description="Total number of orders in this episode")
    fulfilled_count: int = Field(default=0, description="Orders fulfilled so far")
    expired_count: int = Field(default=0, description="Orders that expired")
    invalid_actions: int = Field(default=0, description="Cumulative invalid actions taken")
    total_urgent: int = Field(default=0, description="Total urgent orders in this episode")
    fulfilled_urgent: int = Field(default=0, description="Urgent orders fulfilled")
    unnecessary_skips: int = Field(default=0, description="Skips when pending orders existed")
    unnecessary_replenishes: int = Field(
        default=0, description="Replenishes when stock was above threshold"
    )
