# Copyright (c) 2024. All rights reserved.
# Warehouse Dispatch Environment — Task Definitions.
#
# Defines 3 difficulty levels (easy, medium, hard) with deterministic
# data generation using seeded RNG.

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Order, Product, Worker


# =============================================================================
# Task Configuration
# =============================================================================

@dataclass
class TaskConfig:
    """Configuration for a single task/difficulty level."""
    task_id: str
    description: str
    num_orders: int
    max_steps: int
    num_urgent: int
    deadline_range: Tuple[int, int]  # (min_offset, max_offset) from order creation step
    products: List[Dict]  # pre-defined product configs
    workers: List[Dict]   # pre-defined worker configs
    seed: int = 42


# Worker templates shared across tasks
_WORKERS = [
    {"worker_id": "W-01", "name": "Alice", "zone": "A", "capacity": 5},
    {"worker_id": "W-02", "name": "Bob", "zone": "B", "capacity": 5},
    {"worker_id": "W-03", "name": "Charlie", "zone": "C", "capacity": 5},
]


# =============================================================================
# Task Definitions
# =============================================================================

TASKS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        description="Small warehouse: 5 orders, no urgency, ample stock, generous deadlines",
        num_orders=5,
        max_steps=15,
        num_urgent=0,
        deadline_range=(8, 12),
        products=[
            {"product_id": "PROD-A", "name": "Widget Alpha", "zone": "A", "stock": 20, "reorder_threshold": 3},
            {"product_id": "PROD-B", "name": "Widget Beta", "zone": "B", "stock": 20, "reorder_threshold": 3},
            {"product_id": "PROD-C", "name": "Widget Gamma", "zone": "C", "stock": 20, "reorder_threshold": 3},
        ],
        workers=_WORKERS,
        seed=42,
    ),
    "medium": TaskConfig(
        task_id="medium",
        description="Busy warehouse: 10 orders, 3 urgent, moderate deadlines, one low-stock product",
        num_orders=10,
        max_steps=20,
        num_urgent=3,
        deadline_range=(5, 8),
        products=[
            {"product_id": "PROD-A", "name": "Widget Alpha", "zone": "A", "stock": 12, "reorder_threshold": 3},
            {"product_id": "PROD-B", "name": "Widget Beta", "zone": "B", "stock": 12, "reorder_threshold": 3},
            {"product_id": "PROD-C", "name": "Widget Gamma", "zone": "C", "stock": 12, "reorder_threshold": 3},
            {"product_id": "PROD-D", "name": "Widget Delta", "zone": "A", "stock": 4, "reorder_threshold": 5},
        ],
        workers=_WORKERS,
        seed=123,
    ),
    "hard": TaskConfig(
        task_id="hard",
        description="Crisis warehouse: 15 orders, 6 urgent, tight deadlines, stock shortages",
        num_orders=15,
        max_steps=25,
        num_urgent=6,
        deadline_range=(2, 5),
        products=[
            {"product_id": "PROD-A", "name": "Widget Alpha", "zone": "A", "stock": 8, "reorder_threshold": 4},
            {"product_id": "PROD-B", "name": "Widget Beta", "zone": "B", "stock": 8, "reorder_threshold": 4},
            {"product_id": "PROD-C", "name": "Widget Gamma", "zone": "C", "stock": 8, "reorder_threshold": 4},
            {"product_id": "PROD-D", "name": "Widget Delta", "zone": "A", "stock": 3, "reorder_threshold": 5},
            {"product_id": "PROD-E", "name": "Widget Epsilon", "zone": "B", "stock": 2, "reorder_threshold": 5},
        ],
        workers=_WORKERS,
        seed=456,
    ),
}


# =============================================================================
# Deterministic Data Generation
# =============================================================================

def generate_task_data(
    task_id: str, seed: int | None = None
) -> Tuple[List[Order], List[Worker], List[Product]]:
    """
    Generate deterministic orders, workers, and products for a given task.

    Args:
        task_id: One of "easy", "medium", "hard"
        seed: Optional override seed (uses task's default if None)

    Returns:
        Tuple of (orders, workers, products)
    """
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}. Must be one of: {list(TASKS.keys())}")

    config = TASKS[task_id]
    rng = random.Random(seed if seed is not None else config.seed)

    # Create products
    products = [Product(**p) for p in config.products]
    product_ids = [p.product_id for p in products]

    # Create workers
    workers = [
        Worker(**w, current_task=None, busy_until_step=0)
        for w in config.workers
    ]

    # Create orders
    orders: List[Order] = []
    urgent_count = 0

    for i in range(config.num_orders):
        order_id = f"ORD-{i + 1:03d}"
        product_id = rng.choice(product_ids)
        quantity = rng.randint(1, 3)

        # Determine priority
        if urgent_count < config.num_urgent and (
            # Force urgent for first N orders, or randomly assign remaining
            i < config.num_urgent
            or rng.random() < 0.5
        ):
            priority = "urgent"
            urgent_count += 1
            # Urgent orders get tighter deadlines
            min_d, max_d = config.deadline_range
            deadline_offset = rng.randint(min_d, max(min_d, max_d - 2))
        else:
            priority = "normal"
            deadline_offset = rng.randint(*config.deadline_range)

        deadline_step = deadline_offset + 1  # offset from step 1

        orders.append(
            Order(
                order_id=order_id,
                product_id=product_id,
                quantity=quantity,
                priority=priority,
                deadline_step=deadline_step,
                status="pending",
            )
        )

    return orders, workers, products
