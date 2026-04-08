---
title: Warehouse Dispatch Environment
emoji: 📦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Warehouse Dispatch Environment

An **OpenEnv environment** simulating real-world warehouse dispatch coordination. An AI agent acts as a warehouse dispatcher, making sequential decisions to assign customer orders to workers, manage inventory levels, and meet delivery deadlines under resource constraints.

## The Problem

Every warehouse and fulfillment center faces the same daily challenge: hundreds of orders arrive with varying urgency, limited workers operate in designated zones, and inventory fluctuates constantly. A dispatcher must continuously decide:

- **Which order to fulfill next** when multiple are pending
- **Which worker to assign** based on zone compatibility and availability
- **When to pause fulfillment to replenish stock** before it runs out
- **How to triage** when deadlines are impossible to all meet simultaneously

Poor dispatching leads to missed deadlines, idle workers, stockouts, and wasted labor. This environment captures these trade-offs in a simplified but realistic simulation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation (all tasks, deterministic)
python evaluate.py

# Start HTTP server
python app.py
# API docs: http://localhost:8000/docs

# Docker
docker build -t warehouse-dispatch-env .
docker run -p 8000:8000 warehouse-dispatch-env
```

## Environment Overview

The agent controls a warehouse with:

| Entity       | Description |
|:-------------|:------------|
| **Orders**   | Customer requests with product, quantity, priority (normal/urgent), and a hard deadline (step number) |
| **Workers**  | Staff assigned to zones (A, B, C). Can only fulfill orders for products in their zone. Busy for 1 step after assignment |
| **Products** | Inventory items in specific zones with finite stock. Depletes on fulfillment, can be replenished (+10 units) |

Each step, the agent observes the full warehouse state and makes **exactly one decision**.

## Tasks (Progressive Difficulty)

| Task | Name | Orders | Urgent | Deadlines | Stock | Challenge |
|:-----|:-----|:------:|:------:|:---------:|:-----:|:----------|
| **easy** | Small Warehouse | 5 | 0 | 8-12 steps | Ample (20) | Basic zone matching. No urgency, no edge cases. |
| **medium** | Busy Warehouse | 10 | 3 | 5-8 steps | Mixed (4-12) | Prioritize urgent orders. Manage worker scheduling. One low-stock product. |
| **hard** | Crisis Warehouse | 15 | 6 | 2-5 steps | Scarce (2-8) | Triage under pressure. Conflicting priorities. Stock shortages force trade-offs. Not all orders can be saved. |

## Action Space

The agent submits one action per step:

### `assign` -- Assign an order to a worker
```json
{"action_type": "assign", "order_id": "ORD-001", "worker_id": "W-01"}
```
**Constraints:** Order must be pending. Worker must be free. Worker's zone must match product zone. Sufficient stock required.

### `replenish` -- Restock a product (+10 units)
```json
{"action_type": "replenish", "product_id": "PROD-A"}
```
Positive reward only when stock is at or below the reorder threshold.

### `skip` -- Do nothing this step
```json
{"action_type": "skip"}
```
Small penalty if there are still pending orders (discourages inaction).

## Observation Space

After each step, the agent receives:

| Field | Type | Description |
|:------|:-----|:------------|
| `current_step` | int | Current step in the episode |
| `max_steps` | int | Maximum steps before episode ends |
| `orders` | list[dict] | All orders with `order_id`, `product_id`, `quantity`, `priority`, `deadline_step`, `status` |
| `workers` | list[dict] | All workers with `worker_id`, `name`, `zone`, `capacity`, `current_task`, `busy_until_step` |
| `inventory` | list[dict] | All products with `product_id`, `name`, `zone`, `stock`, `reorder_threshold` |
| `pending_count` | int | Orders still waiting |
| `fulfilled_count` | int | Orders completed |
| `expired_count` | int | Orders that missed deadline |
| `invalid_actions` | int | Cumulative invalid action count |
| `message` | str | Human-readable feedback on last action |
| `reward` | float | Reward for last action |
| `done` | bool | Whether the episode has ended |

## Reward Function

| Event | Reward | Rationale |
|:------|-------:|:----------|
| Fulfill normal order | **+1.0** | Base completion incentive |
| Fulfill urgent order | **+2.0** | Urgency premium |
| Fulfill early (2+ steps before deadline) | **+0.5** | Proactive bonus |
| Order expires | **-1.5** | Deadline failure penalty |
| Invalid action (wrong zone, busy worker, etc.) | **-0.5** | Bad decision penalty |
| Replenish when stock is low | **+0.3** | Proactive stock management |
| Replenish when stock is fine | **-0.2** | Wasteful action penalty |
| Skip when orders pending | **-0.1** | Inaction penalty |
| Skip when nothing pending | **0.0** | No penalty |

## Grading (Deterministic, 0.0 -- 1.0)

Episodes are scored with a weighted formula:

```
score = 0.40 * fulfillment_rate      (fulfilled / total orders)
      + 0.25 * urgency_score         (fulfilled urgent / total urgent)
      + 0.20 * efficiency_score      (1.0 - unnecessary_actions / max_steps)
      + 0.15 * validity_score        (1.0 - invalid_actions / total_actions)
```

| Label | Score Range |
|:------|:----------:|
| PERFECT | 0.95 -- 1.00 |
| GOOD | 0.70 -- 0.94 |
| PARTIAL | 0.40 -- 0.69 |
| POOR | 0.10 -- 0.39 |
| FAIL | 0.00 -- 0.09 |

## Sample Evaluation Output

```
============================================================
  WAREHOUSE DISPATCH ENVIRONMENT - EVALUATION
============================================================

  Task: EASY
  ──────────────────────────────────────────────────────
  Score:      1.0000  (PERFECT)
  Fulfilled:  5/5
  Expired:    0
  Invalid:    0
  Steps:      5/15
  Reward:     +7.50

  Task: MEDIUM
  ──────────────────────────────────────────────────────
  Score:      0.9200  (GOOD)
  Fulfilled:  8/10
  Expired:    2
  Invalid:    0
  Steps:      9/20
  Reward:     +11.30

  Task: HARD
  ──────────────────────────────────────────────────────
  Score:      0.6350  (PARTIAL)
  Fulfilled:  6/15
  Expired:    9
  Invalid:    0
  Steps:      6/25
  Reward:     -4.00

============================================================
  SUMMARY
────────────────────────────────────────────────────────────
  easy      1.0000  PERFECT
  medium    0.9200  GOOD
  hard      0.6350  PARTIAL
────────────────────────────────────────────────────────────
  Average   0.8517  (GOOD)
============================================================
```

## API Usage

### Start the server
```bash
python app.py
# Swagger docs: http://localhost:8000/docs
```

### Health check
```bash
curl http://localhost:8000/health
# {"status": "healthy"}
```

### Reset environment
```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'
```

### Step (flat format -- preferred)
```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "assign", "order_id": "ORD-001", "worker_id": "W-01"}'
```

### Step (nested format -- also supported)
```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "skip"}}'
```

### Get episode state
```bash
curl http://localhost:8000/state
```

### Response structure
```json
{
  "observation": {
    "current_step": 1,
    "max_steps": 15,
    "orders": [...],
    "workers": [...],
    "inventory": [...],
    "pending_count": 4,
    "fulfilled_count": 1,
    "message": "ASSIGNED: Order 'ORD-001' -> Worker 'W-01'. Priority: normal. Reward: +1.5",
    "reward": 1.5,
    "done": false
  },
  "reward": 1.5,
  "done": false,
  "info": {"cumulative_reward": 1.5, "task_id": "easy"}
}
```

## Docker

```bash
# Build
docker build -t warehouse-dispatch-env .

# Run server
docker run -p 8000:8000 warehouse-dispatch-env

# Run evaluation inside container
docker run warehouse-dispatch-env python evaluate.py

# Run baseline agent
docker run warehouse-dispatch-env python baseline_agent.py --quiet
```

## Project Structure

```
warehouse_dispatch_env/
├── app.py                 # HTTP server entrypoint (FastAPI + Uvicorn)
├── models.py              # Pydantic models: Order, Worker, Product, Action, Observation, State
├── baseline_agent.py      # Rule-based priority heuristic agent
├── evaluate.py            # Run all tasks and print scores
├── test_env.py            # 8 smoke tests
├── client.py              # Standalone HTTP client
├── openenv.yaml           # OpenEnv manifest
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container image
├── README.md              # This file
└── server/
    ├── warehouse_environment.py  # Core environment: reset(), step(), state
    ├── tasks.py                  # 3 task definitions (easy, medium, hard)
    ├── grader.py                 # Deterministic 4-metric grader (0.0-1.0)
    └── app.py                    # OpenEnv-compatible server (alternative entrypoint)
```

## How to Run Everything

```bash
# 1. Smoke tests (direct environment, no server needed)
python test_env.py

# 2. Baseline agent with verbose output
python baseline_agent.py

# 3. Full evaluation (deterministic scores)
python evaluate.py

# 4. HTTP server
python app.py
```
