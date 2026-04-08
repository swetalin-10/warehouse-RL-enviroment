#!/usr/bin/env python3
"""
Warehouse Dispatch Environment -- HTTP Server Entrypoint.

Exposes the environment over HTTP with endpoints:
  POST /reset   -- Reset environment (optionally specify task_id and seed)
  POST /step    -- Execute an action (assign, replenish, or skip)
  GET  /state   -- Get current episode state
  GET  /health  -- Health check

Usage:
    python app.py
    python app.py --port 9000
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from models import DispatchAction
from server.warehouse_environment import WarehouseEnvironment
from server.tasks import TASKS


# =============================================================================
# Request / Response Models -- Typed for clean Swagger docs
# =============================================================================

class ResetRequest(BaseModel):
    """Request body for POST /reset."""
    task_id: Literal["easy", "medium", "hard"] = Field(
        default="easy",
        description="Task difficulty level",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional random seed for reproducible episodes (uses task default if omitted)",
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {"task_id": "easy", "seed": None},
            {"task_id": "hard"},
        ]
    }}


class StepRequest(BaseModel):
    """
    Request body for POST /step.

    Supports two formats:
    - Flat: {"action_type": "assign", "order_id": "ORD-001", "worker_id": "W-01"}
    - Nested (legacy): {"action": {"action_type": "assign", "order_id": "ORD-001", "worker_id": "W-01"}}
    """
    action_type: Optional[Literal["assign", "replenish", "skip"]] = Field(
        default=None,
        description="Type of action: 'assign', 'replenish', or 'skip'",
    )
    order_id: Optional[str] = Field(
        default=None,
        description="Order to assign (required for 'assign')",
    )
    worker_id: Optional[str] = Field(
        default=None,
        description="Worker to assign to (required for 'assign')",
    )
    product_id: Optional[str] = Field(
        default=None,
        description="Product to replenish (required for 'replenish')",
    )

    # Legacy nested format support
    action: Optional[Dict[str, Any]] = Field(
        default=None,
        description="(Legacy) Nested action dict -- prefer flat fields above",
        json_schema_extra={"deprecated": True},
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {"action_type": "assign", "order_id": "ORD-001", "worker_id": "W-01"},
            {"action_type": "replenish", "product_id": "PROD-A"},
            {"action_type": "skip"},
        ]
    }}


class StepResponse(BaseModel):
    """Response from POST /step."""
    observation: Dict[str, Any] = Field(description="Full warehouse observation after the action")
    reward: float = Field(description="Reward for this step")
    done: bool = Field(description="Whether the episode has ended")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (cumulative reward, task_id, etc.)",
    )


class ResetResponse(BaseModel):
    """Response from POST /reset."""
    observation: Dict[str, Any] = Field(description="Initial warehouse observation")
    reward: float = Field(description="Initial reward (always 0.0)")
    done: bool = Field(description="Whether the episode has ended (always false)")
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class HealthResponse(BaseModel):
    """Response from GET /health."""
    status: str = Field(description="'healthy' if the server is running")


class ErrorResponse(BaseModel):
    """Error response returned on failures."""
    error: str = Field(description="Error type")
    detail: str = Field(description="Human-readable error message")


# =============================================================================
# App & Environment
# =============================================================================

app = FastAPI(
    title="Warehouse Dispatch Environment",
    description=(
        "OpenEnv environment for warehouse order dispatch coordination. "
        "An AI agent acts as a warehouse dispatcher, assigning orders to workers, "
        "managing inventory, and meeting deadlines under resource constraints.\n\n"
        "**Tasks:** easy, medium, hard\n\n"
        "**Actions:** assign, replenish, skip"
    ),
    version="0.1.0",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

env = WarehouseEnvironment()
_has_been_reset = False


# =============================================================================
# Helper: safe observation serialization
# =============================================================================

def _serialize_obs(obs) -> dict:
    """Safely convert an observation to a JSON-serializable dict."""
    try:
        obs_dict = obs.model_dump()
    except Exception:
        obs_dict = {
            "current_step": getattr(obs, "current_step", 0),
            "max_steps": getattr(obs, "max_steps", 0),
            "orders": getattr(obs, "orders", []),
            "workers": getattr(obs, "workers", []),
            "inventory": getattr(obs, "inventory", []),
            "pending_count": getattr(obs, "pending_count", 0),
            "fulfilled_count": getattr(obs, "fulfilled_count", 0),
            "expired_count": getattr(obs, "expired_count", 0),
            "invalid_actions": getattr(obs, "invalid_actions", 0),
            "message": getattr(obs, "message", ""),
            "reward": getattr(obs, "reward", 0.0),
            "done": getattr(obs, "done", False),
            "metadata": getattr(obs, "metadata", {}),
        }
    return obs_dict


def _build_response(obs) -> dict:
    """Build a standardized response dict from an observation."""
    obs_dict = _serialize_obs(obs)
    return {
        "observation": obs_dict,
        "reward": float(obs.reward) if obs.reward is not None else 0.0,
        "done": bool(obs.done),
        "info": obs_dict.get("metadata", {}),
    }


# =============================================================================
# Endpoints
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Returns healthy status if the server is running.",
)
def health():
    return {"status": "healthy"}


@app.post(
    "/reset",
    response_model=ResetResponse,
    tags=["Environment"],
    summary="Reset the environment",
    description=(
        "Reset the warehouse environment for a new episode. "
        "Optionally specify a task difficulty (easy/medium/hard) and a random seed."
    ),
)
def reset(req: ResetRequest):
    global _has_been_reset
    try:
        task_id = req.task_id
        if task_id not in TASKS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task_id '{task_id}'. Must be one of: {list(TASKS.keys())}",
            )

        obs = env.reset(seed=req.seed, task_id=task_id)
        _has_been_reset = True
        return _build_response(obs)

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "reset_failed", "detail": str(e)},
        )


@app.post(
    "/step",
    response_model=StepResponse,
    tags=["Environment"],
    summary="Execute one action step",
    description=(
        "Execute a single action in the warehouse environment.\n\n"
        "**Action types:**\n"
        "- `assign` -- Assign a pending order to a free worker (requires `order_id` and `worker_id`)\n"
        "- `replenish` -- Add 10 units of stock to a product (requires `product_id`)\n"
        "- `skip` -- Do nothing this step\n\n"
        "**Supports two request formats:**\n"
        "- Flat (preferred): `{\"action_type\": \"assign\", \"order_id\": \"ORD-001\", \"worker_id\": \"W-01\"}`\n"
        "- Nested (legacy): `{\"action\": {\"action_type\": \"assign\", \"order_id\": \"ORD-001\", \"worker_id\": \"W-01\"}}`"
    ),
)
def step(req: StepRequest):
    global _has_been_reset

    if not _has_been_reset:
        return JSONResponse(
            status_code=400,
            content={
                "error": "not_initialized",
                "detail": "Environment has not been reset. Call POST /reset first.",
            },
        )

    try:
        # Resolve action fields: prefer flat format, fall back to nested
        if req.action_type is not None:
            action_data = {"action_type": req.action_type}
            if req.order_id is not None:
                action_data["order_id"] = req.order_id
            if req.worker_id is not None:
                action_data["worker_id"] = req.worker_id
            if req.product_id is not None:
                action_data["product_id"] = req.product_id
        elif req.action is not None:
            action_data = dict(req.action)
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "missing_action",
                    "detail": (
                        "No action specified. Provide 'action_type' (flat format) "
                        "or 'action' dict (nested format). "
                        "Valid action_types: assign, replenish, skip."
                    ),
                },
            )

        # Validate action_type is present
        if "action_type" not in action_data:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "missing_action_type",
                    "detail": "The 'action_type' field is required. Must be one of: assign, replenish, skip.",
                },
            )

        # Validate action_type value
        if action_data["action_type"] not in ("assign", "replenish", "skip"):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_action_type",
                    "detail": (
                        f"Invalid action_type '{action_data['action_type']}'. "
                        "Must be one of: assign, replenish, skip."
                    ),
                },
            )

        # Construct the typed action
        try:
            action = DispatchAction(**action_data)
        except ValidationError as ve:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_action",
                    "detail": f"Action validation failed: {ve.errors()}",
                },
            )

        # Execute the step
        obs = env.step(action)
        return _build_response(obs)

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "step_failed",
                "detail": f"Unexpected error during step: {str(e)}",
            },
        )


@app.get(
    "/state",
    tags=["Environment"],
    summary="Get current episode state",
    description="Returns the current internal episode state including task_id, step count, and scoring metrics.",
)
def state():
    global _has_been_reset

    if not _has_been_reset:
        return JSONResponse(
            status_code=400,
            content={
                "error": "not_initialized",
                "detail": "Environment has not been reset. Call POST /reset first.",
            },
        )

    try:
        state_data = env.state.model_dump()
        return state_data
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "state_failed", "detail": str(e)},
        )


# =============================================================================
# Entrypoint
# =============================================================================

def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Warehouse Dispatch Environment Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()

    print("=" * 56)
    print("  Warehouse Dispatch Environment  v0.1.0")
    print("  Tasks: easy, medium, hard")
    print("  Endpoints: /reset /step /state /health")
    print(f"  Docs: http://{args.host}:{args.port}/docs")
    print("=" * 56)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
