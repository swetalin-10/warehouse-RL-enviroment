# Copyright (c) 2024. All rights reserved.
# Warehouse Dispatch Environment -- FastAPI Server Application.

"""
FastAPI application for the Warehouse Dispatch Environment.

This module provides two modes:
1. OpenEnv mode: Uses openenv's create_app when the SDK is available.
2. Standalone mode: Creates a hardened FastAPI app with full error handling.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Direct run:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

    from .warehouse_environment import WarehouseEnvironment

    # Create the app using OpenEnv's create_app helper
    app = create_app(
        WarehouseEnvironment,
        CallToolAction,
        CallToolObservation,
        env_name="warehouse_dispatch_env",
    )
except ImportError:
    # Standalone mode -- hardened FastAPI app
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, ValidationError
    from typing import Any, Dict, Literal, Optional

    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from server.warehouse_environment import WarehouseEnvironment
    from server.tasks import TASKS
    from models import DispatchAction

    app = FastAPI(
        title="Warehouse Dispatch Environment",
        description=(
            "OpenEnv environment for warehouse order dispatch coordination. "
            "An AI agent acts as a warehouse dispatcher, assigning orders to workers, "
            "managing inventory, and meeting deadlines under resource constraints."
        ),
        version="0.1.0",
    )

    _env = WarehouseEnvironment()
    _has_been_reset = False

    def _build_response(obs) -> dict:
        """Build standardized response from observation."""
        obs_dict = obs.model_dump()
        return {
            "observation": obs_dict,
            "reward": float(obs.reward) if obs.reward is not None else 0.0,
            "done": bool(obs.done),
            "info": obs_dict.get("metadata", {}),
        }

    class ResetRequest(BaseModel):
        task_id: Literal["easy", "medium", "hard"] = "easy"
        seed: Optional[int] = None

    class StepRequest(BaseModel):
        action_type: Optional[Literal["assign", "replenish", "skip"]] = None
        order_id: Optional[str] = None
        worker_id: Optional[str] = None
        product_id: Optional[str] = None
        action: Optional[Dict[str, Any]] = None

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.post("/reset")
    async def reset(req: ResetRequest):
        global _has_been_reset
        try:
            obs = _env.reset(seed=req.seed, task_id=req.task_id)
            _has_been_reset = True
            return _build_response(obs)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": "reset_failed", "detail": str(e)})

    @app.post("/step")
    async def step(req: StepRequest):
        global _has_been_reset
        if not _has_been_reset:
            return JSONResponse(status_code=400, content={
                "error": "not_initialized",
                "detail": "Call POST /reset first.",
            })
        try:
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
                return JSONResponse(status_code=400, content={
                    "error": "missing_action",
                    "detail": "Provide 'action_type' or 'action' dict. Valid: assign, replenish, skip.",
                })

            if "action_type" not in action_data or action_data["action_type"] not in ("assign", "replenish", "skip"):
                return JSONResponse(status_code=400, content={
                    "error": "invalid_action_type",
                    "detail": "action_type must be one of: assign, replenish, skip.",
                })

            try:
                action = DispatchAction(**action_data)
            except ValidationError as ve:
                return JSONResponse(status_code=400, content={
                    "error": "invalid_action", "detail": str(ve.errors()),
                })

            obs = _env.step(action)
            return _build_response(obs)
        except Exception as e:
            return JSONResponse(status_code=500, content={
                "error": "step_failed", "detail": str(e),
            })

    @app.get("/state")
    async def state():
        global _has_been_reset
        if not _has_been_reset:
            return JSONResponse(status_code=400, content={
                "error": "not_initialized",
                "detail": "Call POST /reset first.",
            })
        try:
            return _env.state.model_dump()
        except Exception as e:
            return JSONResponse(status_code=500, content={
                "error": "state_failed", "detail": str(e),
            })


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
