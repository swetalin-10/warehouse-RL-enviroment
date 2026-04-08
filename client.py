# Copyright (c) 2024. All rights reserved.
# Warehouse Dispatch Environment — Client.

"""
Warehouse Dispatch Environment Client.

Provides the client for connecting to a Warehouse Dispatch Environment server.

Example (sync):
    >>> with DispatchEnv(base_url="http://localhost:8000").sync() as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     result = env.call_tool("dispatch", action_type="assign",
    ...                            order_id="ORD-001", worker_id="W-01")
"""

try:
    from openenv.core.mcp_client import MCPToolClient

    class DispatchEnv(MCPToolClient):
        """Client for the Warehouse Dispatch Environment."""
        pass  # MCPToolClient provides all needed functionality

except ImportError:
    # Standalone fallback using requests
    import requests
    from typing import Any, Dict, Optional

    class DispatchEnv:
        """Standalone HTTP client for the Warehouse Dispatch Environment."""

        def __init__(self, base_url: str = "http://localhost:8000"):
            self.base_url = base_url.rstrip("/")

        def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> Dict[str, Any]:
            resp = requests.post(
                f"{self.base_url}/reset",
                json={"task_id": task_id, "seed": seed},
            )
            resp.raise_for_status()
            return resp.json()

        def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
            resp = requests.post(
                f"{self.base_url}/step",
                json={"action": action},
            )
            resp.raise_for_status()
            return resp.json()

        def state(self) -> Dict[str, Any]:
            resp = requests.get(f"{self.base_url}/state")
            resp.raise_for_status()
            return resp.json()
