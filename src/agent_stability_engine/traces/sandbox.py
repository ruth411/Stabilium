from __future__ import annotations

import json
import random
import time
from urllib import error, request

from agent_stability_engine.traces.schema import AgentTask, ToolCall


class SandboxExecutor:
    """Execute a tool call.

    Priority order:
    1. Fault injection — if fault_rate > 0 and RNG triggers it
    2. Real HTTP endpoint — if task.tool_endpoints has a URL for this tool
    3. Sandbox response — if task.sandbox_responses has an entry
    4. Error — neither configured
    """

    def __init__(
        self,
        task: AgentTask,
        fault_rate: float = 0.0,
        rng: random.Random | None = None,
    ) -> None:
        if not (0.0 <= fault_rate <= 1.0):
            msg = "fault_rate must be in [0.0, 1.0]"
            raise ValueError(msg)
        self._task = task
        self._fault_rate = fault_rate
        self._rng = rng or random.Random()

    def execute(self, tool_name: str, arguments: dict[str, object], call_id: str) -> ToolCall:
        start = time.monotonic()

        # 1. Fault injection
        if self._fault_rate > 0 and self._rng.random() < self._fault_rate:
            return ToolCall(
                tool_call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
                result=None,
                is_fault_injected=True,
                error=f"Simulated tool failure for '{tool_name}'",
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        # 2. Real HTTP endpoint
        endpoint = self._task.tool_endpoints.get(tool_name)
        if endpoint:
            try:
                result = self._call_endpoint(endpoint, tool_name, arguments)
                return ToolCall(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                    is_fault_injected=False,
                    error=None,
                    duration_ms=int((time.monotonic() - start) * 1000),
                )
            except Exception:
                # Fall through to sandbox
                pass

        # 3. Sandbox response
        sandbox = self._task.sandbox_responses.get(tool_name)
        if sandbox is not None:
            return ToolCall(
                tool_call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
                result=sandbox,
                is_fault_injected=False,
                error=None,
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        # 4. Neither configured
        return ToolCall(
            tool_call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            result=None,
            is_fault_injected=False,
            error=f"No sandbox_response or tool_endpoint configured for '{tool_name}'",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    def _call_endpoint(self, url: str, tool_name: str, arguments: dict[str, object]) -> str:
        """HTTP POST to a real tool endpoint. Timeout: 30s. Returns response body as str."""
        data = json.dumps({"tool": tool_name, "arguments": arguments}).encode("utf-8")
        req = request.Request(
            url,
            method="POST",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=30) as resp:
                body = resp.read()
                if isinstance(body, bytes):
                    return body.decode("utf-8")
                if isinstance(body, str):
                    return body
                msg = "Tool endpoint returned a non-text response body"
                raise RuntimeError(msg)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            msg = f"Tool endpoint HTTP error {exc.code}: {body}"
            raise RuntimeError(msg) from exc
        except error.URLError as exc:
            msg = f"Tool endpoint request error: {exc.reason}"
            raise RuntimeError(msg) from exc
