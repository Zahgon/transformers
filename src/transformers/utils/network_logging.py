
from __future__ import annotations

import inspect
import json
import os
import threading
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any

import httpx

from .generic import strtobool


class _NetworkRequestTrace:
    def __init__(self, request: httpx.Request):
        self.request = request
        self.started_at = time.perf_counter()
        self.phase_started_at = {}
        self.phases_ms = defaultdict(float)

    def trace(self, name: str, info: dict[str, Any]) -> None:
        parts = name.rsplit(".", 2)
        if len(parts) != 3:
            return

        _, phase, state = parts
        now = time.perf_counter()
        if state == "started":
            self.phase_started_at[phase] = now
        elif state in {"complete", "failed"}:
            phase_started_at = self.phase_started_at.pop(phase, None)
            if phase_started_at is not None:
                self.phases_ms[phase] += (now - phase_started_at) * 1000

    def build_record(
        self,
        *,
        response: httpx.Response | None = None,
        error: BaseException | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        total_ms = (time.perf_counter() - self.started_at) * 1000
        url = self.request.url
        host = url.host or ""
        port = url.port
        default_port = {"http": 80, "https": 443}.get(url.scheme)
        host_display = host if port in (None, default_port) else f"{host}:{port}"

        http_version = None
        status_code = None
        bytes_downloaded = None
        response_complete = False
        if response is not None:
            status_code = response.status_code
            response_complete = response.is_closed
            raw_http_version = response.extensions.get("http_version")
            if isinstance(raw_http_version, bytes):
                http_version = raw_http_version.decode("ascii", errors="replace")
            elif raw_http_version is not None:
                http_version = str(raw_http_version)

            if response_complete:
                try:
                    bytes_downloaded = len(response.content)
                except httpx.ResponseNotRead:
                    pass

        return {
            "method": self.request.method,
            "scheme": url.scheme,
            "host": host,
            "host_display": host_display,
            "port": port,
            "path": url.path,
            "has_query": bool(url.query),
            "url": f"{url.scheme}://{host_display}{url.path}{'?...' if url.query else ''}",
            "request_id": self.request.headers.get("x-amzn-trace-id") or self.request.headers.get("x-request-id"),
            "status_code": status_code,
            "http_version": http_version,
            "bytes_downloaded": bytes_downloaded,
            "total_ms": total_ms,
            "stream": stream,
            "response_complete": response_complete,
            "phases_ms": dict(sorted(self.phases_ms.items())),
            "error": None if error is None else f"{type(error).__name__}: {error}",
        }


class _NetworkDebugProfiler:
    def __init__(self):
        self._records = []
        self._lock = threading.Lock()
        self._enabled = False
        self._output_path = None
        self._original_client_send = None
        self._original_async_client_send = None
        self._shared_dir = None

    @property
    def enabled(self) -> bool:
        pass

    def clear(self) -> None:
        with self._lock:
            self._records = []

    def enable(self, output_path: str | os.PathLike | None = None) -> None:
        if self._enabled:
            self._output_path = None if output_path is None else os.fspath(output_path)
            self.clear()
            return

        self._output_path = None if output_path is None else os.fspath(output_path)
        self.clear()

        profiler = self
        self._original_client_send = httpx.Client.send
        self._original_async_client_send = httpx.AsyncClient.send

        @wraps(self._original_client_send)
        def patched_client_send(client, request, *args, **kwargs):
            pass

        @wraps(self._original_async_client_send)
        async def patched_async_client_send(client, request, *args, **kwargs):
            pass

        httpx.Client.send = patched_client_send
        httpx.AsyncClient.send = patched_async_client_send
        self._enabled = True

    def setup_shared_dir(self) -> str | None:
        pass

    def set_shared_dir(self, shared_dir: str) -> None:
        pass

    def dump_worker_records(self, worker_id: str | None = None) -> None:
        pass

    def load_worker_records(self) -> None:
        pass

    def cleanup_shared_dir(self) -> None:
        pass

    def disable(self) -> None:
        if not self._enabled:
            return

        httpx.Client.send = self._original_client_send
        httpx.AsyncClient.send = self._original_async_client_send
        self._enabled = False
        self._original_client_send = None
        self._original_async_client_send = None
        self._output_path = None
        self.clear()

    def _append_record(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._records.append(record)

    def _wrap_trace_callback(self, request: httpx.Request, trace: _NetworkRequestTrace):
        existing_trace = request.extensions.get("trace")

        def wrapped_trace(name: str, info: dict[str, Any]) -> Any:
            pass

        return wrapped_trace

    async def _awrap_trace_callback(self, request: httpx.Request, trace: _NetworkRequestTrace):
        existing_trace = request.extensions.get("trace")

        async def wrapped_trace(name: str, info: dict[str, Any]) -> Any:
            pass

        return wrapped_trace

    def _send_with_trace(self, original_send, client, request: httpx.Request, *args, **kwargs):
        trace = _NetworkRequestTrace(request)
        request.extensions = dict(request.extensions)
        request.extensions["trace"] = self._wrap_trace_callback(request, trace)

        try:
            response = original_send(client, request, *args, **kwargs)
        except Exception as error:
            self._append_record(trace.build_record(error=error, stream=kwargs.get("stream", False)))
            raise

        self._append_record(trace.build_record(response=response, stream=kwargs.get("stream", False)))
        return response

    async def _async_send_with_trace(self, original_send, client, request: httpx.Request, *args, **kwargs):
        trace = _NetworkRequestTrace(request)
        request.extensions = dict(request.extensions)
        request.extensions["trace"] = await self._awrap_trace_callback(request, trace)

        try:
            response = await original_send(client, request, *args, **kwargs)
        except Exception as error:
            self._append_record(trace.build_record(error=error, stream=kwargs.get("stream", False)))
            raise

        self._append_record(trace.build_record(response=response, stream=kwargs.get("stream", False)))
        return response

    def build_report(self) -> dict[str, Any]:
        with self._lock:
            records = [
                {
                    **record,
                    "phases_ms": dict(record["phases_ms"]),
                }
                for record in self._records
            ]

        phase_totals_ms = defaultdict(float)
        route_totals = {}
        for record in records:
            for phase, duration_ms in record["phases_ms"].items():
                phase_totals_ms[phase] += duration_ms

            route_key = (record["method"], record["host_display"], record["path"])
            route_total = route_totals.setdefault(
                route_key,
                {
                    "method": record["method"],
                    "host_display": record["host_display"],
                    "path": record["path"],
                    "count": 0,
                    "failures": 0,
                    "total_ms": 0.0,
                    "phase_totals_ms": defaultdict(float),
                },
            )
            route_total["count"] += 1
            route_total["total_ms"] += record["total_ms"]
            route_total["failures"] += int(record["error"] is not None)
            for phase, duration_ms in record["phases_ms"].items():
                route_total["phase_totals_ms"][phase] += duration_ms

        routes = []
        for route_total in route_totals.values():
            route_total["avg_ms"] = route_total["total_ms"] / route_total["count"]
            route_total["phase_totals_ms"] = dict(sorted(route_total["phase_totals_ms"].items()))
            routes.append(route_total)

        routes.sort(key=lambda route: route["total_ms"], reverse=True)
        total_time_ms = sum(record["total_ms"] for record in records)
        return {
            "enabled": self._enabled,
            "output_path": self._output_path,
            "total_requests": len(records),
            "failed_requests": sum(int(record["error"] is not None) for record in records),
            "total_time_ms": total_time_ms,
            "phase_totals_ms": dict(sorted(phase_totals_ms.items())),
            "requests": records,
            "routes": routes,
        }

    def maybe_write_report(self) -> str | None:
        pass


_NETWORK_DEBUG_PROFILER = _NetworkDebugProfiler()


_DEFAULT_REPORT_PATH = "network_debug_report.json"


def _parse_network_debug_env() -> tuple[bool, str]:
    pass


def _enable_network_debug_report(output_path: str | os.PathLike | None = None) -> None:
    _NETWORK_DEBUG_PROFILER.enable(output_path=output_path)


def _disable_network_debug_report() -> None:
    _NETWORK_DEBUG_PROFILER.disable()


def _clear_network_debug_report() -> None:
    _NETWORK_DEBUG_PROFILER.clear()


def _get_network_debug_report() -> dict[str, Any]:
    return _NETWORK_DEBUG_PROFILER.build_report()


def _enable_network_debug_report_from_env() -> bool:
    pass


def _format_network_debug_report(max_requests: int = 20, max_routes: int = 10) -> str:
    report = _get_network_debug_report()
    if report["total_requests"] == 0:
        return "Network debug report: no httpx requests captured."

    lines = [
        "Network debug report",
        f"Requests captured: {report['total_requests']}",
        f"Failed requests: {report['failed_requests']}",
        f"Cumulative request time: {report['total_time_ms']:.1f} ms",
    ]

    if report["phase_totals_ms"]:
        phase_summary = ", ".join(
            f"{phase}={duration_ms:.1f} ms"
            for phase, duration_ms in sorted(report["phase_totals_ms"].items(), key=lambda item: item[1], reverse=True)
        )
        lines.append(f"Phase totals: {phase_summary}")

    lines.append("")
    lines.append("Slowest requests:")
    for idx, record in enumerate(
        sorted(report["requests"], key=lambda request: request["total_ms"], reverse=True)[:max_requests],
        start=1,
    ):
        status = record["error"] or f"status={record['status_code']}"
        phase_bits = []
        for phase in ("connect_tcp", "start_tls", "receive_response_headers", "receive_response_body"):
            duration_ms = record["phases_ms"].get(phase)
            if duration_ms is not None:
                phase_bits.append(f"{phase}={duration_ms:.1f} ms")
        phase_suffix = f" ({', '.join(phase_bits)})" if phase_bits else ""
        incomplete_suffix = " incomplete" if record["stream"] and not record["response_complete"] else ""
        lines.append(
            f"{idx:>2}. {record['method']} {record['url']} {record['total_ms']:.1f} ms {status}{incomplete_suffix}{phase_suffix}"
        )

    lines.append("")
    lines.append("Slowest routes:")
    for idx, route in enumerate(report["routes"][:max_routes], start=1):
        lines.append(
            f"{idx:>2}. {route['method']} {route['host_display']}{route['path']} count={route['count']} "
            f"total={route['total_ms']:.1f} ms avg={route['avg_ms']:.1f} ms failures={route['failures']}"
        )

    return "\n".join(lines)


class NetworkDebugPlugin:

    def pytest_configure(self, config):
        pass

    def pytest_configure_node(self, node):
        pass

    def pytest_sessionfinish(self, session, exitstatus):
        pass

    def pytest_terminal_summary(self, terminalreporter):
        pass


def register_network_debug_plugin(config) -> None:
    pass


__all__ = [
    "register_network_debug_plugin",
]
