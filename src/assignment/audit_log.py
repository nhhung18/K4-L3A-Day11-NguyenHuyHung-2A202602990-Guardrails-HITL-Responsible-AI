"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}
        self._pending_by_user: dict[str, deque[str]] = defaultdict(deque)

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Start an auditable request and return its correlation identifier.

        The input text, user, UTC start time, and monotonic timer are retained
        until ``record_output`` pairs the final decision with this request.
        A generated ID keeps overlapping requests from the same user distinct.
        """
        request_id = request_id or uuid.uuid4().hex
        if request_id in self._open:
            raise ValueError(f"request_id is already active: {request_id}")
        self._open[request_id] = {
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "started_monotonic": time.perf_counter(),
        }
        self._pending_by_user[user_id].append(request_id)
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete a request audit record with response, decision, and latency.

        When ``request_id`` is omitted, the oldest outstanding request for
        ``user_id`` is completed. This supports straightforward sequential
        callers while explicit IDs safely correlate concurrent requests.
        """
        pending = self._pending_by_user.get(user_id, deque())
        if request_id is None and pending:
            request_id = pending.popleft()
        elif request_id is not None and request_id in pending:
            pending.remove(request_id)

        started = self._open.pop(request_id, None) if request_id else None
        now = time.perf_counter()
        latency_ms = (
            max(0.0, now - started["started_monotonic"]) * 1000 if started else 0.0
        )
        record = {
            "request_id": request_id,
            "user_id": user_id,
            "input": started["input"] if started else None,
            "started_at": started["started_at"] if started else None,
            "completed_at": utc_now_iso(),
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round(latency_ms, 3),
        }
        self.logs.append(record)
        return record

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        # TODO: path = filepath or default_audit_log_path()
        #       ensure parent dirs exist, dump self.logs with indent=2
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
