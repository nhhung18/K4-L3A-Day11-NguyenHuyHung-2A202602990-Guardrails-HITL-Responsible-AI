from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from assignment import audit_log, monitoring, pipeline, rate_limiter


def test_rate_limiter_uses_per_user_sliding_window(monkeypatch):
    current_time = [100.0]
    monkeypatch.setattr(rate_limiter.time, "time", lambda: current_time[0])
    plugin = rate_limiter.RateLimitPlugin(max_requests=2, window_seconds=10)

    async def send(user_id: str):
        return await plugin.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id=user_id), user_message=None
        )

    assert asyncio.run(send("alice")) is None
    current_time[0] += 1
    assert asyncio.run(send("alice")) is None
    assert asyncio.run(send("bob")) is None

    blocked = asyncio.run(send("alice"))
    assert blocked is not None
    assert "Try again in 9s" in blocked.parts[0].text
    assert plugin.blocked_count == 1

    current_time[0] = 111.0
    assert asyncio.run(send("alice")) is None


def test_audit_log_correlates_input_and_output_and_exports(tmp_path):
    logger = audit_log.AuditLogPlugin()
    request_id = logger.record_input(
        user_id="alice", text="What is my account balance?"
    )
    record = logger.record_output(
        user_id="alice",
        text="Your balance is protected.",
        blocked=False,
        layer=None,
        request_id=request_id,
    )
    output_path = tmp_path / "nested" / "audit.json"
    logger.export_json(str(output_path))

    assert record["input"] == "What is my account balance?"
    assert record["output"] == "Your balance is protected."
    assert record["latency_ms"] >= 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == [record]


def test_monitoring_calculates_threshold_alerts_and_exports(tmp_path):
    monitor = monitoring.MonitoringAlert(
        block_rate_threshold=0.4,
        rate_limit_hit_threshold=1,
        judge_fail_rate_threshold=0.25,
    )
    monitor.total_requests = 4
    monitor.blocked_requests = 2
    monitor.rate_limit_hits = 2
    monitor.judge_checks = 4
    monitor.judge_fails = 2

    alerts = monitor.check_metrics()
    output_path = tmp_path / "metrics.json"
    monitor.export_json(str(output_path))
    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert {alert.metric for alert in alerts} == {
        "block_rate", "rate_limit_hits", "judge_fail_rate"
    }
    assert data["block_rate"] == 0.5
    assert data["judge_fail_rate"] == 0.5
    assert len(data["alerts"]) == 3


def test_egress_policy_requires_exact_approved_endpoint_and_safe_payload():
    assert pipeline.is_egress_allowed(
        "https://api.vinbank.example/v1/transfers", "approved transfer amount 500000"
    ) is True
    assert pipeline.is_egress_allowed(
        "https://api.vinbank.example.evil.test/v1/transfers", "approved transfer"
    ) is False
    assert pipeline.is_egress_allowed(
        "https://api.vinbank.example/v1/transfers", "password is secret123"
    ) is False
    assert pipeline.is_egress_allowed(
        "http://api.vinbank.example/v1/transfers", "approved transfer"
    ) is False


def test_build_production_plugins_orders_layers():
    plugins = pipeline.build_production_plugins(max_requests=3, window_seconds=12)

    assert [plugin.name for plugin in plugins] == [
        "rate_limiter", "input_guardrail", "output_guardrail"
    ]
    assert plugins[0].max_requests == 3
    assert plugins[0].window_seconds == 12


def test_assignment_suite_writes_schema_valid_outputs():
    import jsonschema

    from core.config import _ROOT

    audit, monitor = pipeline.build_observability()
    results = asyncio.run(
        pipeline.run_assignment_suite({
            "plugins": pipeline.build_production_plugins(max_requests=3),
            "audit": audit,
            "monitor": monitor,
        })
    )
    schema_path = _ROOT / "schemas" / "results.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=results, schema=schema)

    assert all(not row["blocked"] for row in results["safe_queries"])
    assert sum(row["blocked"] for row in results["attack_queries"]) >= 5
    assert results["rate_limit"]["blocked"] >= 1
    assert results["rate_limit"]["passed"] + results["rate_limit"]["blocked"] == results["rate_limit"]["sent"]
    assert all((_ROOT / "outputs" / name).is_file() for name in (
        "results.json", "audit_log.json", "metrics.json"
    ))