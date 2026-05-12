from __future__ import annotations

import json
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any


DEFAULT_OBSERVABILITY_PATH = Path("outputs/observability/traces.jsonl")


@dataclass
class Span:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_attributes(self, values: dict[str, Any]) -> None:
        self.attributes.update(values)


class ObservabilityRecorder:
    def __init__(self, output_path: Path = DEFAULT_OBSERVABILITY_PATH) -> None:
        self.output_path = output_path

    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        parent_span_id: str | None = None,
        trace_id: str | None = None,
    ) -> "_RecordedSpan":
        return _RecordedSpan(
            recorder=self,
            span=Span(name=name, attributes=dict(attributes or {})),
            parent_span_id=parent_span_id,
            trace_id=trace_id,
        )

    def write(self, record: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")


class _RecordedSpan:
    def __init__(
        self,
        recorder: ObservabilityRecorder,
        span: Span,
        parent_span_id: str | None,
        trace_id: str | None,
    ) -> None:
        self.recorder = recorder
        self.span = span
        self.parent_span_id = parent_span_id
        self.trace_id = trace_id or uuid.uuid4().hex
        self.span_id = uuid.uuid4().hex[:16]
        self.started_at = 0.0
        self.started_monotonic = 0.0

    def __enter__(self) -> Span:
        self.started_at = time.time()
        self.started_monotonic = time.perf_counter()
        self.span.set_attribute("trace_id", self.trace_id)
        self.span.set_attribute("span_id", self.span_id)
        if self.parent_span_id:
            self.span.set_attribute("parent_span_id", self.parent_span_id)
        return self.span

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        ended_at = time.time()
        status = "error" if exc else "ok"
        record: dict[str, Any] = {
            "name": self.span.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "status": status,
            "started_at_unix": self.started_at,
            "ended_at_unix": ended_at,
            "duration_ms": round((time.perf_counter() - self.started_monotonic) * 1000, 3),
            "attributes": self.span.attributes,
        }
        if exc:
            record["error"] = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
            }
        self.recorder.write(record)
        return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
