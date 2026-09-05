"""State and reporting helpers for the HLS resource retry ladder."""

from __future__ import annotations

import json


class RetryTrace:

    def __init__(self):
        self.items = []

    def record(self,
               reason: str,
               action: str,
               before=None,
               after=None) -> None:
        item = {"reason": reason, "action": action}
        if before is not None:
            item["before"] = before
        if after is not None:
            item["after"] = after
        self.items.append(item)

    def replay(self, values: dict) -> None:
        self.items[:] = list(values.get("retry_trace", ()))


def decisions(config, performance_plan, retry_trace) -> dict:
    return {
        "external_buffer_threshold": int(config.external_buffer_threshold),
        "array_partition_max_factor": int(config.array_partition_max_factor),
        "fft_io_unroll": int(config.fft_io_unroll),
        "fft_parallel_rows": int(config.fft_parallel_rows),
        "performance_plan": performance_plan.to_dict(),
        "retry_trace": list(retry_trace),
    }


def load(text: str) -> dict:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("HLS decisions must be a JSON object")
    for key in ("external_buffer_threshold", "array_partition_max_factor",
                "fft_io_unroll", "fft_parallel_rows"):
        if not isinstance(value.get(key), int):
            raise ValueError(f"HLS decisions have no integer {key!r}")
    trace = value.get("retry_trace")
    if not isinstance(trace, list) or any(not isinstance(item, dict)
                                          for item in trace):
        raise ValueError("HLS decisions have an invalid retry trace")
    plan = value.get("performance_plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("values"), dict):
        raise ValueError("HLS decisions have no performance plan")
    for key in ("clock_ns", "timing_budget_ns", "on_chip_bytes",
                "memory_accesses", "operation_count"):
        if not isinstance(plan.get(key), (int, float)):
            raise ValueError(f"HLS performance plan has no numeric {key!r}")
    return value
