"""Exception types shared across the SAR-DSL Python components."""

import re

__all__ = [
    "SARError", "TraceError", "ToolchainError", "CompilationError",
    "LaunchError"
]


class SARError(Exception):
    """Base class for SAR-DSL errors."""


class TraceError(SARError, TypeError):
    """Raised when a kernel is traced with inconsistent types or shapes.

    Also a `TypeError`: a mistyped tensor expression is a type error."""


class ToolchainError(SARError):
    """A required external tool or library could not be located."""


class CompilationError(SARError):
    """An external tool failed while compiling a kernel."""

    def __init__(self, stage: str, command, stderr: str):
        self.stage = stage
        self.command = list(command)
        self.stderr = stderr
        self.location = self._location(stderr)
        self.diagnostic = self._diagnostic(stderr)
        pretty = " ".join(str(c) for c in self.command)
        super().__init__(f"stage '{stage}' failed\n  command: {pretty}\n"
                         f"---- tool output ----\n{stderr.strip()}")

    @staticmethod
    def _location(output):
        match = re.search(r'(?m)^([^:\n]+):(\d+):(\d+):\s+(?:error|warning):',
                          output or "")
        if not match:
            return None
        return {
            "file": match.group(1),
            "line": int(match.group(2)),
            "column": int(match.group(3)),
        }

    @staticmethod
    def _diagnostic(output):
        match = re.search(r'(?m)^(?:[^:\n]+:\d+:\d+:\s+)?error:\s*(.+)$',
                          output or "")
        return match.group(1).strip() if match else None

    def as_dict(self) -> dict:
        return {
            "kind": type(self).__name__,
            "stage": self.stage,
            "command": list(self.command),
            "diagnostic": self.diagnostic,
            "location": self.location,
            "output": self.stderr,
        }


class LaunchError(SARError):
    """A compiled artifact was used in a way it does not support: a
    kernel invoked with invalid arguments, a design that cannot be
    executed directly, or a testbench request the interface cannot
    serve."""
