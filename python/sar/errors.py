"""Exception types shared across the SAR-DSL Python components."""

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
        pretty = " ".join(str(c) for c in self.command)
        super().__init__(f"stage '{stage}' failed\n  command: {pretty}\n"
                         f"---- tool output ----\n{stderr.strip()}")


class LaunchError(SARError):
    """A compiled artifact was used in a way it does not support: a
    kernel invoked with invalid arguments, a design that cannot be
    executed directly, or a testbench request the interface cannot
    serve."""
