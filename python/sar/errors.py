"""Exception types shared across the SAR-DSL Python components."""

__all__ = ["SARError", "ToolchainError", "CompilationError", "LaunchError"]


class SARError(Exception):
    """Base class for SAR-DSL errors."""


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
    """A compiled kernel was invoked with invalid arguments."""
