from enum import Enum, auto


class InstanceStatus(Enum):
    """Enum representing the possible states of the application."""

    STOPPED = "stopped"
    INITIALIZED = "initialized"
    RUNNING = "running"
