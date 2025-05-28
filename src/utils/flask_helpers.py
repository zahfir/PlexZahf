from flask import current_app
from typing import Optional, TYPE_CHECKING
from utils.app_types import InstanceStatus

if TYPE_CHECKING:
    # These imports are only used for type checking, not at runtime
    from controller import Controller


def get_controller() -> Optional["Controller"]:
    """Get the controller from the current app configuration."""
    return current_app.config.get("CONTROLLER", None)


def get_instance_status() -> InstanceStatus:
    """Get the current instance status with proper typing."""
    return current_app.config["INSTANCE_STATUS"]


def set_instance_status(status: InstanceStatus) -> None:
    """Set the instance status."""
    current_app.config["INSTANCE_STATUS"] = status
