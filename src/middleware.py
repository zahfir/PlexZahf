from functools import wraps
from flask import jsonify

from utils.app_types import InstanceStatus
from utils.flask_helpers import get_instance_status


def require_initialized(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if get_instance_status() == InstanceStatus.STOPPED:
            return (
                jsonify({"status": "error", "message": "Service not initialized"}),
                400,
            )
        return f(*args, **kwargs)

    return decorated_function


def require_running_session(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if get_instance_status() != InstanceStatus.RUNNING:
            return jsonify({"status": "error", "message": "No active session"}), 400
        return f(*args, **kwargs)

    return decorated_function
