from flask import Blueprint, jsonify, request, current_app
from middleware import require_initialized
from utils.flask_helpers import get_controller, get_instance_status, set_instance_status
from utils.app_types import InstanceStatus

main_bp = Blueprint("main", __name__, url_prefix="/api")


@main_bp.route("/init", methods=["POST"])
def init_service():
    if get_instance_status() != InstanceStatus.STOPPED:
        print("Service is already initialized or running. Stopping current session...")
        stop_service()

    try:
        use_local_server = (
            request.json.get("use_local_server", True) if request.json else True
        )

        # Import here to avoid circular imports
        from controller import Controller

        controller = Controller(use_local_server=use_local_server)
        response = controller.init_plex_service()

        if response.get("status") == "success":
            current_app.config["CONTROLLER"] = controller
            set_instance_status(InstanceStatus.INITIALIZED)

        return jsonify(response)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@main_bp.route("/status", methods=["GET"])
def get_status():
    status = get_instance_status()
    result = {"status": status.value}

    controller = get_controller()
    if status == InstanceStatus.RUNNING and controller and controller.main_service:
        result["is_syncing_lights"] = controller.main_service.is_syncing_lights
        if controller.main_service.current_lighting:
            result["current_lights"] = (
                controller.main_service.current_lighting.to_json()
            )

    return jsonify(result)


@main_bp.route("/start-session", methods=["POST"])
@require_initialized
def start_session():
    if get_instance_status() == InstanceStatus.RUNNING:
        return jsonify({"status": "already running"})

    try:
        controller = get_controller()

        # Get parameters from request
        movie_config_data = request.json.get("config", {}) if request.json else {}
        brightness = request.json.get("brightness", 100) if request.json else 100

        response = controller.initiate_session(movie_config_data, brightness)

        if response.get("status") == "success":
            set_instance_status(InstanceStatus.RUNNING)

        return jsonify(response)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@main_bp.route("/stop", methods=["POST"])
def stop_service():
    status = get_instance_status()
    if status == InstanceStatus.STOPPED:
        return jsonify({"status": "success", "message": "Service is already stopped"})

    try:
        controller = get_controller()

        # Clean up the controller
        if controller:
            controller.cleanup()
            current_app.config.pop("CONTROLLER", None)

        # Reset status to STOPPED
        set_instance_status(InstanceStatus.STOPPED)

        return jsonify({"status": "success", "message": "Service stopped successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
