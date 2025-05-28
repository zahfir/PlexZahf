from flask import Blueprint, jsonify, request
from middleware import require_running_session
from utils.flask_helpers import get_controller

lighting_bp = Blueprint("lighting", __name__, url_prefix="/api/lighting")


@lighting_bp.route("/toggle", methods=["POST"])
@require_running_session
def toggle_lights():
    controller = get_controller()
    is_syncing = controller.main_service._toggle_sync_lights()
    return jsonify({"status": "success", "is_syncing_lights": is_syncing})


@lighting_bp.route("/brightness", methods=["POST"])
@require_running_session
def set_brightness():
    controller = get_controller()
    brightness = request.json.get("brightness")
    if brightness is None:
        return (
            jsonify({"status": "error", "message": "Brightness value is required"}),
            400,
        )

    result_val: int = controller.main_service._set_brightness_pct(brightness)
    return jsonify({"status": "success", "brightness": result_val})
