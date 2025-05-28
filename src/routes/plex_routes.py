from flask import Blueprint, jsonify
from middleware import require_initialized
from utils.flask_helpers import get_controller

plex_bp = Blueprint("plex", __name__, url_prefix="/api/plex")


@plex_bp.route("/now-playing", methods=["GET"])
@require_initialized
def get_now_playing():
    controller = get_controller()
    return jsonify(controller.get_now_playing())


@plex_bp.route("/available-configs", methods=["GET"])
@require_initialized
def get_available_configs():
    controller = get_controller()
    return jsonify(controller.load_db_for_existing_schedules())
