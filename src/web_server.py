from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import threading
from main import Main

app = Flask(__name__, static_folder="static")
CORS(app)  # Allow cross-origin requests

session_thread_id = "StartSessionThread"

# Global state
plexzahf_instance = None
plexzahf_thread = None
instance_status = "stopped"


@app.route("/api/start", methods=["POST"])
def start_service():
    global plexzahf_instance, plexzahf_thread, instance_status

    if instance_status == "running":
        return jsonify({"status": "already running"})

    try:
        plexzahf_instance = Main()
        plexzahf_thread = threading.Thread(
            target=plexzahf_instance.start, daemon=True, name=session_thread_id
        )
        plexzahf_thread.start()
        instance_status = "running"
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/status", methods=["GET"])
def get_status():
    result = {"status": instance_status}

    if instance_status == "running" and plexzahf_instance:
        result["is_syncing_lights"] = plexzahf_instance.is_syncing_lights
        if plexzahf_instance.current_lighting:
            result["current_lights"] = plexzahf_instance.current_lighting.to_json()

    return jsonify(result)


@app.route("/api/now-playing", methods=["GET"])
def get_now_playing():
    if (
        instance_status == "running"
        and plexzahf_instance
        and plexzahf_instance.plex_service
    ):
        return jsonify(plexzahf_instance.plex_service.now_playing())
    return jsonify({"status": instance_status})


@app.route("/api/toggle-sync-lights", methods=["POST"])
def toggle_sync_lights():
    result = {"status": instance_status}
    if instance_status == "running" and plexzahf_instance:
        result["is_syncing_lights"] = plexzahf_instance._toggle_sync_lights()
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="localhost", port=8000)
