from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import threading
from main import Main

app = Flask(__name__, static_folder="static")
CORS(app)  # Allow cross-origin requests

# Global state
plexzahf_instance = None
plexzahf_thread = None
status = "stopped"


@app.route("/api/start", methods=["POST"])
def start_service():
    global plexzahf_instance, plexzahf_thread, status

    if status == "running":
        return jsonify({"status": "already running"})

    try:
        plexzahf_instance = Main()
        plexzahf_thread = threading.Thread(target=plexzahf_instance.start)
        plexzahf_thread.daemon = True
        plexzahf_thread.start()
        status = "running"
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({"status": status})


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
