from flask import Flask, jsonify
from flask_cors import CORS

from utils.app_types import InstanceStatus


def create_app(config=None):
    app = Flask(__name__, static_folder="static")
    CORS(app)  # Allow cross-origin requests

    # Configure app
    app.config.from_mapping(CONTROLLER=None, INSTANCE_STATUS=InstanceStatus.STOPPED)

    if config:
        app.config.from_mapping(config)

    # Register error handlers
    register_error_handlers(app)

    # Register blueprints
    from routes.main_routes import main_bp
    from routes.plex_routes import plex_bp
    from routes.lighting_routes import lighting_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(plex_bp)
    app.register_blueprint(lighting_bp)

    return app


def register_error_handlers(app):
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"status": "error", "message": "Resource not found"}), 404

    @app.errorhandler(500)
    def server_error(error):
        return jsonify({"status": "error", "message": "Internal server error"}), 500

    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"status": "error", "message": "Bad request"}), 400
