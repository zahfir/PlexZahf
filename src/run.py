from app import create_app
import os

from utils.logger import configure_logging

app = create_app()

if __name__ == "__main__":
    configure_logging()
    host = "0.0.0.0"
    port = int(os.getenv("FLASK_PORT", 8000))

    app.run(host=host, port=port)
