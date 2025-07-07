from collections import defaultdict
import os
import logging

from dotenv import load_dotenv
from plex.plex_service import PlexService
from main import Main
from movie import Movie
from config.movie_config import MovieConfig
from home_assistant.home_assistant import HomeAssistant
from utils.db.database_service import DatabaseService
from utils.logger import LOGGER_NAME
from threading import Thread

load_dotenv()

logger = logging.getLogger(LOGGER_NAME)

NO_ACTIVE_SESSION_RESPONSE = {
    "status": "error",
    "message": "No active session found",
}
INIT_PLEX_FAILED_RESPONSE = {
    "status": "error",
    "message": "Failed to initialize Plex service",
}
SUCCESS_RESPONSE = {"status": "success"}


class Controller:
    def __init__(self, use_local_server: bool = False):
        """
        Initialize the Controller with a flag to use a local server.

        Args:
            use_local_server: Boolean flag to indicate if a local server should be used
        """
        self.use_local_server = use_local_server
        self.plex = None
        self.stream_url = None
        self.movie_name = None
        self.main_service = None

        self.home_assistant = HomeAssistant(
            os.getenv("HASS_URL"),
            os.getenv("HASS_ACCESS_TOKEN"),
        )

    def _stream_url(self):
        """
        Get the stream URL for the current Plex session.
        Returns:
            The stream URL or an error message if no session is found.
        """
        if not self.plex:
            self.plex = self._plex_login(self.use_local_server)

        self.stream_url = self.plex.get_stream_url(self.plex.session)

        return (
            {"stream_url": self.stream_url}
            if self.stream_url
            else NO_ACTIVE_SESSION_RESPONSE
        )

    @staticmethod
    def _movie_config(movie_config_data: dict) -> MovieConfig:
        """
        Takes frontend requests with either an existing config's ID or a config data dict.

        Args:
            movie_config_data: Frontend request body containing movie configuration data

        Returns:
            MovieConfig object
        """
        if "id" in movie_config_data and movie_config_data["id"]:
            # If the movie_config_data contains an 'id', it is a database entry
            with DatabaseService() as db:
                movie_config_data = db.get_config_by_id(movie_config_data["id"])
                return MovieConfig.from_db_tuple(db_tuple=movie_config_data)
        # Otherwise, it is a new movie config with fps, width, height, etc.
        return MovieConfig.from_dict(data=movie_config_data)

    @staticmethod
    def _plex_login(use_local_server: bool):
        """

        Log into Plex server.

        Args:
            use_local_server: Boolean flag to indicate if a local server should be used

        Returns:
            PlexService object
        """
        if use_local_server:
            # Use local server credentials
            return PlexService(
                username=os.getenv("PLEX_USERNAME"),
                secret=os.getenv("PLEX_PASSWORD"),
                server=os.getenv("PLEX_SERVER"),
            )

        # Use remote server credentials
        return PlexService(
            username=os.getenv("PLEX_USERNAME"),
            secret=os.getenv("PLEX_TOKEN"),
            server=os.getenv("PLEX_SERVER_ADDRESS"),
        )

    def init_plex_service(self) -> dict:
        """
        Starts Plex service and sets class field.
        """
        self.plex = self._plex_login(self.use_local_server)
        if not self.plex:
            return INIT_PLEX_FAILED_RESPONSE
        return SUCCESS_RESPONSE

    def get_now_playing(self) -> dict:
        """
        Get the currently playing movie from Plex.
        Sets movie_name class field.
        Returns:
            dict: {title (str): movie title}"""
        if not self.plex:
            return NO_ACTIVE_SESSION_RESPONSE

        data = self.plex.now_playing()
        if data:
            self.movie_name = data.get("title")
            self.stream_url = self.plex.get_stream_url(self.plex.session)
            return {"status": "success", "data": data}
        return NO_ACTIVE_SESSION_RESPONSE

    def load_db_for_existing_schedules(self) -> dict:
        """
        Load existing schedules from the database for the current movie.
        Returns:
            dict: {config_id (str): scene_count (int)}
        """
        if not self.plex or not self.movie_name:
            return {"status": "error", "message": "No active session or movie name"}

        # Load all existing scheduled scenes from the database for this movie
        with DatabaseService() as db:
            try:
                scenes = db.get_all_scenes_by_movie_name(self.movie_name)

                # Return entire scene tuples
                scene_dict = defaultdict(list)
                for scene in scenes:
                    # scene[1] is the config_id
                    scene_dict[scene[1]].append(
                        {
                            "start": scene[2],
                            "end": scene[3],
                            "color": [scene[4], scene[5], scene[6]],
                            "saturation": scene[7] if len(scene) > 7 else 100,
                        }
                    )
                # Sort by start time
                for scene_list in scene_dict.values():
                    scene_list.sort(key=lambda x: x["start"])

                return {
                    "status": "success",
                    "data": scene_dict,
                }
            except Exception as e:
                logger.error(f"Error loading movie data from DB: {e}")
                return {"status": "error", "message": str(e)}

    def initiate_session(self, movie_config_data: dict, brightness: int):
        """
        Called from the web server endpoint.
        Runs a Main service session which produces lighting scenes that sync with current session.

        Args:
            movie_config_data: Frontend request body with id OR movie configuration fields
            brightness: Brightness percentage for the lights
        """
        if (
            not self.plex
            or not self.plex.session
            or not self.movie_name
            or not self.stream_url
        ):
            return NO_ACTIVE_SESSION_RESPONSE

        movie_config: MovieConfig = Controller._movie_config(movie_config_data)

        movie: Movie = Movie(
            path=self.stream_url,
            name=self.movie_name,
            movie_config=movie_config,
        )

        # Create and start the main service in a separate thread
        self.main_service: Main = Main(self.plex, self.home_assistant)
        Thread(
            target=self.main_service.run,
            args=(movie, movie_config, brightness),
            daemon=True,  # Make thread daemon so it exits when main program exits
        ).start()

        return {
            "status": "success",
            "message": f"Movie {self.movie_name} is now playing.",
        }

    def cleanup(self):
        """
        End the current Plex session and stop the main service.
        """
        if not self.plex:
            return NO_ACTIVE_SESSION_RESPONSE

        if self.main_service:
            self.main_service.stop()
            self.main_service = None
        self.plex = None
        self.stream_url = None
        self.movie_name = None
        self.home_assistant = None

        return {"status": "success", "message": "Session ended successfully."}


if __name__ == "__main__":
    controller = Controller()
    result = controller.init_plex_service()
    print(result)
