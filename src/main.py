import os
from dotenv import load_dotenv
from threading import Thread
import time

from home_assistant.home_assistant import HomeAssistant

from plex.plex_service import PlexService
from plex.plex_progress_tracker import PlexProgressTracker
from plex.playback_simulator import PlaybackSimulator

from movie import Movie
from scene import Scene

from constants import (
    BRIGHTNESS,
    PLAYBACK_POLL_INTERVAL_SEC,
    SCENE_BOUNDARY_THRESHOLD_MS,
)
from utils.logger import logger

load_dotenv()


class Main:
    # def drop_all_tables(conn, cursor):
    #     """Drop all tables in the database.

    #     Returns:
    #         bool: True if successful, False otherwise
    #     """
    #     tables = ["FRAME_COLOR", "SCENE", "FRAME", "MOVIE_CONFIG", "MOVIE"]

    #     try:
    #         # Temporarily disable foreign key constraints
    #         cursor.execute("PRAGMA foreign_keys = OFF")

    #         # Drop tables in reverse order to handle dependencies
    #         for table in tables:
    #             cursor.execute(f"DROP TABLE IF EXISTS {table}")
    #             print(f"Table '{table}' dropped")

    #         # Re-enable foreign key constraints
    #         cursor.execute("PRAGMA foreign_keys = ON")
    #         conn.commit()
    #         print("All tables dropped successfully")
    #         return True
    #     except:
    #         return

    # def get_table_counts(cursor):
    #     """Get the number of rows in each table.

    #     Returns:
    #         dict: Dictionary with table names as keys and row counts as values.
    #     """
    #     tables = ["MOVIE", "FRAME", "SCENE", "MOVIE_CONFIG", "FRAME_COLOR"]
    #     counts = {}

    #     try:
    #         for table in tables:
    #             cursor.execute(f"SELECT COUNT(*) FROM {table}")
    #             count = cursor.fetchone()[0]
    #             counts[table] = count

    #         return counts
    #     except:
    #         return

    # db = DatabaseService()
    # db._create_tables()

    # if os.path.exists("birdman_frame_colors.pkl"):
    #     with open("birdman_frame_colors.pkl", "rb") as f:
    #         import pickle

    #         frame_colors = pickle.load(f)

    # if os.path.exists("birdman_schedule.pkl"):
    #     with open("birdman_schedule.pkl", "rb") as f:
    #         import pickle

    #         lighting_schedule = pickle.load(f)

    # # movie_id = db.save_file("birdman", frame_colors)

    # movie_id = db.load_movie("birdman")
    # recon = db.load_movie_frame_colors(movie_id)
    # pickle_scenes = Movie("").schedule_to_connected_scenes(lighting_schedule)
    # config_id = db.save_schedule(movie_id, pickle_scenes)

    # Plex
    PLEX_USERNAME = os.getenv("PLEX_USERNAME")
    PLEX_PASSWORD = os.getenv("PLEX_PASSWORD")
    PLEX_SERVER = os.getenv("PLEX_SERVER")
    PLEX_TOKEN = os.getenv("PLEX_TOKEN")
    PLEX_SERVER_ADDRESS = os.getenv("PLEX_SERVER_ADDRESS")

    # Home Assistant configuration
    HASS_URL = os.getenv("HASS_URL")
    ACCESS_TOKEN = os.getenv("HASS_ACCESS_TOKEN")

    def __init__(self):
        self.plex_service = PlexService(
            username=Main.PLEX_USERNAME,
            secret=Main.PLEX_PASSWORD,
            server=Main.PLEX_SERVER,
        )

        self.home_assistant = HomeAssistant(Main.HASS_URL, Main.ACCESS_TOKEN)

        self.tracker: PlexProgressTracker | None = None

        # State variable - Scene used for current lighting
        self.current_lighting: Scene = None

        self.is_syncing_lights: bool = False

        self.brightness_pct: int = BRIGHTNESS

    def _toggle_sync_lights(self) -> bool:
        self.is_syncing_lights = not self.is_syncing_lights
        return self.is_syncing_lights

    def _set_brightness_pct(self, brightness) -> int:
        self.brightness_pct = brightness
        return self.brightness_pct

    def on_position_change(self, position_ms):
        """
        Called every 10 seconds or when user seeks. Updates 'Scene' class attributes.
        Driven by Plex API's 'viewOffset' playback session attribute.
        """
        # Case 1: Playback uninterrupted
        lighting = self.current_lighting
        if lighting:
            start = lighting.start - SCENE_BOUNDARY_THRESHOLD_MS
            playback_was_steady = lighting and start <= position_ms < lighting.end
            if playback_was_steady:
                logger.info(f"Playback {position_ms} is on track.")
                logger.debug(f"{self.current_lighting}")
                return

        # Case 2: Either first scene or user seeked to new position
        logger.warning(f"JUMP/SEEK to {position_ms}")
        self.current_lighting = None
        new_scene = self.movie.get_scene_at_time(position_ms)
        # TODO CHANGE TO new_scene.next if position is nearing_end of new_scene
        self.change_to_scene(new_scene)

    def extract_frames_in_background(self):
        """Run frame extraction in background thread"""
        logger.info("Starting frame extraction in background thread")
        self.movie.analyse()
        logger.info("Frame extraction completed")

    def change_lights_to_match_scene(self, scene: Scene):
        if self.is_syncing_lights:
            logger.info(f"Changing lights: {scene}")
            self.home_assistant.set_living_room_lights_color(
                rgb_color=scene.color,
                brightness_pct=self.brightness_pct,
            )

    def change_to_scene(self, scene):
        """Updates lighting in both class state and room"""
        if scene:
            self.change_lights_to_match_scene(scene)
            self.current_lighting = scene

    def handle_scene_boundary(self, position_ms):
        """Called often. Changes lights preemptively as playback nears the end of a scene."""
        logger.debug(f"{position_ms}ms")
        curr = self.current_lighting
        if curr:
            nearing_end_of_scene = curr.end - position_ms < SCENE_BOUNDARY_THRESHOLD_MS
            if nearing_end_of_scene:
                next_scene = self.movie.get_next_scene(curr)
                self.change_to_scene(next_scene)

    def start(self):
        # Get the stream URL
        stream_url = self.plex_service.get_stream_url_by_username(Main.PLEX_USERNAME)
        if not stream_url:
            logger.error("No active session found for this user")
            return

        logger.info(f"Found stream URL: {stream_url}")

        # Create a movie from a file
        self.movie = Movie(stream_url, self.plex_service.session._prettyfilename())

        # Start frame extraction in a separate thread
        extraction_thread = Thread(
            target=self.extract_frames_in_background,
            name="FrameExtractionThread",
            daemon=True,
        )
        extraction_thread.start()

        # Continue with tracker setup without waiting for extraction to complete
        self.tracker = PlexProgressTracker(self.plex_service, Main.PLEX_USERNAME)
        self.simulator = PlaybackSimulator(self.tracker)
        self.simulator.add_callback(self.on_position_change)

        # Start the tracker
        self.tracker.start()

        # Start syncing lights
        self.is_syncing_lights = True

        try:
            # Keep main thread alive
            while True:
                time.sleep(PLAYBACK_POLL_INTERVAL_SEC)
                self.handle_scene_boundary(self.simulator.current_position_ms)
        except KeyboardInterrupt:
            logger.info("Program interrupted by user")
        finally:
            # Clean up
            if self.tracker:
                self.tracker.stop()


# Run the function with a video stream
if __name__ == "__main__":
    Main().start()
