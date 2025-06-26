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
from config.movie_config import MovieConfig
from utils.logger import LOGGER_NAME

import logging

logger = logging.getLogger(LOGGER_NAME)


class Main:
    def __init__(self, plex_service: PlexService, home_assistant: HomeAssistant):
        self.plex_service = plex_service
        self.home_assistant = home_assistant
        self.tracker: PlexProgressTracker | None = None
        self.simulator: PlaybackSimulator | None = None
        self.movie: Movie = None
        self.movie_config: MovieConfig = None
        # State variables
        self.current_lighting: Scene = None
        self.is_syncing_lights: bool = False
        self.brightness_pct: int = BRIGHTNESS
        self.running: bool = False

    def _toggle_sync_lights(self) -> bool:
        self.is_syncing_lights = not self.is_syncing_lights
        if self.is_syncing_lights:
            logger.info("is_syncing_lights is toggled ON")
            self.change_to_scene(self.current_lighting)
        else:
            logger.info("is_syncing_lights is toggled OFF")
        return self.is_syncing_lights

    def _set_brightness_pct(self, brightness) -> int:
        self.brightness_pct = brightness
        if self.current_lighting:
            logger.info(
                f"Manual brightness change {self.brightness_pct}% for current scene: {self.current_lighting}"
            )
            self.home_assistant.set_living_room_lights_color(
                color=self.current_lighting.color,
                brightness_pct=self.brightness_pct,
            )
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
        self.change_to_scene(new_scene)

    def process_video_in_background(self):
        """Run frame extraction in background thread"""
        logger.info("Starting video processing in background thread")
        self.movie.analyse()
        logger.info("Video processing completed")

    def change_lights_to_match_scene(self, scene: Scene):
        if self.is_syncing_lights:
            logger.info(f"Changing lights: {scene}")
            self.home_assistant.set_living_room_lights_color(
                color=scene.color,
                brightness_pct=self.brightness_pct,
            )

    def change_to_scene(self, scene):
        """Updates lighting in both class state and room"""
        if scene:
            self.change_lights_to_match_scene(scene)
            self.current_lighting = scene

    def handle_scene_boundary(self, position_ms):
        """Called often. Changes lights preemptively as playback nears the end of a scene."""
        curr = self.current_lighting
        if curr:
            nearing_end_of_scene = curr.end - position_ms < SCENE_BOUNDARY_THRESHOLD_MS
            if nearing_end_of_scene:
                next_scene = self.movie.get_next_scene(curr)
                self.change_to_scene(next_scene)

    def start(self):
        # Video processing
        extraction_thread = Thread(
            target=self.process_video_in_background,
            name="VideoProcessingThread",
            daemon=True,
        )
        extraction_thread.start()

        # Position tracking
        self.tracker = PlexProgressTracker(
            self.plex_service, self.plex_service.username
        )
        self.simulator = PlaybackSimulator(self.tracker)
        self.simulator.add_callback(self.on_position_change)
        self.tracker.start()

        self.is_syncing_lights = True
        self.running: bool = True
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(PLAYBACK_POLL_INTERVAL_SEC)
                if self.simulator and self.running:
                    self.handle_scene_boundary(self.simulator.current_position_ms)
        except KeyboardInterrupt:
            logger.info("Program interrupted by user")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            self.running = False
        finally:
            self.stop()

    def stop(self):
        """
        Stop the main service and clean up resources.
        """
        logger.info("Stopping Main service")
        self.running = False  # Signal the main loop to exit

        if self.tracker:
            self.tracker.stop()
            self.tracker = None

        self.simulator = None
        self.is_syncing_lights = False
        self.current_lighting = None
        logger.info("Main service stopped")

    def run(self, movie: Movie, movie_config: MovieConfig, brightness: int):
        """
        Run from controller.py.

        """
        self.movie = movie
        self.movie_config = movie_config
        self.brightness_pct = brightness

        self.start()
