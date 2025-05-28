import logging
from io import BytesIO
import time
from typing import List, Optional

import numpy as np

from constants import (
    COLORFUL_BIAS,
    FPS,
    HEIGHT,
    HUE_TOLERANCE,
    MAX_GAP_FRAMES,
    MIN_SCENE_FRAMES,
    SCORE_THRESHOLD,
    WIDTH,
)
from lighting_schedule import LightingSchedule
from config.movie_config import MovieConfig
from utils.db.database_service import (
    ConfigNotFoundError,
    DatabaseService,
    MovieNotFoundError,
)
from utils.file_handler import FileHandler
from utils.logger import LOGGER_NAME
from scene import Scene
from frame.frame_analysis import FrameAnalysis

from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(LOGGER_NAME)


class Movie:
    """
    Class representing a movie with scenes and frames.

    Attributes:
        path: Path to the movie file
        scenes: List of Scene objects
    """

    def __init__(self, path: str, movie_config: MovieConfig, name: str = ""):
        """
        Initialize a Movie object.

        Args:
            path: Path to the movie file
        """

        self.path = path
        self.movie_config = movie_config
        self.name = name
        self._id = None  # Database id
        self.frame_colors: List[dict] = []  # Types: dicts + np arrays
        self.scenes: List[Scene] = []

    def load_db_for_existing_data(self):
        """Load any existing data on this movie from the database."""
        with DatabaseService() as db:
            try:
                self._id = db.load_movie(self.name)
                self.frame_colors = db.load_movie_frame_colors(self._id)
                self.scenes = db.load_movie_schedule(
                    self._id,
                    self.movie_config,
                )

            except MovieNotFoundError:
                logger.info(f"Movie {self.name} not in the database. Downloading...")
                return
            except ConfigNotFoundError:
                logger.info(
                    f"Movie config for {self.name} not in the database. Making schedule..."
                )
                return
            except Exception as e:
                logger.error(f"Error loading movie data from DB: {e}")
                return

    def save_file_to_db(self):
        """Save the movie and frame to the database."""
        with DatabaseService() as db:
            self._id = db.save_file(self.name, self.frame_colors)

    def save_scenes_to_db(self):
        """Save the movie and self.scenes to the database."""
        with DatabaseService() as db:
            db.save_schedule(
                self._id,
                self.scenes,
                self.movie_config,
            )

    def _download_movie(self) -> BytesIO:
        """Download the movie file and return it as a BytesIO object."""
        process = FileHandler.create_ffmpeg_process(self.path)
        return FileHandler.read_stream_to_memory(process)

    def download_movie_and_extract_frames(self):
        memfile: BytesIO = self._download_movie()
        if not memfile:
            logger.error("Failed to download movie.")
            return

        start = time.time()
        buffer = memfile.read()

        with ProcessPoolExecutor(max_workers=10) as executor:
            for batch_results in executor.map(
                process_frame_batch,
                chunk_frames(buffer, WIDTH, HEIGHT, chunk_size=100),
            ):
                self.frame_colors.extend(batch_results)

        logger.info(
            f"TIME - concurrent workers read memfile, process frame data, extract colors: {time.time() - start:.2f} seconds"
        )

    def analyse(self):
        """
        Downloads file to memory or fetches existing DB data.
        Analyses frames and constructs scenes.
        This method uses multiprocessing to speed up the analysis of frames.
        Avoids reprocessing data that exists in the database.
        """
        # Check if the movie is already in the database
        self.load_db_for_existing_data()

        if not self.frame_colors:
            self.download_movie_and_extract_frames()
            self.save_file_to_db()

        if not self.scenes:
            self.frame_colors_to_connected_scenes()
            self.save_scenes_to_db()

    def frame_colors_to_connected_scenes(self):
        """
        Find lighting instructions in the frame colors.
        This method is used to find the lighting schedule for the movie.
        """
        self.instructions = Movie.frame_colors_to_instructions(self.frame_colors)
        self.lighting_schedule = Movie.instructions_to_schedule(self.instructions)
        self.scenes = Movie.schedule_to_connected_scenes(self.lighting_schedule)

    @staticmethod
    def frame_colors_to_instructions(frame_colors):
        """
        Find lighting instructions in the frame colors.
        This method is used to find the lighting schedule for the movie.
        """
        instructions = []
        starttime = time.time()
        logger.info(
            "Finding lighting instructions in frame colors, this may take a while..."
        )
        for hue in range(180):
            instructions.extend(
                LightingSchedule.find_lighting_instructions(
                    frame_colors,
                    target_hue=hue,
                    hue_tol=HUE_TOLERANCE,
                    min_duration=MIN_SCENE_FRAMES,
                    max_gap=MAX_GAP_FRAMES,
                    min_avg_score=SCORE_THRESHOLD,
                    colorful_bias=COLORFUL_BIAS,
                )
            )

        logger.info(
            f"TIME - finding lighting instructions for {len(frame_colors)} frames: {time.time() - starttime:.2f} seconds"
        )

        return instructions

    @staticmethod
    def instructions_to_schedule(instructions):
        return LightingSchedule.greedy_schedule_from_heap(instructions)

    @staticmethod
    def schedule_to_connected_scenes(lighting_schedule) -> List[Scene]:
        scenes = Movie.scenes_from_schedule(lighting_schedule)
        connected = Movie.connect_scene_schedule(scenes)
        for i in range(len(connected)):
            connected[i].array_index = i
            logger.debug(f"Created Scene {i}: {connected[i]}")
        return connected

    @staticmethod
    def scenes_from_schedule(lighting_schedule):
        scenes = []
        for light_event in lighting_schedule:
            scenes.append(
                Scene(
                    start=int((light_event["start"] / FPS) * 1000),
                    end=int((light_event["end"] / FPS) * 1000),
                    color=light_event["color"],
                )
            )
        return scenes

    @staticmethod
    def connect_scene_schedule(scenes: List[Scene]) -> List[Scene]:
        """Adds 'dark' scenes in gaps between scenes"""
        n = len(scenes)

        if n == 1:
            return scenes

        connected_scenes = []
        for i in range(n - 1):
            cur, nxt = scenes[i], scenes[i + 1]
            connected_scenes.append(cur)
            connected_scenes.append(Movie._dark_filler_scene(cur, nxt))

        connected_scenes.append(scenes[-1])
        return connected_scenes

    @staticmethod
    def _dark_filler_scene(scene1: Scene, scene2: Scene) -> Scene:
        return Scene(
            start=scene1.end,
            end=scene2.start,
            color=[int(-scene2.color[0]), scene2.color[1], scene2.color[2]],
        )

    def extend_last_scene(self, new_end_position: int):
        self.scenes[-1].end = new_end_position

    def get_scene_at_time(self, time_ms: int) -> Optional[Scene]:
        """Get the scene at a specific time point."""
        for scene in self.scenes:
            if scene.start <= time_ms < scene.end:
                return scene
        return None

    def get_next_scene(self, scene: Scene):
        if scene:
            i = scene.array_index
            if i + 1 < len(self.scenes):
                return self.scenes[i + 1]


def chunk_frames(buffer, width, height, chunk_size=100):
    frame_size = width * height * 3
    total_frames = len(buffer) // frame_size
    for i in range(0, total_frames, chunk_size):
        frames = []
        for j in range(chunk_size):
            idx = i + j
            if idx * frame_size + frame_size > len(buffer):
                break
            raw = buffer[idx * frame_size : (idx + 1) * frame_size]
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            frames.append(frame)
        yield frames  # one batch of chunk_size frames


def process_frame_batch(frames):
    return [FrameAnalysis.get_top_colors(f) for f in frames]
