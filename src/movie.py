import time
from typing import List, Optional

import numpy as np

from scene_generation import SceneGeneration
from constants import (
    FPS,
    HEIGHT,
    HUE_TOLERANCE,
    MAX_GAP_FRAMES,
    MIN_SCENE_FRAMES,
    SCENE_BOUNDARY_THRESHOLD_MS,
    SCORE_THRESHOLD,
    WIDTH,
)
from lighting_schedule import LightingSchedule
from utils.file_handler import FileHandler
from utils.logger import logger
from scene import Scene
from frame.frame_analysis import FrameAnalysis

from concurrent.futures import ProcessPoolExecutor


class Movie:
    """
    Class representing a movie with scenes and frames.

    Attributes:
        path: Path to the movie file
        scenes: List of Scene objects
    """

    def __init__(self, path: str):
        """
        Initialize a Movie object.

        Args:
            path: Path to the movie file
        """
        self.path = path
        self.frame_colors: List[dict] = []
        self.scenes: List[Scene] = []

    def analyse(self):
        """Downloads file to memory, then analyses frames and constructs scenes."""
        # import os
        import pickle

        # if os.path.exists("birdman_frame_colors.pkl"):
        #     with open("birdman_frame_colors.pkl", "rb") as f:
        #         self.frame_colors = pickle.load(f)
        # if os.path.exists("birdman_schedule.pkl"):
        #     with open("birdman_schedule.pkl", "rb") as f:
        #         lighting_schedule = pickle.load(f)

        # self.scenes = self.schedule_to_connected_scenes(lighting_schedule)
        # return
        if not self.frame_colors:
            process = FileHandler.create_ffmpeg_process(self.path)
            memory_file = FileHandler.read_stream_to_memory(process)

            start = time.time()
            buffer = memory_file.read()

            with ProcessPoolExecutor(max_workers=10) as executor:
                for batch_results in executor.map(
                    process_frame_batch,
                    chunk_frames(buffer, WIDTH, HEIGHT, chunk_size=100),
                ):
                    self.frame_colors.extend(batch_results)

            print("Frame analysis: ", time.time() - start)

        with open("revenant_colors.pkl", "wb") as f:
            pickle.dump(self.frame_colors, f)

        lighting_schedule = self.frame_colors_to_schedule(self.frame_colors)
        self.scenes = self.schedule_to_connected_scenes(lighting_schedule)

    def frame_colors_to_schedule(self, frame_colors):
        instructions = []
        for hue in range(180):
            instructions.extend(
                LightingSchedule.find_lighting_instructions(
                    frame_colors,
                    target_hue=hue,
                    hue_tol=HUE_TOLERANCE,
                    min_duration=MIN_SCENE_FRAMES,
                    max_gap=MAX_GAP_FRAMES,
                    min_avg_score=SCORE_THRESHOLD,
                )
            )

        return LightingSchedule.greedy_schedule_from_heap(instructions)

    def schedule_to_connected_scenes(self, lighting_schedule) -> List[Scene]:
        scenes = self.scenes_from_schedule(lighting_schedule)
        connected = self.connect_scene_schedule(scenes)
        for i in range(len(connected)):
            connected[i].array_index = i
        return connected

    def scenes_from_schedule(self, lighting_schedule):
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

    def connect_scene_schedule(self, scenes: List[Scene]) -> List[Scene]:
        """Adds 'dark' scenes in large gaps between scenes and connects close scenes"""
        n = len(scenes)

        if n == 1:
            return scenes

        connected_scenes = []
        for i in range(n - 1):
            cur, nxt = scenes[i], scenes[i + 1]
            connected_scenes.append(cur)

            gap_ms = nxt.start - cur.end

            if gap_ms > SCENE_BOUNDARY_THRESHOLD_MS * 2:
                connected_scenes.append(self._dark_filler_scene(cur, nxt))
                continue
            cur.end = nxt.start

        connected_scenes.append(scenes[-1])
        return connected_scenes

    def _dark_filler_scene(self, scene1: Scene, scene2: Scene) -> Scene:
        return Scene(
            start=scene1.end,
            end=scene2.start,
            color=[int(-scene2.color[0]), scene2.color[1], scene2.color[2]],
        )

    def generate_scenes(self):
        generation = SceneGeneration()

        for scene in generation.start(self.path):
            # Assign actual array index
            scene.array_index = len(self.scenes)

            # If possible, merge similar color scenes
            if self.scenes and self.scenes[-1].is_combinable(scene):
                self.extend_last_scene(scene.end)
                logger.debug(f"Extended last scene with {scene}")
                continue

            self.scenes.append(scene)
            logger.debug(f"Created scene - {scene}")

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
