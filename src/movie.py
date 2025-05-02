from typing import List, Optional

from scene_generation import SceneGeneration
from utils.logger import logger
from scene import Scene


class Movie:
    """
    Class representing a movie with scenes and frames.

    Attributes:
        path: Path to the movie file
        scenes: List of Scene objects
        frames: List of Frame objects
        processor: VideoProcessor for extracting frames
    """

    def __init__(self, path: str):
        """
        Initialize a Movie object.

        Args:
            path: Path to the movie file
        """
        self.path = path
        self.scenes: List[Scene] = []

    def generate_scenes(self):
        generation = SceneGeneration()
        for scene in generation.run(self.path):
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
