from typing import List, Tuple

from constants import MERGE_SCENE_COLOR_THRESHOLD, MIN_MERGE_SCENE_DURATION_MS
from utils.color.color_utils import color_difference


class Scene:
    """
    Class representing a scene with a start time, end time, and associated color.

    Attributes:
        start: Start time in milliseconds
        end: End time in milliseconds
        color: RGB color as a list of 3 integers (0-255)
    """

    def __init__(
        self,
        start: int,
        end: int,
        color: List[int] | Tuple[int, int, int],
        array_index: int | None = None,
    ):
        """
        Initialize a Scene object.

        Args:
            start: Start time in milliseconds
            end: End time in milliseconds
            color: RGB color as a list or tuple of 3 integers (0-255)

        Raises:
            ValueError: If end time is not after start time or if color format is invalid
        """
        if end < start:
            raise ValueError("End time must be after start time")

        if not isinstance(color, (list, tuple)) or len(color) != 3:
            raise ValueError("Color must be a list or tuple of 3 integers")

        self.start = start
        self.end = end
        self.color = list(color)
        self.array_index = array_index

    def duration(self) -> int:
        """Return the duration of the scene in milliseconds."""
        return self.end - self.start

    def is_combinable(
        self,
        other_scene: "Scene",
        min_length=MIN_MERGE_SCENE_DURATION_MS,
        color_threshold=MERGE_SCENE_COLOR_THRESHOLD,
    ):
        """Returns True if other_scene is small and similar enough to be merged"""
        color1, color2 = self.color, other_scene.color

        scene_is_short: bool = other_scene.duration() < min_length
        scene_colors_similar: bool = color_difference(color1, color2) < color_threshold

        return scene_is_short and scene_colors_similar

    def __repr__(self) -> str:
        """String representation of the Scene."""
        return f"Scene(i={self.array_index}, start={self.start}ms, end={self.end}ms, color={self.color})"
