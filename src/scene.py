from typing import List, Tuple

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

    @staticmethod
    def ms_to_mmss(ms: int) -> str:
        """
        Convert milliseconds to MM:SS format.

        Args:
            ms: Time in milliseconds

        Returns:
            String in MM:SS format
        """
        total_seconds = ms // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    @property
    def time_range_mmss(self) -> str:
        """Return the scene's time range in MM:SS format."""
        start_mmss = self.ms_to_mmss(self.start)
        end_mmss = self.ms_to_mmss(self.end)
        return f"{start_mmss} - {end_mmss}"

    def to_json(self):
        return {
            "start": self.start,
            "end": self.end,
            "color": self.color,
            "array_index": self.array_index,
        }

    def __repr__(self):
        return f"Scene(i={self.array_index}, start={self.start}, end={self.end}, color={self.color}, timerange={self.time_range_mmss})"
