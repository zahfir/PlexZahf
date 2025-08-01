from typing import List, Tuple


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
        saturation: int = 100,
        array_index: int | None = None,
    ):
        """
        Initialize a Scene object.

        Args:
            start: Start time in milliseconds
            end: End time in milliseconds
            color: Hue (0 - 179) at [0] iff [1] < 0 and [2] < 0 or RGB as a list or tuple of 3 integers (0-255)
            saturation: Saturation percentage (0-100)
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
        self.saturation = saturation
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

    @staticmethod
    def pad_hue_to_triple(hue: int) -> List[int]:
        """
        Helper util to make a hue value (0-179) look like an RGB triple.
        """
        if not (0 <= hue <= 179):
            raise ValueError("Hue must be between 0 and 179")
        return [hue, -1, -1]

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
        return f"Scene(i={self.array_index}, ms={self.start}-{self.end}, color={self.color}, sat={self.saturation}, timerange={self.time_range_mmss})"
