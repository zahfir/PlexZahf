from typing import List, Tuple


class Frame:
    """
    Class representing a single frame with a timestamp and associated color.

    Attributes:
        time: Time in milliseconds
        color: RGB color as a list of 3 integers (0-255)
    """

    def __init__(self, time: int, color: List[int] | Tuple[int, int, int]):
        """
        Initialize a Frame object.

        Args:
            time: Time in milliseconds
            color: RGB color as a list or tuple of 3 integers (0-255)

        Raises:
            ValueError: If time is negative or if color format is invalid
        """
        if time < 0:
            raise ValueError("Time cannot be negative")

        if (
            not isinstance(color, (list, tuple))
            or len(color) != 3
            or not all(0 <= c <= 255 for c in color)
        ):
            raise ValueError(
                "Color must be a list or tuple of 3 integers between 0-255"
            )

        self.time = time
        self.color = list(color)

    def __repr__(self) -> str:
        """String representation of the Frame."""
        return f"Frame(time={self.time}ms, color={self.color})"
