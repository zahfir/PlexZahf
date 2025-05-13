from av import VideoFrame


class Frame:
    """
    Class representing a single frame with a timestamp and associated color.

    Attributes:
        time: Time in milliseconds
        color: RGB color as a list of 3 integers (0-255)
    """

    def __init__(self, frame_data: VideoFrame):
        """
        Initializes a lightweight Frame object from VideoFrame data.

        """
        self.time = frame_data.time * 1000  # S --> MS

        # TODO FRAME ANALYSIS FOR SCORES
        self.color_scores = {}

    def __repr__(self) -> str:
        """String representation of the Frame."""
        return f"Frame(time={self.time}ms, color={self.color})"
