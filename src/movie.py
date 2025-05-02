from typing import List, Optional
import numpy as np

from color_analysis import ColorAnalysis
from scene_generation import SceneGeneration
from utils.logger import logger
from scene import Scene
from frame import Frame
from video_processor import VideoProcessor
from scenedetect import (
    ContentDetector,
    SceneManager,
    open_video,
)


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
        self.frames: List[Frame] = []
        # self.processor = VideoProcessor(path)
        # self.duration_ms = self.processor.duration_ms

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
        print()

    def extend_last_scene(self, new_end_position: int):
        self.scenes[-1].end = new_end_position

    def extract_frames_streaming(
        self,
        frequency_ms=1000,
        build_scenes=True,
        color_threshold=40,
        min_duration_ms=10000,
    ) -> List[Frame]:
        """
        Process frames one by one as they are extracted from the video stream.
        Optionally build scenes at the same time.

        Args:
            frequency_ms: Interval between frames in milliseconds
            build_scenes: Whether to build scenes while processing frames
            color_threshold: RGB difference threshold to consider frames as part of the same scene
            min_duration_ms: Minimum scene duration in milliseconds

        Returns:
            List of processed Frame objects
        """
        extracted_frames = []

        # Scene tracking variables
        if build_scenes:
            current_start = None
            current_scene_color = None

        # Get frames one by one using the generator
        for timestamp_ms, raw_frame in self.processor.frame_generator(
            frequency_ms=frequency_ms
        ):

            # Process this frame immediately
            b, g, r = ColorAnalysis.extract_dominant_color_hsv(raw_frame)
            avg_color = [int(r), int(g), int(b)]

            # Create Frame object
            frame = Frame(timestamp_ms, avg_color)
            self.frames.append(frame)
            extracted_frames.append(frame)

            # Optional scene building
            if build_scenes:
                if current_start is None:
                    # Initialize first scene
                    current_start = 0
                    current_scene_color = avg_color
                else:
                    # Check for scene change
                    color_diff = sum(
                        abs(a - b) for a, b in zip(current_scene_color, avg_color)
                    )

                    if color_diff > color_threshold:
                        # End current scene if it meets minimum duration
                        if timestamp_ms - current_start >= min_duration_ms:
                            new_scene = Scene(
                                current_start,
                                timestamp_ms,
                                current_scene_color,
                                len(self.scenes),
                            )
                            self.scenes.append(new_scene)
                            logger.debug(
                                f"Created scene - {round((timestamp_ms/self.duration_ms) * 100)}% - {new_scene}"
                            )

                        # Start new scene
                        current_start = timestamp_ms
                        current_scene_color = avg_color

        # Add final scene if needed and meets duration requirements
        if build_scenes and current_start is not None and extracted_frames:
            last_frame_time = extracted_frames[-1].time
            if last_frame_time - current_start >= min_duration_ms:
                final_scene = Scene(
                    current_start,
                    last_frame_time,
                    current_scene_color,
                    len(self.scenes),
                )
                self.scenes.append(final_scene)
                logger.debug(f"Created final scene: {final_scene}")

        return extracted_frames

    def on_new_scene(self, frame_img: np.ndarray, frame_num: int):
        print("New scene found at frame %d." % frame_num)
        # Fetch the scene details

        # Create Scene and add to self

        # possibly give it the frame_img color until the 2nd pass

    def detect_scenes_with_colors(
        self,
        min_scene_len: int = 200,
        threshold: float = 30.0,
        frame_skip: int = 24,
        color_samples_per_scene: int = 5,
    ) -> List[Scene]:
        """
        Detect scenes and extract their colors efficiently.

        Args:
            min_scene_len: Minimum scene length in frames
            threshold: Threshold for content detection
            frame_skip: Number of frames to skip during detection
            color_samples_per_scene: Number of frames to sample for color in each scene

        Returns:
            List of Scene objects with colors
        """
        video = open_video(path=self.path, backend="pyav")
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(min_scene_len=min_scene_len))
        scene_manager.detect_scenes(
            video,
            frame_skip=frame_skip,
            callback=self.on_new_scene,
        )
        scenes = scene_manager.get_scene_list()

    def extract_frames(self, frequency_ms: int = 1000) -> List[Frame]:
        """
        Extract frames from the movie at regular intervals and add them to the frames list.

        Args:
            frequency_ms: Interval between frames in milliseconds

        Returns:
            List of extracted Frame objects
        """
        extracted_frames = []

        # Extract raw frame images
        raw_frames = self.processor.extract_frames_fast(frequency_ms=frequency_ms)
        logger.info(f"{len(raw_frames)} frames")

        # Process each frame to extract dominant color
        position_ms = 0
        for raw_frame in raw_frames:
            # Calculate average color
            b, g, r = ColorAnalysis.extract_dominant_color_hsv(raw_frame)
            avg_color = [int(r), int(g), int(b)]

            # Create Frame object
            frame = Frame(position_ms, avg_color)
            self.frames.append(frame)
            extracted_frames.append(frame)

            position_ms += frequency_ms

        return extracted_frames

    def generate_scenes_from_frames(
        self, color_threshold: int = 30, min_duration_ms: int = 10000
    ) -> List[Scene]:
        """
        Generate scenes by grouping similar consecutive frames.

        Args:
            color_threshold: RGB difference threshold to consider frames as part of the same scene
            min_duration_ms: Minimum scene duration in milliseconds

        Returns:
            List of generated Scene objects
        """
        if not self.frames or len(self.frames) < 2:
            return []

        scenes = []
        current_start = self.frames[0].time
        current_color = self.frames[0].color

        for i in range(1, len(self.frames)):
            frame = self.frames[i]

            # Calculate color difference
            color_diff = sum(abs(a - b) for a, b in zip(current_color, frame.color))

            # If color changed significantly, end the current scene and start a new one
            if color_diff > color_threshold:
                # Only create scene if it meets minimum duration
                if frame.time - current_start >= min_duration_ms:
                    new_scene = Scene(current_start, frame.time, current_color)
                    scenes.append(new_scene)
                    self.scenes.append(new_scene)

                # Start new scene
                current_start = frame.time
                current_color = frame.color

        # Add final scene if it meets the duration requirement
        if self.frames[-1].time - current_start >= min_duration_ms:
            final_scene = Scene(current_start, self.frames[-1].time, current_color)
            scenes.append(final_scene)
            self.scenes.append(final_scene)

        return scenes

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

    def __del__(self):
        """Ensure resources are released."""
        if hasattr(self, "processor"):
            self.processor.close()


if __name__ == "__main__":
    avatar_path = (
        "Avatar_ The Last Airbender S1 _ Episode 1 _ The Boy in the Iceberg.mp4"
    )
    path = "The.Long.Goodbye.1973.1080p.BluRay.x264-[YTS.AM].mp4"
    movie = Movie(path=avatar_path)
    movie.generate_scenes()
    print()
