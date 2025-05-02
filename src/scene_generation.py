from typing import List
from numpy import ndarray
from scenedetect import (
    AdaptiveDetector,
    ContentDetector,
    open_video,
)
from scenedetect.backends.opencv import VideoStreamCv2
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_detector import FlashFilter

from color_analysis import ColorAnalysis
from scene import Scene
from utils.logger import logger
from utils.color_utils import bgr_to_rgb
from utils.color_averaging import ColorAveraging


class SceneGeneration:
    def __init__(self):

        self.detector = AdaptiveDetector(
            adaptive_threshold=2,
            min_scene_len=125,
            window_width=2,
            min_content_val=15,
            weights=ContentDetector.Components(
                delta_hue=3,
                delta_sat=3,
                delta_lum=1,
                delta_edges=0,
            ),
        )

        self.recent_dominant_colors: List[List[int]] = []  # RGB

        self.num_scenes: int = 0

    def run(self, path: str, seek_ms: int = 500):
        """
        Generator function for scenes of entire video

        Yields:
            Scenes with average color included
        """
        video: VideoStreamCv2 = open_video(path=path, backend="opencv")
        if not video.is_seekable:
            raise Exception("SceneGeneration: Video file is not seekable")
        video.seek(1)  # Seek to start

        seek: float = seek_ms / 1000
        scene_start_ms: int = 0
        while True:
            frame = self.get_current_frame(video=video)

            frame_color = self.get_frame_dominant_color(frame=frame)
            self.recent_dominant_colors.append(frame_color)

            cuts: List[int] = self.detector.process_frame(
                frame_num=video.frame_number,
                frame_img=frame,
            )

            if cuts:
                logger.info(cuts)
                scene_end_ms = int(video.position_ms)
                yield self.new_scene(scene_start_ms, scene_end_ms)

                # Move start pointer up
                scene_start_ms = scene_end_ms

            next_position = video.position.get_seconds() + seek

            if next_position >= video.duration.get_seconds():
                logger.info("Finished last scene.")
                scene_end_ms = int(video.position_ms)
                yield self.new_scene(scene_start_ms, scene_end_ms)
                break

            video.seek(target=float(next_position))

    def new_scene(
        self,
        start: int,
        end: int,
    ) -> Scene:
        scene = Scene(
            start=start,
            end=end,
            color=self.average_recent_colors(),
            array_index=self.num_scenes,
        )

        # Clear recent colors for next scene
        self.recent_dominant_colors = []
        self.num_scenes += 1

        return scene

    def get_current_frame(self, video: VideoStreamCv2):
        """Returns current video frame as a numpy.ndarray"""
        return video.read(advance=False)

    def get_frame_dominant_color(self, frame: ndarray):
        """Returns the dominant RGB of a frame with ColorAnalysis"""
        bgr = ColorAnalysis.extract_dominant_color_hsv(frame=frame)
        return bgr_to_rgb(bgr)

    def average_recent_colors(self) -> List[int]:
        """Returns a single RGB average of the recent colors"""
        return ColorAveraging.median(self.recent_dominant_colors)
