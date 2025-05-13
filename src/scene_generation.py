from typing import List
import ffmpeg
from io import BytesIO
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from numpy import ndarray
from scenedetect import (
    AdaptiveDetector,
    ContentDetector,
    open_video,
)
from scenedetect.backends.opencv import VideoStreamCv2

from constants import MIN_FRAMES_AFTER_CUT, VIDEO_READ_FREQ_MS
from utils.color.color_analysis import ColorAnalysis
from utils.color.color_utils import bgr_to_rgb
from utils.color.color_averaging import ColorAveraging

from utils.chunk_downloader import ChunkDownloader

from scene import Scene

from utils.logger import logger

import av


class SceneGeneration:
    def __init__(self):

        self.detector = AdaptiveDetector(
            adaptive_threshold=2,
            min_scene_len=MIN_FRAMES_AFTER_CUT,
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

    def start(self, download_url: str):
        # Build stream URL
        parsed_url = urlparse(download_url)
        params = parse_qs(parsed_url.query)
        params = {"X-Plex-Token": params.get("X-Plex-Token", [""])[0]}
        modified_query = urlencode(params, doseq=False)
        stream_url = urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                parsed_url.params,
                modified_query,
                parsed_url.fragment,
            )
        )
        # Create ffmpeg process with pipe output
        process = (
            ffmpeg.input(stream_url)
            .output(
                "pipe:",  # Output to pipe instead of file
                format="mpegts",  # Specify container format
                **{
                    "an": None,
                    "s": "640x268",
                    "b:v": "500k",
                    "r": "5",
                    "preset": "ultrafast",
                    "tune": "fastdecode",
                    "movflags": "+faststart",
                },
            )
            .run_async(pipe_stdout=True)  # Run asynchronously with stdout piped
        )

        # Read from the pipe
        memory_file = BytesIO()
        while True:
            # Read in chunks to avoid memory issues with very large files
            chunk = process.stdout.read(8192)
            if not chunk:
                break
            memory_file.write(chunk)

        # Wait for the process to complete
        process.wait()

        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=5)

        memory_file.seek(0)

        container = av.open(memory_file)

        try:
            video_stream = next(s for s in container.streams if s.type == "video")
        except StopIteration:
            print(f"No video stream found in the container")
            memory_file.close()

        scene_start_ms: int = 0
        for i, frame_data in enumerate(container.decode(video_stream)):
            frame_ndarray = frame_data.reformat(format="bgr24").to_ndarray()
            frame_color = self.get_frame_dominant_color(frame_ndarray)
            self.recent_dominant_colors.append(frame_color)

            cuts: List[int] = self.detector.process_frame(
                frame_num=i,
                frame_img=frame_ndarray,
            )

            if cuts:
                scene_end_ms = int(frame_data.time * 1000)
                logger.debug(f"Cut at frame {cuts[0]}.")
                yield self.new_scene(scene_start_ms, scene_end_ms)

                # Move start pointer up
                scene_start_ms = scene_end_ms

    def run_av_chunks(self, download_url: str):
        i = 1
        for memory_file, num, total in ChunkDownloader(download_url).download_chunks():
            memory_file.seek(0)
            container = av.open(memory_file)

            try:
                video_stream = next(s for s in container.streams if s.type == "video")
            except StopIteration:
                print(f"No video stream found in the container {num}/{total}")
                memory_file.close()
                continue

            for frame in container.decode(video_stream):
                if i % 30 != 0:
                    i += 1
                    continue
                frame_ndarray = frame.reformat(format="bgr24").to_ndarray()
                frame_color = self.get_frame_dominant_color(frame_ndarray)
                self.recent_dominant_colors.append(frame_color)

                cuts: List[int] = self.detector.process_frame(
                    frame_num=i,
                    frame_img=frame_ndarray,
                )

                if cuts:
                    # frame.to_image().save("output.jpg")
                    logger.debug(cuts)

                i += 1

            memory_file.close()

    def run(self, path: str, seek_ms: int = VIDEO_READ_FREQ_MS):
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
