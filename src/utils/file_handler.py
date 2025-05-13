import ffmpeg
from io import BytesIO
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import av
import numpy as np

from constants import FPS, HEIGHT, WIDTH
from utils.logger import logger


class FileHandler:
    """
    Utility class for handling video file operations including:
    - URL parsing and authentication
    - FFmpeg process management
    - Stream reading and processing
    - Container management
    """

    @staticmethod
    def build_authenticated_url(download_url: str) -> str:
        """Build stream URL with authentication preserved"""
        parsed_url = urlparse(download_url)
        params = parse_qs(parsed_url.query)
        params = {"X-Plex-Token": params.get("X-Plex-Token", [""])[0]}
        modified_query = urlencode(params, doseq=False)
        return urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                parsed_url.params,
                modified_query,
                parsed_url.fragment,
            )
        )

    @staticmethod
    def create_ffmpeg_process(
        url: str, ffmpeg_options: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Create and start an FFmpeg process with the given options"""
        if ffmpeg_options is None:
            ffmpeg_options = {
                "an": None,
                "s": f"{WIDTH}x{HEIGHT}",
                "b:v": "500k",
                "r": f"{FPS}",
                "preset": "ultrafast",
                "tune": "fastdecode",
                "movflags": "+faststart",
            }

        return (
            ffmpeg.input(url)
            .output(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                **ffmpeg_options,
            )
            .run_async(pipe_stdout=True)
        )

    @staticmethod
    def read_stream_to_memory(process: Any, chunk_size: int = 8192) -> BytesIO:
        """Read process output stream into memory"""
        memory_file = BytesIO()

        try:
            while True:
                chunk = process.stdout.read(chunk_size)
                if not chunk:
                    break
                memory_file.write(chunk)

            # Wait for the process to complete
            process.wait()

            # Reset file pointer to beginning
            memory_file.seek(0)
            return memory_file
        except Exception as e:
            logger.error(f"Error reading stream: {e}")
            raise
        finally:
            # Ensure process is terminated
            if process and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

    @staticmethod
    def open_video_container(
        memory_file: BytesIO,
    ) -> Tuple[av.container.Container, av.stream.Stream]:
        """Open a video container from a memory file and get the video stream"""
        container = av.open(memory_file)

        try:
            video_stream = next(s for s in container.streams if s.type == "video")
            return container, video_stream
        except StopIteration:
            logger.error("No video stream found in the container")
            memory_file.close()
            raise ValueError("No video stream found in the container")

    @staticmethod
    def read_frames_from_rawvideo(memory_file: BytesIO, width, height):
        frame_size = width * height * 3  # RGB24
        memory_file.seek(0)  # Ensure we're at start
        while True:
            raw_frame = memory_file.read(frame_size)
            if len(raw_frame) < frame_size:
                break  # End of stream
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
            yield frame

    @classmethod
    def stream_video(
        cls, url: str, ffmpeg_options: Optional[Dict[str, Any]] = None
    ) -> Tuple[BytesIO, av.container.Container, av.stream.Stream]:
        """
        Stream a video from URL to memory and prepare it for processing

        Returns:
            Tuple containing (memory_file, container, video_stream)
        """
        authenticated_url = cls.build_authenticated_url(url)
        process = cls.create_ffmpeg_process(authenticated_url, ffmpeg_options)
        memory_file = cls.read_stream_to_memory(process)
        container, video_stream = cls.open_video_container(memory_file)

        return memory_file, container, video_stream

    @classmethod
    def process_with_direct_pipe(
        cls, url: str, ffmpeg_options: Optional[Dict[str, Any]] = None
    ):
        """
        Process video directly from the pipe without storing complete file in memory

        This generator yields frames directly from the stream
        """
        authenticated_url = cls.build_authenticated_url(url)
        process = cls.create_ffmpeg_process(authenticated_url, ffmpeg_options)

        try:
            # Process frames directly from the pipe
            container = av.open(process.stdout, format="mpegts", mode="r")
            video_stream = next(s for s in container.streams if s.type == "video")

            for frame_data in container.decode(video_stream):
                yield frame_data

        except Exception as e:
            logger.error(f"Error processing direct pipe: {e}")
            raise
        finally:
            # Clean up resources
            if "container" in locals():
                container.close()
            if process and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
