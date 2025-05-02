import cv2
import av
import numpy as np
from utils.logger import logger
import multiprocessing as mp
import gc


class VideoProcessor:
    """Process video streams and extract frames at regular intervals."""

    def __init__(self, stream_url):
        """
        Initialize the video processor with a stream URL.

        Args:
            stream_url: URL of the video stream to process
        """
        self.stream_url = stream_url
        self.video_stream = self._open_stream()

        self._duration_ms = self._get_video_duration()

    def __del__(self):
        """Destructor to ensure resources are released"""
        self.close()

    def close(self):
        """Release all resources"""
        # If you store any capture objects as instance variables, release them here
        if self.video_stream and self.video_stream.isOpened():
            self.video_stream.release()

    def _open_stream(self):
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            cap.release()
            raise Exception("Could not open stream")

            # Set resolution before capturing frames
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        return cap

    def _get_video_duration(self):
        """Get the duration of the video in milliseconds."""
        cap = self.video_stream

        # Get frame count and frame rate
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate duration in milliseconds
        duration_ms = (frame_count / fps) * 1000 if fps > 0 else 0

        return duration_ms

    def extract_frames_fast_parallel(
        self, position_ms=0, frequency_ms=100, max_frames=None
    ):
        """
        Extract frames using PyAV with multiprocessing for maximum performance.

        Args:
            position_ms: Starting position in milliseconds
            frequency_ms: Interval between frames in milliseconds
            max_frames: Maximum number of frames to extract

        Returns:
            List of frames as numpy arrays
        """
        # Generate timestamps for all frames we need to extract
        timestamps = []
        current_ms = position_ms
        while current_ms < self._duration_ms:
            timestamps.append(current_ms)
            current_ms += frequency_ms
            if max_frames and len(timestamps) >= max_frames:
                break

        if not timestamps:
            return []

        # Split work among available cores
        num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free for system
        chunk_size = max(1, len(timestamps) // num_cores)
        chunks = [
            timestamps[i : i + chunk_size]
            for i in range(0, len(timestamps), chunk_size)
        ]

        # Use multiprocessing to process chunks in parallel
        # Now using the module-level function instead of a nested function
        with mp.Pool(num_cores) as pool:
            results = pool.starmap(
                _extract_chunk, [(chunk, self.stream_url) for chunk in chunks]
            )

        # Combine and sort results
        all_frames = []
        for result in results:
            all_frames.extend(result)

        # Sort by timestamp and extract just the frames
        all_frames.sort(key=lambda x: x[0])
        return [frame for _, frame in all_frames]

    def frame_generator(self, position_ms=0, frequency_ms=500, max_frames=None):
        """
        Generator that yields frames one at a time with a single stream connection.

        Args:
            position_ms: Starting position in milliseconds
            frequency_ms: Interval between frames in milliseconds
            max_frames: Maximum number of frames to extract

        Yields:
            Tuples of (timestamp_ms, frame) where frame is a numpy array
        """
        try:
            # Open connection only once
            options = {"video_size": "320x240", "hwaccel": "auto"}
            container = av.open(self.stream_url, options=options)
            stream = container.streams.video[0]

            # Calculate all timestamps we'll need
            frame_count = 0
            current_ms = position_ms

            while current_ms < self._duration_ms:
                # Convert ms to stream timebase
                timestamp = int(current_ms * stream.time_base.denominator / 1000)

                # Seek and decode
                container.seek(timestamp, stream=stream)
                for frame in container.decode(video=0):
                    # Convert to numpy array
                    frame = frame.reformat(width=320, height=240)
                    img = frame.to_ndarray(format="bgr24")
                    yield (current_ms, img)
                    break  # Only take first frame after seeking

                # Move to next position
                current_ms += frequency_ms
                frame_count += 1

                if max_frames and frame_count >= max_frames:
                    break

            # Clean up resources when generator is exhausted
            container.close()

        except Exception as e:
            logger.error(f"Error in frame generator: {e}")
        finally:
            # Ensure container is closed even if an exception occurs
            if "container" in locals():
                container.close()

    def extract_frames(self, position_ms=0, frequency_ms=500, max_frames=None):
        """
        Extract frames at regular intervals from the video.

        Args:
            frequency_ms: Interval between frames in milliseconds (default: 500ms)
            max_frames: Maximum number of frames to extract (default: None, extract all possible)

        Returns:
            List of frames as numpy arrays
        """
        if self._duration_ms <= 0:
            return []

        frames = []
        frame_count = 0

        cap = self.video_stream
        if not cap.isOpened():
            return []

        # More efficient to keep the capture open and seek for multiple frames
        while position_ms < self._duration_ms:
            # Set position
            cap.set(cv2.CAP_PROP_POS_MSEC, position_ms)

            # Read frame
            ret, frame = cap.read()

            if ret:
                frames.append(frame)
                frame_count += 1

            # Move to next position
            position_ms += frequency_ms

            # Check if we've reached max frames
            if max_frames and frame_count >= max_frames:
                break
        return frames

    def get_frame_at_position(self, position_ms):
        self.video_stream.set(cv2.CAP_PROP_POS_MSEC, position_ms)
        ret, frame = self.video_stream.read()
        if ret:
            return frame

    @property
    def duration_ms(self):
        """Get the video duration in milliseconds."""
        return self._duration_ms


# Define worker function at module level so it can be pickled
def _extract_chunk(ts_chunk, stream_url):
    """Worker function to extract frames from a video at specified timestamps"""
    frames = []
    try:
        # Each process gets its own container
        options = {"hwaccel": "auto"}
        container = av.open(stream_url, options=options)
        stream = container.streams.video[0]

        for ts in ts_chunk:
            # Calculate timestamp in stream's timebase
            timestamp = int(ts * stream.time_base.denominator / 1000)

            # Seek and decode
            container.seek(timestamp, stream=stream)
            for frame in container.decode(video=0):
                # Convert to numpy array
                frame = frame.reformat(width=320, height=240)
                img = frame.to_ndarray(format="bgr24")
                frames.append((ts, img))  # Store with timestamp for sorting
                break  # Only take first frame after seeking

        container.close()
        # Free memory more aggressively
        del container
        gc.collect()
    except Exception as e:
        logger.error(f"Error in worker process: {e}")

    return frames
