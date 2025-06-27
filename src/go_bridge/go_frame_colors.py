import subprocess
import json
import os
from typing import Dict, Any, List
import numpy as np
from logging import getLogger
from utils.logger import LOGGER_NAME

logger = getLogger(LOGGER_NAME)


class GoFrameColors:
    """Bridge class to interface with the Go code and frame color extraction tool."""

    def __init__(self, go_executable_path: str = "go/extract.exe"):
        """Initialize the analyzer with path to the Go executable.

        Args:
            go_executable_path: Path to the compiled Go executable
        """
        self.go_executable = go_executable_path

    def video_to_frame_colors(
        self,
        video_path: str,
        output_dir: str = "./results",
        start_time: str | None = None,
        end_time: str | None = None,
        sample_rate: int = 5,
        pixels_per_frame: int = 5000,
    ) -> List[dict]:
        """Analyze video colors using the Go frame extractor.

        Args:
            video_path: Path to video file (local or URL)
            output_dir: Directory to save analysis results
            start_time: Start time (format: HH:MM:SS)
            end_time: End time (format: HH:MM:SS)
            sample_rate: Process every Nth frame
            pixels_per_frame: Number of random pixels to sample per frame

        Returns:
            Dictionary containing the analysis results

        Raises:
            ValueError: If video_path is empty
            subprocess.CalledProcessError: If the Go process fails
            FileNotFoundError: If results file doesn't exist after processing
        """
        if not video_path:
            raise ValueError("Video path is required")

        # Build command arguments
        cmd = [self.go_executable, "-video", video_path, "-output", output_dir]

        if start_time:
            cmd.extend(["-start", start_time])
            if end_time:
                cmd.extend(["-end", end_time])

        cmd.extend(["-sample-rate", str(sample_rate), "-pixels", str(pixels_per_frame)])

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Run Go executable
        try:
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            logger.info(f"Go process completed with exit code {result.returncode}")

            # For debugging
            if result.stderr:
                logger.error(f"STDERR output:\n{result.stderr}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running Go executable: {e}")
            logger.error(f"STDERR: {e.stderr}")
            raise

        # Read results from the JSON file
        results_file = os.path.join(output_dir, "analysis_results.json")
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")

        with open(results_file, "r") as f:
            results: List[dict] = json.load(f)

        results.sort(key=lambda x: x["frame_number"])
        return GoFrameColors._to_ndarrays(results)

    @staticmethod
    def _to_ndarrays(results: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Convert results to NumPy arrays for easier manipulation.

        Args:
            results: Dictionary containing the analysis results

        Returns:
            Dictionary with NumPy arrays for colors and proportions
        """
        for frame in results:
            frame["colors"] = np.array(frame["colors"])
            frame["proportions"] = np.array(frame["proportions"])
            frame["hues"] = np.array(frame["hues"])
            frame["saturations"] = np.array(frame["saturations"])

        return results
