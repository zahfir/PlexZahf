import numpy as np
from sklearn.cluster import DBSCAN
from config.movie_config import MovieConfig
from scene import Scene
import heapq
import time
from utils.logger import LOGGER_NAME
import logging

logger = logging.getLogger(LOGGER_NAME)


class HueStreakDetector:
    def __init__(self, movie_config: MovieConfig):
        self.config = movie_config
        self.hue_tolerance = movie_config.hue_tolerance
        self.min_scene_frames = movie_config.min_scene_frames
        self.max_gap_frames = movie_config.max_gap_frames
        self.score_threshold = movie_config.score_threshold
        self.max_frames = 3000  # Maximum frames for a streak

    def get_hue_presence_array(self, frame_colors, target_hue):
        """
        Returns a binary array: 1 if target_hue is present in the frame within tolerance
        """
        presence = []
        for frame in frame_colors:
            found = False
            for h in frame["hues"]:
                h_diff = abs(int(h) - int(target_hue)) % 180
                dist = min(h_diff, 180 - h_diff)
                if dist <= self.hue_tolerance:
                    found = True
                    break
            presence.append(1 if found else 0)
        return np.array(presence, dtype=np.uint8)

    def run_dbscan_on_presence(self, presence, eps=None, min_samples=None):
        """
        DBSCAN on 1D time index for frames where hue is present
        """
        if eps is None:
            eps = self.max_gap_frames / 2  # Half the max gap as epsilon
        if min_samples is None:
            min_samples = self.min_scene_frames // 2  # Half the min scene frames

        indices = np.where(presence == 1)[0].reshape(-1, 1)
        if len(indices) == 0:
            return []

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(indices)

        # Extract streaks from labels
        streaks = []
        for label in set(labels):
            if label == -1:
                continue
            cluster_indices = indices[labels == label].flatten()
            start = int(cluster_indices[0])
            end = int(cluster_indices[-1])
            streaks.append((start, end))
        return streaks

    def find_all_hue_streaks(self, frame_colors, eps=None, min_samples=None):
        """
        Find streaks for all possible hues using DBSCAN
        """
        hue_streaks = []

        for h in range(180):
            presence = self.get_hue_presence_array(frame_colors, h)
            streaks = self.run_dbscan_on_presence(
                presence, eps=eps, min_samples=min_samples
            )
            for start, end in streaks:
                hue_streaks.append(
                    {"hue": h, "start": start, "end": end, "length": end - start + 1}
                )

        return hue_streaks

    def median_saturation_for_streak(self, streak, frame_colors) -> int:
        """
        Calculate median saturation for pixels of a specific hue in a streak
        Converts 0-255 saturation to 0-100 scale
        """
        start, end = streak["start"], streak["end"]
        target_hue = streak["hue"]

        sats = []
        for i in range(start, end + 1):
            frame = frame_colors[i]
            for h, s in zip(frame["hues"], frame["saturations"]):
                h_diff = abs(int(h) - int(target_hue)) % 180
                dist = min(h_diff, 180 - h_diff)
                if dist <= self.hue_tolerance:
                    sats.append(s)

        median_255 = np.median(sats) if sats else 255
        median_100 = (median_255 / 255) * 100
        return int(median_100)

    def score_streak(self, streak, frame_colors, alpha=1.0, beta=1.0):
        """
        Score a streak based on saturation and duration
        """
        start, end = streak["start"], streak["end"]

        streak["saturation"] = self.median_saturation_for_streak(streak, frame_colors)
        duration = end - start + 1

        # Composite score
        score = (streak["saturation"] ** alpha) * (duration**beta)
        return score

    def score_all_streaks(self, streaks, frame_colors, alpha=1.0, beta=1.0):
        """
        Score all streaks in the collection
        """
        for streak in streaks:
            streak["score"] = self.score_streak(streak, frame_colors, alpha, beta)
        return streaks

    def select_top_streaks(self, streaks, frame_colors, min_sat=33):
        """
        Select top non-overlapping streaks by score using a max heap
        """
        heap = [(int(-streak["score"]), i, streak) for i, streak in enumerate(streaks)]
        heapq.heapify(heap)

        selected = []
        used_ranges = []

        while heap:
            _, i, s = heapq.heappop(heap)

            if s["score"] < self.score_threshold:
                break

            if s["saturation"] < min_sat or s["length"] > self.max_frames:
                continue

            overlap = any(
                not (s["end"] < u_start or s["start"] > u_end)
                for u_start, u_end in used_ranges
            )
            if not overlap:
                selected.append(s)
                used_ranges.append((s["start"], s["end"]))

        return selected

    def plot_streaks_bars(self, streaks, frame_colors, title="Hue Streaks Timeline"):
        """
        Plot streaks as vertical bars that fill the entire height for each time range.
        Each streak's color fills the full vertical space for its duration.

        Args:
            streaks: List of streak dictionaries with 'start', 'end', and 'hue' keys
            frame_colors: List of frame color data to determine timeline length
            title: Title for the plot
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        plt.figure(figsize=(12, 6))

        for streak in streaks:
            start = streak["start"]
            end = streak["end"]
            hue = streak["hue"]

            # Convert hue (0-179) to RGB color for plotting
            # HSV: (hue/179, 1.0, 1.0) -> RGB
            rgb_color = mcolors.hsv_to_rgb([hue / 179.0, 1.0, 1.0])

            # Plot vertical bar that fills the entire height
            plt.axvspan(start, end, color=rgb_color, alpha=0.8)

        plt.xlabel("Frame Index")
        plt.ylabel("Timeline")
        plt.title(title)
        plt.xlim(0, len(frame_colors) - 1)
        plt.yticks([])  # Remove y-axis ticks since height doesn't represent data
        plt.grid(True, alpha=0.3, axis="x")  # Only show vertical grid lines
        plt.show(block=False)

    def get_schedule_from_streaks(self, streaks, frame_colors):
        """
        Convert streaks to a schedule format (more readable)
        """
        schedule = []
        for s in streaks:
            schedule.append(
                {
                    "hue": s["hue"],
                    "start": Scene.ms_to_mmss(int(s["start"] * 1000 / self.config.fps)),
                    "end": Scene.ms_to_mmss(int(s["end"] * 1000 / self.config.fps)),
                    "length": Scene.ms_to_mmss(
                        int(s["length"] * 1000 / self.config.fps)
                    ),
                    "avg_saturation": self.median_saturation_for_streak(
                        s, frame_colors
                    ),
                    "score": int(s.get("score", 0)),
                }
            )
        return schedule

    def detect_top_streaks(self, frame_colors, sat_weight=1, len_weight=1):
        """
        Main method to detect and return top hue streaks
        """
        # Use config values for clustering parameters
        eps = self.max_gap_frames
        min_samples = self.min_scene_frames

        # Find all hue streaks
        s = time.time()
        streaks = self.find_all_hue_streaks(
            frame_colors, eps=eps, min_samples=min_samples
        )
        logger.info(f"Found {len(streaks)} streaks in {time.time() - s:.2f} seconds")

        # Score all streaks
        s = time.time()
        scored_streaks = self.score_all_streaks(
            streaks,
            frame_colors,
            alpha=sat_weight,
            beta=len_weight,
        )
        logger.info(
            f"Scored {len(scored_streaks)} streaks in {time.time() - s:.2f} seconds"
        )

        # Select top non-overlapping streaks
        s = time.time()
        top_streaks = self.select_top_streaks(scored_streaks, frame_colors)
        logger.info(
            f"Selected {len(top_streaks)} top streaks in {time.time() - s:.2f} seconds"
        )

        # Sort by start time
        sorted_top_streaks = sorted(top_streaks, key=lambda s: s["start"])

        # Convert streak hues to Scene colors
        for s in sorted_top_streaks:
            s["color"] = Scene.pad_hue_to_triple(s["hue"])

        return sorted_top_streaks
