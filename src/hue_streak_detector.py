import numpy as np
from sklearn.cluster import DBSCAN
from config.movie_config import MovieConfig
from scene import Scene


class HueStreakDetector:
    def __init__(self, movie_config: MovieConfig):
        self.config = movie_config
        self.hue_tolerance = movie_config.hue_tolerance
        self.min_scene_frames = movie_config.min_scene_frames
        self.max_gap_frames = movie_config.max_gap_frames
        self.score_threshold = movie_config.score_threshold

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

    def avg_saturation_for_streak(self, streak, frame_colors):
        """
        Calculate average saturation for pixels of a specific hue in a streak
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

        return np.mean(sats) if sats else 0.0

    def score_streak(self, streak, frame_colors, alpha=1.0, beta=1.0):
        """
        Score a streak based on saturation and duration
        """
        start, end = streak["start"], streak["end"]
        target_hue = streak["hue"]

        total_weight = 0
        total_saturation = 0

        for i in range(start, end + 1):
            added = False
            frame = frame_colors[i]
            for h, s in zip(frame["hues"], frame["saturations"]):
                h_diff = abs(int(h) - int(target_hue)) % 180
                dist = min(h_diff, 180 - h_diff)
                if dist <= self.hue_tolerance:
                    added = True
                    total_saturation += float(s)
            total_weight += added

        avg_saturation = total_saturation / total_weight if total_weight > 0 else 0
        duration = end - start + 1

        # Composite score
        score = (avg_saturation**alpha) * (duration**beta)
        return score

    def score_all_streaks(self, streaks, frame_colors, alpha=1.0, beta=1.0):
        """
        Score all streaks in the collection
        """
        for streak in streaks:
            streak["score"] = self.score_streak(streak, frame_colors, alpha, beta)
        return streaks

    def select_top_streaks(self, streaks, frame_colors, min_sat=None):
        """
        Select top non-overlapping streaks by score
        FIXME USE HEAP FOR EFFICIENCY
        """
        if min_sat is None:
            min_sat = 33  # Default minimum saturation

        # Sort by score descending
        sorted_streaks = sorted(streaks, key=lambda s: -s["score"])

        selected = []
        used_ranges = []

        for s in sorted_streaks:
            if (
                s["score"] < self.score_threshold
                or int(self.avg_saturation_for_streak(s, frame_colors)) < min_sat
            ):
                continue

            overlap = any(
                not (s["end"] < u_start or s["start"] > u_end)
                for u_start, u_end in used_ranges
            )
            if not overlap:
                selected.append(s)
                used_ranges.append((s["start"], s["end"]))

        return selected

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
                    "avg_saturation": int(
                        self.avg_saturation_for_streak(s, frame_colors)
                    ),
                    "score": int(s.get("score", 0)),
                }
            )
        return schedule

    def detect_top_streaks(self, frame_colors):
        """
        Main method to detect and return top hue streaks
        """
        # Use config values for clustering parameters
        eps = self.max_gap_frames
        min_samples = self.min_scene_frames // 2

        # Find all hue streaks
        streaks = self.find_all_hue_streaks(
            frame_colors, eps=eps, min_samples=min_samples
        )

        # Score all streaks
        scored_streaks = self.score_all_streaks(streaks, frame_colors, alpha=1, beta=1)

        # Select top non-overlapping streaks
        top_streaks = self.select_top_streaks(scored_streaks, frame_colors)

        # Sort by start time
        sorted_top_streaks = sorted(top_streaks, key=lambda s: s["start"])

        # Convert streak hues to Scene colors
        for s in sorted_top_streaks:
            s["color"] = Scene.pad_hue_to_triple(s["hue"])

        return sorted_top_streaks
