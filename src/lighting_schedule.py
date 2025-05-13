import numpy as np
import heapq


class LightingSchedule:
    # -----------------------------
    # Utility: hue distance
    # -----------------------------
    @staticmethod
    def hue_distance(h1, h2):
        diff = np.abs(h1 - h2)
        return np.minimum(diff, 180 - diff)

    # -----------------------------
    # Step 1: Compute frame score
    # -----------------------------
    @classmethod
    def compute_frame_score(cls, frame, target_hue, hue_tol):
        hues = frame["hues"]
        sats = frame["saturations"]
        props = frame["proportions"]

        dists = cls.hue_distance(hues, target_hue)
        mask = dists <= hue_tol

        if not np.any(mask):
            return 0.0

        # Weighted scoring: emphasize prominent + saturated colors
        return np.sum((props[mask] ** 1.5) * sats[mask])

    # -----------------------------
    # Step 2: Compute presence array
    # -----------------------------
    @classmethod
    def compute_hue_presence(cls, frame_colors, target_hue, hue_tol):
        presence = []
        for frame in frame_colors:
            hues = frame["hues"]
            props = frame["proportions"]

            dist = cls.hue_distance(hues, target_hue)

            proportion = sum(prop for diff, prop in zip(dist, props) if diff <= hue_tol)

            presence.append(proportion > 0.5)

        return np.array(presence, dtype=np.uint8)

    # -----------------------------
    # Step 3: Extract scored streaks
    # -----------------------------
    @classmethod
    def extract_scored_streaks(
        cls, frame_scores, presence_array, min_duration, max_gap, min_avg_score
    ):
        streaks = []
        start = None
        gap = 0
        length = 0
        score_sum = 0.0

        for i, present in enumerate(presence_array):
            score = frame_scores[i]
            if present:
                if start is None:
                    start = i
                    length = 1
                    score_sum = score
                    gap = 0
                else:
                    length += 1
                    score_sum += score
                    gap = 0
            else:
                if start is not None:
                    if gap < max_gap:
                        length += 1
                        # score_sum += score
                        gap += 1
                    else:
                        length -= max_gap
                        avg_score = score_sum / length
                        if length >= min_duration and avg_score >= min_avg_score:
                            streaks.append((start, start + length, avg_score))
                        start = None
                        gap = 0
                        length = 0
                        score_sum = 0.0

        if start is not None:
            avg_score = score_sum / length
            if length >= min_duration and avg_score >= min_avg_score:
                streaks.append((start, len(frame_scores) - 1, avg_score))

        return streaks

    # -----------------------------
    # Step 4: Weighted Interval Scheduling
    # -----------------------------
    @classmethod
    def select_optimal_streaks(cls, streaks):
        if not streaks:
            return []
        streaks = sorted(streaks, key=lambda x: x[1])
        n = len(streaks)

        def find_last_non_conflict(i):
            for j in range(i - 1, -1, -1):
                if streaks[j][1] < streaks[i][0]:
                    return j
            return -1

        p = [find_last_non_conflict(i) for i in range(n)]

        dp = [0.0] * n
        dp[0] = streaks[0][2]

        for i in range(1, n):
            incl = streaks[i][2]
            if p[i] != -1:
                incl += dp[p[i]]
            dp[i] = max(incl, dp[i - 1])

        result = []
        i = n - 1
        while i >= 0:
            incl = streaks[i][2]
            prev = p[i]
            if prev != -1:
                incl += dp[prev]

            if incl > (dp[i - 1] if i > 0 else 0):
                result.append(streaks[i])
                i = prev
            else:
                i -= 1

        return result[::-1]

    # -----------------------------
    # Step 5: Dominant RGB extractor
    # -----------------------------
    @classmethod
    def get_most_dominant_rgb_in_range(
        cls, frame_colors, start, end, target_hue, hue_tol
    ):
        rgb_sum = np.zeros(3)
        total_weight = 0

        for i in range(start, end + 1):
            frame = frame_colors[i]
            hues = frame["hues"]
            colors = frame["colors"]
            props = frame["proportions"]

            dists = cls.hue_distance(hues, target_hue)
            mask = dists <= hue_tol

            for idx in np.where(mask)[0]:
                rgb_sum += colors[idx] * props[idx]
                total_weight += props[idx]

        if total_weight == 0:
            return (0, 0, 0)

        return [round(val) for val in (rgb_sum / total_weight).tolist()]

    # -----------------------------
    # Step 6: Generate light instructions
    # -----------------------------
    @classmethod
    def generate_lighting_schedule(cls, frame_colors, streaks, target_hue, hue_tol):
        instructions = []
        for start, end, score in streaks:
            rgb = cls.get_most_dominant_rgb_in_range(
                frame_colors, start, end, target_hue, hue_tol
            )
            instructions.append(
                {"start": start, "end": end, "color": rgb, "score": round(score, 3)}
            )
        return instructions

    # -----------------------------
    # Interface function
    # -----------------------------
    @classmethod
    def find_lighting_instructions(
        cls,
        frame_colors,
        target_hue,
        hue_tol=5,
        min_duration=5,
        max_gap=2,
        min_avg_score=0.1,
    ):
        presence = cls.compute_hue_presence(frame_colors, target_hue, hue_tol)
        scores = [cls.compute_frame_score(f, target_hue, hue_tol) for f in frame_colors]
        streaks = cls.extract_scored_streaks(
            scores, presence, min_duration, max_gap, min_avg_score
        )
        best_streaks = cls.select_optimal_streaks(streaks)
        return cls.generate_lighting_schedule(
            frame_colors, best_streaks, target_hue, hue_tol
        )

    # -----------------------------
    # Interface function
    # -----------------------------
    @classmethod
    def greedy_schedule_from_heap(cls, instructions):
        """
        Greedily selects non-overlapping streaks from a max-heap of instructions.
        Returns: list of selected instructions sorted by start time.
        """
        # Build max-heap based on score
        heap = [
            (int(-instr["score"]), i, instr) for i, instr in enumerate(instructions)
        ]
        heapq.heapify(heap)

        selected = []
        used_intervals = []

        while heap:
            _, i, instr = heapq.heappop(heap)
            s1, e1 = instr["start"], instr["end"]

            # Check for overlap with any accepted interval
            conflict = any(not (e1 < s2 or s1 > e2) for s2, e2 in used_intervals)
            if not conflict:
                selected.append(instr)
                used_intervals.append((s1, e1))

        # Optional: sort selected by start time
        selected.sort(key=lambda x: x["start"])
        return selected
