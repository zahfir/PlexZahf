import time
from typing import Callable

from plex_progress_tracker import PlexProgressTracker
from utils.logger import logger


class PlaybackSimulator:
    """
    Simulates continuous playback position between Plex tracker updates.

    This class uses the periodic updates from PlexProgressTracker combined with
    the local system clock to estimate the current position in the video between
    updates from Plex.
    """

    def __init__(self, tracker: PlexProgressTracker, playback_speed: float = 1.0):
        """
        Initialize the playback simulator.

        Args:
            tracker: The PlexProgressTracker to listen to for position updates
            playback_speed: Playback speed multiplier (1.0 = normal speed)
        """
        self.tracker = tracker
        self.playback_speed = playback_speed
        self._last_position_ms = 0
        self._last_update_time = time.time()
        self._is_playing = False
        self._callbacks = []

        # Register with the tracker
        self.tracker.add_callback(self._on_position_update)

    def __del__(self):
        """Clean up by removing the callback when object is destroyed."""
        try:
            if self.tracker:
                self.tracker.remove_callback(self._on_position_update)
        except:
            pass

    def _on_position_update(self, position_ms: int):
        """
        Handle position updates from the Plex tracker.

        Args:
            position_ms: Current position in milliseconds
        """
        now = time.time()

        # Calculate elapsed video time since last update
        if self._is_playing:
            elapsed_real_time = now - self._last_update_time
            expected_elapsed_ms = int(elapsed_real_time * 1000 * self.playback_speed)
            expected_position = self._last_position_ms + expected_elapsed_ms

            # If the difference between expected and actual position is large,
            # playback likely paused, skipped, or had speed change
            if abs(position_ms - expected_position) > 3000:  # 3-second threshold
                logger.debug(
                    f"Playback jump detected: expected {expected_position}ms, got {position_ms}ms"
                )

        # Update our tracking variables
        self._last_position_ms = position_ms
        self._last_update_time = now
        self._is_playing = True

        # Notify our own callbacks
        self._notify_callbacks(position_ms)

    def _notify_callbacks(self, position_ms: int):
        """Notify all registered callbacks with the position."""
        for callback in self._callbacks:
            try:
                callback(position_ms)
            except Exception as e:
                logger.error(f"Error in simulator callback: {e}")

    def add_callback(self, callback: Callable[[int], None]):
        """
        Add a callback that will be triggered on position updates.

        Args:
            callback: Function to call with position in milliseconds
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[int], None]):
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    @property
    def current_position_ms(self) -> int:
        """
        Get the current estimated position in milliseconds.

        This calculates the position based on the last update from Plex
        and the elapsed time since that update.

        Returns:
            Current estimated position in milliseconds
        """
        if not self._is_playing:
            return self._last_position_ms

        # Calculate time elapsed since last update
        elapsed_seconds = time.time() - self._last_update_time
        elapsed_ms = int(elapsed_seconds * 1000 * self.playback_speed)

        # Return estimated current position
        return self._last_position_ms + elapsed_ms

    @property
    def is_playing(self) -> bool:
        """Returns whether playback is active."""
        return self._is_playing

    def set_playback_speed(self, speed: float):
        """
        Set the playback speed multiplier.

        Args:
            speed: Speed multiplier (1.0 = normal speed)
        """
        # Update current position before changing speed
        current_position = self.current_position_ms
        self._last_position_ms = current_position
        self._last_update_time = time.time()

        self.playback_speed = max(0.0, speed)
