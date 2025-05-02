import threading
import time
from typing import Callable

from plex_service import PlexService


class PlexProgressTracker:
    """Tracks playback progress of a Plex session and provides updates via callbacks."""

    def __init__(self, plex_service: PlexService, username: str, interval: float = 1.0):
        """
        Initialize the progress tracker.

        Args:
            plex_service: The PlexService instance to use for updates
            username: Username whose session to track
            interval: How frequently to check for updates (in seconds)
        """
        self.plex_service = plex_service
        self.username = username
        self.interval = interval
        self.running = False
        self._thread = None
        self._callbacks = []
        self.last_position = 0

    def __del__(self):
        """Ensure thread is stopped when object is garbage collected."""
        self.stop()

    def start(self):
        """Start the tracking thread."""
        if self.running:
            return

        self.running = True
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the tracking thread."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=self.interval * 2)
            self._thread = None

    def add_callback(self, callback: Callable[[int], None]):
        """
        Add a callback function that will be called with updated position.

        Args:
            callback: Function that takes position in milliseconds as argument
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[int], None]):
        """Remove a previously registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _tracking_loop(self):
        """Internal method that polls for updates and triggers callbacks."""
        while self.running:
            try:
                # Find the session for this user
                session = self.plex_service.find_session_by_username(self.username)

                if session and hasattr(session, "viewOffset"):
                    current_position = session.viewOffset

                    # If position changed, notify callbacks
                    if current_position != self.last_position:
                        self.last_position = current_position
                        self._notify_callbacks(current_position)
            except Exception as e:
                print(f"Error updating playback position: {e}")

            # Wait for next update interval
            time.sleep(self.interval)

    def _notify_callbacks(self, position: int):
        """Notify all registered callbacks with the new position."""
        for callback in self._callbacks:
            try:
                callback(position)
            except Exception as e:
                print(f"Error in callback: {e}")

    @property
    def current_position(self) -> int:
        """Get the last known position."""
        return self.last_position
