from plexapi.myplex import MyPlexAccount
from plexapi.server import PlexServer


class PlexService:
    """Service class to interact with Plex API and retrieve stream URLs."""

    def __init__(self, username: str, password: str, server_name: str):
        """
        Initialize the PlexService with account credentials and server name.

        Args:
            username: Plex account username
            password: Plex account password
            server_name: Name of the Plex server to connect to
        """
        self.account = MyPlexAccount(username=username, password=password)
        self.server: PlexServer = self.account.resource(server_name).connect()

    def find_session_by_username(self, username: str):
        """
        Find a session that contains the given username in its usernames list.

        Args:
            username: Username to search for

        Returns:
            The first session matching the username, or None if not found
        """
        for session in self.server.sessions():
            # Check if this session belongs to the user we're looking for
            if hasattr(session, "usernames") and username in session.usernames:
                return session
        return None

    def get_stream_url(self, session, quality=None, resolution=None) -> str:
        """
        Get the streaming URL for the given session with transcoding options.

        Args:
            session: The Plex session object
            quality: Video quality (bitrate in kbps e.g. "128")
            resolution: Desired resolution (e.g., "320x240")

        Returns:
            URL string for the transcoded media stream
        """
        if not session or not hasattr(session, "media") or not session.media:
            return ""

        media = session.media[0]
        if not media.parts:
            return ""

        part = media.parts[0]
        base_url = f"{self.server._baseurl}{part.key}?X-Plex-Token={self.server._token}"

        # Add transcoding parameters
        quality_params = resolution_params = ""
        if quality:
            quality_params = f"&videoQuality={quality}&maxVideoBitrate={quality}"
        if resolution:
            resolution_params = f"&videoResolution={resolution}"

        transcode_params = quality_params + resolution_params

        return base_url + transcode_params

    def get_stream_url_by_username(self, username: str) -> str | None:
        """
        Convenience method to find a session by username and return its stream URL.

        Args:
            username: Username to search for

        Returns:
            URL string for the media stream or None if not found
        """
        session = self.find_session_by_username(username)
        if not session:
            return None
        return self.get_stream_url(session, "128", "320x240")
