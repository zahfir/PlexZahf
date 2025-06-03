from plexapi.myplex import MyPlexAccount
from plexapi.server import PlexServer
import logging
from utils.logger import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class PlexService:
    """Service class to interact with Plex API and retrieve stream URLs."""

    def __init__(self, username: str, secret: str, server: str):
        """
        Initialize the PlexService with account credentials and server name.

        Args:
            username: Plex account username
            password: Plex account password
            server_name: Name of the Plex server to connect to
        """
        self.username = username
        self.server: PlexServer = self._server_login(
            username=username,
            secret=secret,
            server=server,
        )

        self.session = self.find_session_by_username(username)

    def _server_login(self, username: str, secret: str, server: str):
        try:
            if "http" in server:
                # If the server name contains 'http', treat it as a base URL
                return PlexServer(baseurl=server, token=secret)

            # Otherwise, assume local server and log in as user/pass account
            self.account = MyPlexAccount(username=username, password=secret)
            return self.account.resource(server).connect()
        except Exception as e:
            raise Exception(f"Error logging into Plex server: {e}")

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
                self.session = session
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
        if session:
            return self.get_direct_file_url(session) or session.getStreamURL()

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

    def get_direct_file_url(self, session) -> str | None:
        """
        Get the direct file URL for the given session.

        Args:
            session: The Plex session object

        Returns:
            URL string for the direct media file or None if not available
        """
        try:
            # Search for the media in the session
            query: str = session._prettyfilename()
            # if query ends with a year, remove it
            if query.endswith(")"):
                query = query[:-7]

            search_results = self.server.search(query=query)

            file_key = search_results[0].media[0].parts[0].key

            return self.server.url(file_key, includeToken=True)
        except Exception as e:
            logger.info(f"Couldn't get direct file URL: {e}")
            return None

    def now_playing(self):
        """
        Get the currently playing media information.

        Returns:
            dict: {
                "artUrl": URL for the media's artwork
                "thumbUrl": URL for the media's thumbnail
                "title": Pretty filename of the media
            }
        """
        if self.session:
            return {
                "artUrl": self.session.artUrl,
                "thumbUrl": self.session.thumbUrl,
                "title": self.session._prettyfilename(),
            }
