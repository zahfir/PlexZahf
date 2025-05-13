import requests
from io import BytesIO
import urllib.parse

from utils.logger import logger

from constants import CHUNK_SIZE_BYTES


class ChunkDownloader:
    def __init__(self, url):
        """Initialize with the URL to download."""
        self.url = url
        self._file_size = self._get_file_size()
        self._filename = self._get_filename()

    def _get_file_size(self):
        """Get the total size of the file."""
        response = requests.head(self.url)
        if "Content-Length" in response.headers:
            return int(response.headers["Content-Length"])
        else:
            raise ValueError(
                "Cannot determine file size. Content-Length header is missing."
            )

    def _get_filename(self):
        """Extract filename from URL."""
        url_path = urllib.parse.urlparse(self.url).path
        return url_path.split("/")[-1]

    def download_chunks(self, chunk_size_bytes=CHUNK_SIZE_BYTES):  # 1GB default
        """
        Download the file in chunks and yield each chunk as a BytesIO object.

        Args:
            chunk_size_bytes: Size of each chunk in bytes

        Yields:
            tuple: (BytesIO object containing the chunk data, chunk number, total chunks)
        """
        total_chunks = (self._file_size + chunk_size_bytes - 1) // chunk_size_bytes

        for chunk_num in range(total_chunks):
            logger.info(f"Downloading {chunk_num}...")
            start_byte = chunk_num * chunk_size_bytes
            end_byte = min(start_byte + chunk_size_bytes - 1, self._file_size - 1)

            # Download the chunk
            headers = {"Range": f"bytes={start_byte}-{end_byte}"}
            response = requests.get(self.url, headers=headers, stream=True)

            if response.status_code not in (200, 206):
                raise Exception(f"Failed to download chunk: {response.status_code}")

            # Create memory file
            memory_file = BytesIO(response.content)
            logger.info(f"Downloaded {chunk_num}!")

            # Yield the memory file along with metadata
            yield memory_file, chunk_num + 1, total_chunks

            # Clear memory
            memory_file.close()
            del memory_file

    @property
    def file_size(self):
        """Get total file size in bytes."""
        return self._file_size

    @property
    def filename(self):
        """Get the filename."""
        return self._filename
