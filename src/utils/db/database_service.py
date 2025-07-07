import sqlite3
from pathlib import Path
from typing import List
from uuid import uuid4

import numpy as np

from scene import Scene
from config.movie_config import MovieConfig

DEFAULT_DB_FILEPATH = "database.db"


class DatabaseService:
    """A service class for SQLite database operations."""

    def __init__(self, db_path=DEFAULT_DB_FILEPATH):
        """Initialize database connection.

        Args:
            db_path (str): Path to the SQLite database file
        """
        try:
            # Create directory if it doesn't exist
            db_file = Path(db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

            # Connect to database
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()

            # Enable foreign key support
            self.cursor.execute("PRAGMA foreign_keys = ON")

            sqlite3.register_adapter(np.int64, int)
            sqlite3.register_adapter(np.int32, int)
            sqlite3.register_adapter(np.int16, int)
            sqlite3.register_adapter(np.int8, int)
            sqlite3.register_adapter(np.uint8, int)
            sqlite3.register_adapter(np.uint16, int)
            sqlite3.register_adapter(np.uint32, int)
            sqlite3.register_adapter(np.uint64, int)
            sqlite3.register_adapter(np.float64, float)
            sqlite3.register_adapter(np.float32, float)

            print(f"Connected to database: {db_path}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            raise

    def _create_tables(self):
        """Create all necessary database tables if they don't exist."""
        try:
            # Create MOVIE table
            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS MOVIE (
                movie_id TEXT PRIMARY KEY,
                name TEXT NOT NULL
            )
            """
            )

            # Create FRAME table
            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS FRAME (
                frame_id TEXT PRIMARY KEY,
                idx INTEGER NOT NULL,
                movie_id TEXT NOT NULL,
                FOREIGN KEY (movie_id) REFERENCES MOVIE(movie_id)
            )
            """
            )

            # Create SCENE table
            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS SCENE (
                scene_id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_id TEXT NOT NULL,
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                red INTEGER NOT NULL,
                green INTEGER NOT NULL,
                blue INTEGER NOT NULL,
                saturation INTEGER DEFAULT 100,
                FOREIGN KEY (config_id) REFERENCES MOVIE_CONFIG(config_id)
            )
            """
            )

            # Create MOVIE_CONFIG table
            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS MOVIE_CONFIG (
                config_id TEXT PRIMARY KEY,
                movie_id TEXT NOT NULL,
                fps INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                hue_tolerance INTEGER NOT NULL,
                max_gap_frames INTEGER NOT NULL,
                min_scene_frames INTEGER NOT NULL,
                scene_boundary_threshold_ms INTEGER NOT NULL,
                score_threshold INTEGER NOT NULL,
                colorful_bias INTEGER NOT NULL,
                FOREIGN KEY (movie_id) REFERENCES MOVIE(movie_id)
            )
            """
            )

            # Create FRAME_COLOR table
            self.cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS FRAME_COLOR (
                color_id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id TEXT NOT NULL,
                proportion REAL NOT NULL,
                red INTEGER NOT NULL,
                green INTEGER NOT NULL,
                blue INTEGER NOT NULL,
                hue INTEGER NOT NULL,
                sat INTEGER NOT NULL,
                FOREIGN KEY (frame_id) REFERENCES FRAME(frame_id)
            )
            """
            )

            self.conn.commit()
            print("All tables created successfully")
        except sqlite3.Error as e:
            self.conn.rollback()
            print(f"Error creating tables: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if hasattr(self, "conn") and self.conn:
            self.conn.close()

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()

    def __enter__(self):
        """Support for context manager protocol (with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure connection is closed when exiting context."""
        self.close()
        return False  # Allow exceptions to propagate

    @staticmethod
    def _new_id() -> str:
        """Generate a new unique ID for a movie or frame."""
        return str(uuid4())

    def load_movie(self, name) -> str:
        """Load a movie by its name. Returns the movie ID."""
        try:
            return self.get_movie_by_name(name)[0]
        except:
            raise MovieNotFoundError()

    def load_movie_frame_colors(self, movie_id):
        """Load frame colors for a movie by its name. Returns a list of frame dictionaries."""
        rows = self.get_frame_colors_by_movie_id(movie_id)
        return [frame for frame in self.frame_color_rows_to_dicts(rows)]

    def load_movie_schedule(
        self,
        movie_id: str,
        config: MovieConfig,
    ) -> List[Scene]:
        """Load the schedule for a movie by its name and config.

        Args:
            movie_id (str): The movie ID.
            config (MovieConfig): The movie configuration.

        Returns:
            List[Scene]: List of scenes for the movie.
        """
        db_config = self.get_movie_config(movie_id, config)

        if not db_config:
            raise ConfigNotFoundError()

        rows = self.get_scenes_by_config_id(db_config[0])

        return [DatabaseService._scene_from_row(i, row) for i, row in enumerate(rows)]

    @staticmethod
    def _scene_from_row(i, row):
        """Convert a database row to a Scene object."""
        return Scene(
            start=row[2],
            end=row[3],
            color=[row[4], row[5], row[6]],
            saturation=(
                row[7] if len(row) > 7 else 100
            ),  # Handle old rows without saturation
            array_index=i,
        )

    def save_schedule(
        self,
        movie_id: str,
        scenes: List[Scene],
        config: MovieConfig,
    ):
        """Save new Scenes + Config to the DB.
        Two tables are updated - SCENE, MOVIE_CONFIG

        Args:
            movie_id (str): The movie ID.
            scenes (list): List of Scene objects.
            config (MovieConfig): Configuration for the movie.
        """
        try:
            config_id = self.add_scenes_and_config(
                movie_id=movie_id,
                scenes=scenes,
                config=config,
            )
            print(f"Scenes saved successfully with config id: {config_id}")
            return config_id
        except sqlite3.Error as e:
            self.conn.rollback()
            print(f"Error saving movie: {e}")

    def save_file(self, name, frame_dicts):
        """Save a new Movie + Frames + Colors to the DB.
        Three tables are updated - MOVIE, FRAME, FRAME_COLOR

        Args:
            name (str): The name of the movie.
        """
        try:
            movie_id = self.add_movie(name)

            frame_rows, frame_color_rows = self.rows_from_frame_dicts(
                movie_id=movie_id,
                frame_dicts=frame_dicts,
            )
            self.add_frames(frame_rows)
            self.add_frame_colors(frame_color_rows)
            print(f"Movie '{name}' saved successfully with movie_id: {movie_id}")
            return movie_id
        except sqlite3.Error as e:
            self.conn.rollback()
            print(f"Error saving movie: {e}")

    def get_movie_by_name(self, name):
        """Get a movie by its ID.

        Args:
            name (str): The Plex "prettyfilename" of the movie to find.

        Returns:
            tuple: Movie data or None if not found.
        """
        try:
            self.cursor.execute(
                """
                SELECT * FROM MOVIE WHERE name = ?
                """,
                (name,),
            )
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print(f"Error retrieving movie: {e}")
            return None

    # Method for looking up a [Scene] with a config ID
    def get_scenes_by_config_id(self, config_id):
        """Retrieve all scenes with a given config_id, ordered by start time.

        Args:
            config_id (str): The config_id to filter scenes by.

        Returns:
            list: A list of tuples, where each tuple represents a scene.
                  Returns an empty list if no scenes are found.
        """
        try:
            self.cursor.execute(
                """
                SELECT *
                FROM SCENE
                WHERE config_id = ?
                ORDER BY start
                """,
                (config_id,),
            )
            scenes = self.cursor.fetchall()
            return scenes
        except sqlite3.Error as e:
            print(f"Error retrieving scenes by config_id: {e}")
            return []

    def get_movie_config(
        self,
        movie_id,
        config: MovieConfig,
    ):
        """Retrieve a movie config based on its attributes.

        Args:
            movie_id (str): The movie ID.
            config (MovieConfig): The movie configuration.

        Returns:
            tuple: A tuple representing the movie config, or None if not found.
        """
        try:
            self.cursor.execute(
                """
                SELECT *
                FROM MOVIE_CONFIG
                WHERE movie_id = ? AND fps = ? AND width = ? AND height = ? 
                AND hue_tolerance = ? AND max_gap_frames = ? AND min_scene_frames = ? 
                AND scene_boundary_threshold_ms = ? AND score_threshold = ? AND colorful_bias = ?
                """,
                (
                    movie_id,
                    config.fps,
                    config.width,
                    config.height,
                    config.hue_tolerance,
                    config.max_gap_frames,
                    config.min_scene_frames,
                    config.scene_boundary_threshold_ms,
                    config.score_threshold,
                    config.colorful_bias,
                ),
            )
            config = self.cursor.fetchone()
            return config
        except sqlite3.Error as e:
            print(f"Error retrieving movie config: {e}")
            return None

    # Method for looking up a [Frame] AKA frame_colors with movie_id
    def get_frame_colors_by_movie_id(self, movie_id):
        """Retrieve all frame colors associated with frames of a given movie_id, ordered by frame index.

        Args:
            movie_id (str): The movie_id to filter frame colors by.

        Returns:
            list: A list of tuples, where each tuple represents a frame color.
              Returns an empty list if no frame colors are found.
        """
        try:
            self.cursor.execute(
                """
                SELECT FRAME.idx, FRAME_COLOR.proportion, 
                        FRAME_COLOR.red, FRAME_COLOR.green, FRAME_COLOR.blue,
                        FRAME_COLOR.hue, FRAME_COLOR.sat
                FROM MOVIE 
                JOIN FRAME USING (movie_id) 
                JOIN FRAME_COLOR USING (frame_id) 
                WHERE MOVIE.movie_id = ? 
                ORDER BY FRAME.idx, FRAME_COLOR.proportion DESC
                """,
                (movie_id,),
            )
            frame_colors = self.cursor.fetchall()
            return frame_colors
        except sqlite3.Error as e:
            print(f"Error retrieving frame colors by movie_id: {e}")
            return []

    @staticmethod
    def frame_color_rows_to_dicts(fc_rows):
        """
        Reconstruct frame color dictionaries from DB rows.
        Convert DB frame color rows to a one dict per frame format. (For loading from DB)

        Yields:
            dict: A dictionary containing frame color data for each frame.
        """
        # Initialize lists to hold color data
        colors, props, hues, sats = [], [], [], []
        i, n = 0, len(fc_rows)
        while i < n:
            # Process the current row
            row = fc_rows[i]
            db_idx, prop, r, g, b, hue, sat = row
            colors.append((r, g, b))
            props.append(prop)
            hues.append(hue)
            sats.append(sat)
            i += 1

            # Check if we have reached the end of the current frame
            frame_complete = i == n or fc_rows[i][0] != db_idx

            if frame_complete:
                yield {
                    "colors": np.array(colors, dtype=np.int16),
                    "proportions": np.array(props, dtype=np.float64),
                    "hues": np.array(hues, dtype=np.uint8),
                    "saturations": np.array(sats, dtype=np.uint8),
                }
                # Reset for the next frame
                colors, props, hues, sats = [], [], [], []

    @staticmethod
    def frame_dicts_to_frame_color_rows(frame_dicts):
        """
        Convert frame color dicts to DB frame color rows. (For saving to DB)

        Args:
            frame_colors (list): List of dictionaries containing frame data.

        Yields:
            list[int, list]: Frame idx and a list representing a frame color row
        """
        i, n = 0, len(frame_dicts)
        while i < n:
            # Process the current frame
            frame = frame_dicts[i]
            # Yield a row for each color in the frame (most to least prominent)
            for j in range(len(frame["colors"])):
                r, g, b = frame["colors"][j]
                prop = frame["proportions"][j]
                hue = frame["hues"][j]
                sat = frame["saturations"][j]
                row = [None, prop, r, g, b, hue, sat]
                yield [i, row]
            i += 1

    def add_scenes_and_config(
        self,
        movie_id: str,
        scenes: List[Scene],
        config: MovieConfig,
    ):
        config_id = self.add_movie_config(
            movie_id=movie_id,
            config=config,
        )

        scene_rows = [
            (
                config_id,
                s.start,
                s.end,
                s.color[0],
                s.color[1],
                s.color[2],
                s.saturation,
            )  # Convert Scene to tuple for DB insertion
            for s in scenes
        ]
        self.add_scenes(scene_rows)

        return config_id

    def add_movie_config(
        self,
        movie_id,
        config: MovieConfig,
    ):
        """Add a movie config to the MOVIE_CONFIG table.

        Args:
            movie_id (str): The movie ID.
            config (MovieConfig): The movie configuration.

        Returns:
            str: The config_id of the newly inserted config, or None if insertion failed.
        """
        config_id = DatabaseService._new_id()

        try:
            self.cursor.execute(
                """
                INSERT INTO MOVIE_CONFIG (config_id, movie_id, fps, width, height, 
                                        hue_tolerance, max_gap_frames, min_scene_frames, 
                                        scene_boundary_threshold_ms, score_threshold, colorful_bias)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    config_id,
                    movie_id,
                    config.fps,
                    config.width,
                    config.height,
                    config.hue_tolerance,
                    config.max_gap_frames,
                    config.min_scene_frames,
                    config.scene_boundary_threshold_ms,
                    config.score_threshold,
                    config.colorful_bias,
                ),
            )
            self.conn.commit()
            print(f"Movie config added successfully with config_id: {config_id}")
            return config_id
        except sqlite3.Error as e:
            self.conn.rollback()
            print(f"Error adding movie config: {e}")
            return None

    def add_scenes(self, scenes):
        """Add a list of scenes to the SCENE table.

        Args:
            scenes (list): A list of tuples, where each tuple represents a scene
                           (config_id, start, end, red, green, blue, saturation).
        """
        try:
            self.cursor.executemany(
                """
                INSERT INTO SCENE (config_id, start, end, red, green, blue, saturation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                scenes,
            )
            self.conn.commit()
            print(f"Added {len(scenes)} scenes successfully.")
        except sqlite3.Error as e:
            self.conn.rollback()
            print(f"Error adding scenes: {e}")
            raise

    def add_movie(self, name):
        """Add a movie to the MOVIE table.

        Args:
            name (str): The name of the movie.
        """
        movie_id = DatabaseService._new_id()
        try:
            self.cursor.execute(
                """
                INSERT INTO MOVIE (movie_id, name)
                VALUES (?, ?)
                """,
                (movie_id, name),
            )
            self.conn.commit()
            print(f"Movie added successfully with movie_id: {movie_id}")
            return movie_id
        except sqlite3.Error as e:
            self.conn.rollback()
            print(f"Error adding movie: {e}")
            raise

    def add_frame(self, idx, movie_id):
        """Add a frame to the FRAME table.

        Args:
            idx (int): The index of the frame.
            movie_id (str): The movie ID this frame belongs to.

        Returns:
            str: The frame_id of the newly inserted frame, or None if insertion failed.
        """
        frame_id = DatabaseService._new_id()
        try:
            self.cursor.execute(
                """
                INSERT INTO FRAME (frame_id, idx, movie_id)
                VALUES (?, ?, ?)
                """,
                (frame_id, idx, movie_id),
            )
            self.conn.commit()
            print(f"Frame added successfully with frame_id: {frame_id}")
            return frame_id
        except sqlite3.Error as e:
            self.conn.rollback()
            print(f"Error adding frame: {e}")
            return None

    def add_frames(self, frame_rows):
        """Add multiple frames to the FRAME table.

        Args:
            frame_rows (list): List of tuples, each containing (frame_id, idx, movie_id).

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            self.cursor.executemany(
                """
                INSERT INTO FRAME (frame_id, idx, movie_id)
                VALUES (?, ?, ?)
                """,
                frame_rows,
            )
            self.conn.commit()
            print(f"Added {len(frame_rows)} frames successfully.")
            return True
        except sqlite3.Error as e:
            self.conn.rollback()
            print(f"Error adding frames: {e}")
            return False

    def rows_from_frame_dicts(self, movie_id, frame_dicts):
        """Save frames from a list of frame dictionaries to the database.

        Args:
            frame_dicts (list): List of dictionaries containing frame data.

        Returns:
            tuple (list, list): frame_rows, frame_color_rows

            - frame_rows (frame_id, i, movie_id)
            - frame_color_rows [frame_id, proportion, red, green, blue, hue, sat]
        """
        frame_rows = []
        frame_color_rows = []
        to_row_converter = DatabaseService.frame_dicts_to_frame_color_rows

        # Track frames we've seen
        frame_ids = {}

        for i, frame_color_row in to_row_converter(frame_dicts):
            # Generate frame_id only once per frame index
            if i not in frame_ids:
                frame_id = DatabaseService._new_id()
                frame_ids[i] = frame_id
                # Add exactly ONE frame row per frame
                frame_rows.append((frame_id, i, movie_id))
            else:
                frame_id = frame_ids[i]

            # Set the frame_id in the color row
            frame_color_row[0] = frame_id
            frame_color_rows.append(frame_color_row)

        return frame_rows, frame_color_rows

    def get_frames_by_movie_id(self, movie_id):
        """Retrieve all frames for a specific movie, ordered by index.

        Args:
            movie_id (str): The movie ID to filter frames by.

        Returns:
            list: List of frame tuples, or empty list if none found.
        """
        try:
            self.cursor.execute(
                """
                SELECT * FROM FRAME
                WHERE movie_id = ?
                ORDER BY idx
                """,
                (movie_id,),
            )
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error retrieving frames: {e}")
            return []

    def add_frame_colors(self, color_entries):
        """Add multiple color entries for frames at once.

        Args:
            color_entries (list): List of iterables, each containing
                                 (frame_id, proportion, red, green, blue, hue, sat)

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            self.cursor.executemany(
                """
                INSERT INTO FRAME_COLOR (frame_id, proportion, red, green, blue, hue, sat)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                color_entries,
            )
            self.conn.commit()
            print(f"Added {len(color_entries)} frame colors successfully.")
            return True
        except sqlite3.Error as e:
            self.conn.rollback()
            print(f"Error adding frame colors: {e}")
            return False

    def get_colors_by_frame_id(self, frame_id):
        """Get color data for a specific frame.

        Args:
            frame_id (str): The frame ID to find colors for.

        Returns:
            list: List of color tuples for the frame, ordered by proportion (descending).
        """
        try:
            self.cursor.execute(
                """
                SELECT * FROM FRAME_COLOR
                WHERE frame_id = ?
                ORDER BY proportion DESC
                """,
                (frame_id,),
            )
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error retrieving frame colors: {e}")
            return []

    def get_config_by_id(self, config_id):
        """Get a movie config by its ID.

        Args:
            config_id (str): The config ID to find.

        Returns:
            tuple: Movie config data or None if not found.
        """
        try:
            self.cursor.execute(
                """
                SELECT * FROM MOVIE_CONFIG WHERE config_id = ?
                """,
                (config_id,),
            )
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print(f"Error retrieving movie config: {e}")
            return None

    def get_all_scenes_by_movie_name(self, name):
        """Get all scenes for each config of a movie by its name.

        Args:
            name (str): The name of the movie to find scenes for.

        Returns:
            list: List of tuples, where each tuple contains the config_id and a list of scenes.
        """
        try:
            self.cursor.execute(
                """
                SELECT SCENE.*
                FROM MOVIE
                JOIN MOVIE_CONFIG ON MOVIE.movie_id = MOVIE_CONFIG.movie_id
                JOIN SCENE ON MOVIE_CONFIG.config_id = SCENE.config_id
                WHERE MOVIE.name = ?
                """,
                (name,),
            )
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error retrieving all scenes for each config: {e}")
            return []

    def reset_scene_and_config_tables(self):
        """
        Drop and recreate the SCENE and MOVIE_CONFIG tables.
        This will delete all existing scene and config data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Disable foreign key constraints temporarily
            self.cursor.execute("PRAGMA foreign_keys = OFF")

            # Drop tables in correct order (SCENE depends on MOVIE_CONFIG)
            self.cursor.execute("DROP TABLE IF EXISTS SCENE")
            self.cursor.execute("DROP TABLE IF EXISTS MOVIE_CONFIG")

            # Recreate the tables
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS MOVIE_CONFIG (
                    config_id TEXT PRIMARY KEY,
                    movie_id TEXT NOT NULL,
                    fps INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    hue_tolerance INTEGER NOT NULL,
                    max_gap_frames INTEGER NOT NULL,
                    min_scene_frames INTEGER NOT NULL,
                    scene_boundary_threshold_ms INTEGER NOT NULL,
                    score_threshold INTEGER NOT NULL,
                    colorful_bias INTEGER NOT NULL,
                    FOREIGN KEY (movie_id) REFERENCES MOVIE(movie_id)
                )
                """
            )

            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS SCENE (
                    scene_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_id TEXT NOT NULL,
                    start INTEGER NOT NULL,
                    end INTEGER NOT NULL,
                    red INTEGER NOT NULL,
                    green INTEGER NOT NULL,
                    blue INTEGER NOT NULL,
                    saturation INTEGER DEFAULT 100,
                    FOREIGN KEY (config_id) REFERENCES MOVIE_CONFIG(config_id)
                )
                """
            )

            # Re-enable foreign key constraints
            self.cursor.execute("PRAGMA foreign_keys = ON")

            self.conn.commit()
            print("Scene and Config tables have been reset successfully")
            return True
        except sqlite3.Error as e:
            self.conn.rollback()
            print(f"Error resetting Scene and Config tables: {e}")
            return False


class MovieNotFoundError(Exception):
    """Custom exception raised when a movie is not found in the database."""

    pass


class ConfigNotFoundError(Exception):
    """Custom exception raised when a movie config is not found in the database."""

    pass
