# Buffer zone before end of scene to preemptively change lights
SCENE_BOUNDARY_THRESHOLD_MS = 2400
# How often to poll the current position for scene/light changes
PLAYBACK_POLL_INTERVAL_SEC = 0.2

# How long to seek between frame readings
VIDEO_READ_FREQ_MS = 500
# Minimum frames to seek after scenedetect finds a cut
MIN_FRAMES_AFTER_CUT = 35

# Merge short and similar scenes
MIN_MERGE_SCENE_DURATION_MS = 15000
MERGE_SCENE_COLOR_THRESHOLD = 0.15

# Size of each file download/write to RAM
CHUNK_SIZE_BYTES = 1073741824

# FFMPEG transcode settings
WIDTH = 80
HEIGHT = 60
FPS = 5

# Home assistant lighting percentage
BRIGHTNESS = 100

# Lighting instruction settings
HUE_TOLERANCE = 5
MIN_SCENE_FRAMES = 35
MAX_GAP_FRAMES = 10
SCORE_THRESHOLD = 50
