# Buffer zone before end of scene to preemptively change lights
SCENE_BOUNDARY_THRESHOLD_MS = 2000
# How often to poll the current position for scene/light changes
PLAYBACK_POLL_INTERVAL_SEC = 0.2

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
