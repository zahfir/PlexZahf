# Home assistant lighting percentage
BRIGHTNESS = 70

# How often to poll the current position for scene/light changes
PLAYBACK_POLL_INTERVAL_SEC = 0.2

# Buffer zone before end of scene to preemptively change lights
SCENE_BOUNDARY_THRESHOLD_MS = 2000

# KMeans clustering settings
COLORS_PER_FRAME = 5
PIXEL_SAMPLE_SIZE = 1000

# FFMPEG transcode settings (these are hardcoded in Go module)
WIDTH = 426
HEIGHT = 240
FPS = 5


# Lighting instruction settings
HUE_TOLERANCE = 1
MIN_SCENE_FRAMES = 20
MAX_GAP_FRAMES = 20
SCORE_THRESHOLD = 1200

# Exponential bias for more colorful scenes
# Penalizes white/black scenes and frames
# 1 = no penalty
COLORFUL_BIAS = 25
