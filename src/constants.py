# Buffer zone before end of scene to preemptively change lights
SCENE_BOUNDARY_THRESHOLD_MS = 2000
# How often to poll the current position for scene/light changes
PLAYBACK_POLL_INTERVAL_SEC = 0.5

# How long to seek between frame readings
VIDEO_READ_FREQ_MS = 500
# Minimum frames to seek after scenedetect finds a cut
MIN_FRAMES_AFTER_CUT = 125

# Merge short and similar scenes
MIN_MERGE_SCENE_DURATION_MS = 10000
MERGE_SCENE_COLOR_THRESHOLD = 0.20
