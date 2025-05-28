from dataclasses import dataclass


@dataclass
class MovieConfig:
    id: str | None
    fps: int
    width: int
    height: int
    hue_tolerance: int
    max_gap_frames: int
    min_scene_frames: int
    scene_boundary_threshold_ms: int
    score_threshold: int
    colorful_bias: int

    @classmethod
    def from_dict(cls, data: dict) -> "MovieConfig":
        """Use this when the frontend sends a creation request."""
        return cls(
            id=data.get("id"),
            fps=data.get("fps"),
            width=data.get("width"),
            height=data.get("height"),
            hue_tolerance=data.get("hue_tolerance"),
            max_gap_frames=data.get("max_gap_frames"),
            min_scene_frames=data.get("min_scene_frames"),
            scene_boundary_threshold_ms=data.get("scene_boundary_threshold_ms"),
            score_threshold=data.get("score_threshold"),
            colorful_bias=data.get("colorful_bias"),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "hue_tolerance": self.hue_tolerance,
            "max_gap_frames": self.max_gap_frames,
            "min_scene_frames": self.min_scene_frames,
            "scene_boundary_threshold_ms": self.scene_boundary_threshold_ms,
            "score_threshold": self.score_threshold,
            "colorful_bias": self.colorful_bias,
        }

    @classmethod
    def from_db_tuple(cls, db_tuple: tuple) -> "MovieConfig":
        return cls(
            id=db_tuple[0],
            fps=db_tuple[2],
            width=db_tuple[3],
            height=db_tuple[4],
            hue_tolerance=db_tuple[5],
            max_gap_frames=db_tuple[6],
            min_scene_frames=db_tuple[7],
            scene_boundary_threshold_ms=db_tuple[8],
            score_threshold=db_tuple[9],
            colorful_bias=db_tuple[10],
        )
