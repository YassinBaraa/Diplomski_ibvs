from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING
import numpy as np

@dataclass
class IBVSContext:
    frame: np.ndarray
    point: Optional[np.ndarray] = None
    extracted_features: Optional[np.ndarray] = None  # FAST+Harris keypoints [N,2]
    estimated_point: Optional[np.ndarray] = None      # KLT-tracked perch point this frame (point = features + their stored offsets)
    target_size: Optional[float] = None               # apparent size [px] of the target at estimated_point, measured on this frame (None = not measured)
    t_frame: Optional[float] = None                   # time.monotonic() right after the camera grabbed this frame (None = source has no grab time)
    debug: dict[str, Any] = field(default_factory=dict)
