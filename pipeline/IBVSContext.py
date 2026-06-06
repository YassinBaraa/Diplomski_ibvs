from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING
import numpy as np

@dataclass
class IBVSContext:
    frame: np.ndarray
    point: Optional[np.ndarray] = None
    extracted_features: Optional[np.ndarray] = None  # FAST+Harris keypoints [N,2]
    reference_frame: Optional[np.ndarray] = None      # frame snapshot when final_point was locked
    distance_mm: Optional[float] = None
    warmup_complete: bool = False
    estimated_point: Optional[np.ndarray] = None      # KLT-estimated branch position this frame
    debug: dict[str, Any] = field(default_factory=dict)
