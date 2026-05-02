from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING
import numpy as np

@dataclass
class IBVSContext:
    frame: np.ndarray
    point: Optional[np.ndarray] = None
    extracted_features: Optional[np.ndarray] = None
    debug: dict[str, Any] = field(default_factory=dict)
