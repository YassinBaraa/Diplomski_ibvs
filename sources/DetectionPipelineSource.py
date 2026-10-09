import numpy as np
from sources.FrameSource import FrameSource

# A cross-section this close to the image edge may be clipped: its width is left out.
BORDER_MARGIN_PX = 3


def _branch_width(candidate, frame_shape):
    """(width, reason): apparent size [px] of the branch = its WIDTH at the perch point, not
    its length (which the image edges cut off): twice the distance from the point, which sits
    on the skeleton, to the mask edge. reason says why it must not be sent (unknown, or that
    cross-section reaches within BORDER_MARGIN_PX of the image edge), else None."""
    half = candidate.get("distance_to_boundary") or 0.0
    if not half > 0:
        return None, "no width at the perch point (not on the skeleton)"
    h, w = frame_shape[:2]
    x, y = candidate["x"], candidate["y"]
    m = BORDER_MARGIN_PX
    if x - half < m or y - half < m or x + half > w - 1 - m or y + half > h - 1 - m:
        return 2.0 * float(half), f"cross-section within {m}px of the image edge"
    return 2.0 * float(half), None


class DetectionPipelineSource(FrameSource):

    def __init__(self, detection_dict_iterator):
        self._iterator = iter(detection_dict_iterator)

    def read(self):
        try:
            data = next(self._iterator)
        except StopIteration:
            return False, None, None, {}

        frame = data.get("frame")
        if frame is None:
            return False, None, None, {}

        best_candidate = data.get("best_candidate")
        info = {"t_frame": data.get("t_frame"), "size": None, "size_raw": None, "omit": "no branch candidate"}

        if isinstance(best_candidate, dict):
            x, y = best_candidate.get("x"), best_candidate.get("y")
            point = np.array([round(x), round(y)], dtype=int) if x is not None and y is not None else None
            if point is not None:
                width, omit = _branch_width(best_candidate, frame.shape)
                info.update(size=None if omit else width, size_raw=width, omit=omit)
            print(f"[DetectionPipelineSource] best_candidate={point} width={info['size_raw']}")
        else:
            point = None
            print(f"[DetectionPipelineSource] no point this frame")

        return True, frame, point, info

    def release(self):
        pass
