import numpy as np
from .FrameSource import FrameSource


class DetectionPipelineSource(FrameSource):

    def __init__(self, detection_ctx_iterator):
        self._iterator = iter(detection_ctx_iterator)

    def read(self):
        try:
            ctx = next(self._iterator)
        except StopIteration:
            return False, None, None

        frame = ctx.frame
        if frame is None:
            return False, None, None

        point = None
        best_candidate = ctx.best_candidate
        if isinstance(best_candidate, dict):
            x = best_candidate.get("x")
            y = best_candidate.get("y")
            if x is not None and y is not None:
                point = np.asarray([x, y], dtype=np.float32)

        return True, frame, point

    def release(self):
        return None
