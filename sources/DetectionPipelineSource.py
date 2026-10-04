import numpy as np
from sources.FrameSource import FrameSource


class DetectionPipelineSource(FrameSource):

    def __init__(self, detection_dict_iterator):
        self._iterator = iter(detection_dict_iterator)

    def read(self):
        try:
            data = next(self._iterator)
        except StopIteration:
            return False, None, None

        frame = data.get("frame")
        if frame is None:
            return False, None, None

        best_candidate = data.get("best_candidate")

        if isinstance(best_candidate, dict):
            x, y = best_candidate.get("x"), best_candidate.get("y")
            point = np.array([round(x), round(y)], dtype=int) if x is not None and y is not None else None
            print(f"[DetectionPipelineSource] best_candidate={point}")
        else:
            point = None
            print(f"[DetectionPipelineSource] no point this frame")

        return True, frame, point

    def release(self):
        pass
