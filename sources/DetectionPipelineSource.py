import numpy as np
from sources.FrameSource import FrameSource


class DetectionPipelineSource(FrameSource):

    def __init__(self, detection_dict_iterator):
        self._iterator = iter(detection_dict_iterator)
        self.reference_frame = None
        self.distance_mm = None
        self.warmup_complete = False

    def read(self):
        try:
            data = next(self._iterator)
        except StopIteration:
            return False, None, None

        frame = data.get("frame")
        if frame is None:
            return False, None, None

        self.reference_frame = data.get("reference_frame")
        self.distance_mm = data.get("distance_mm")

        final_point = data.get("final_point")
        best_candidate = data.get("best_candidate")

        if final_point is not None:
            self.warmup_complete = True
            point = np.array([final_point["x"], final_point["y"]], dtype=np.float32)
        elif isinstance(best_candidate, dict):
            x, y = best_candidate.get("x"), best_candidate.get("y")
            point = np.array([x, y], dtype=np.float32) if x is not None and y is not None else None
        else:
            point = None

        return True, frame, point

    def release(self):
        pass
