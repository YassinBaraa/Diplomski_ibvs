import numpy as np
from sources.FrameSource import FrameSource


class DetectionPipelineSource(FrameSource):

    def __init__(self, detection_dict_iterator):
        self._iterator = iter(detection_dict_iterator)
        self.reference_frame = None
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

        final_point = data.get("final_point")
        best_candidate = data.get("best_candidate")

        if final_point is not None:
            self.warmup_complete = True
            point = np.array([round(final_point["x"]), round(final_point["y"])], dtype=int)
            print(f"[DetectionPipelineSource] final_point={point}")
        elif isinstance(best_candidate, dict):
            x, y = best_candidate.get("x"), best_candidate.get("y")
            point = np.array([round(x), round(y)], dtype=int) if x is not None and y is not None else None
            print(f"[DetectionPipelineSource] no final_point, best_candidate={point}")
        else:
            point = None
            print(f"[DetectionPipelineSource] no point this frame")

        return True, frame, point

    def release(self):
        pass
