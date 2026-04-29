from pipeline.IBVSContext import IBVSContext
import logging
import numpy as np

logger = logging.getLogger(__name__)

class IBVSPipeline:
    def __init__(self, source, feature_extractor):
        self.source = source
        self.feature_extractor = feature_extractor

    def _read_source_frame(self):
        read_output = self.source.read()
        if not isinstance(read_output, tuple):
            raise TypeError("source.read() must return a tuple")

        if len(read_output) == 3:
            ret, frame, point = read_output
        elif len(read_output) == 2:
            ret, frame = read_output
            point = None
        else:
            raise ValueError("source.read() must return (ret, frame) or (ret, frame, point)")

        return ret, frame, point

    def run(self):
        
        while True:
            ret, frame, point = self._read_source_frame()
            if not ret or frame is None:
                break

            ctx = IBVSContext(frame=frame)
            if point is not None:
                ctx.point = np.asarray(point, dtype=np.float32)

            try:
                ctx.extracted_features = self.feature_extractor.extract(ctx)
            except Exception as e:
                logger.error(f"Pipeline error processing frame: {e}", exc_info=True)
                ctx.extracted_features = np.empty((0, 2), dtype=np.float32)
                ctx.debug["feature_extraction_error"] = str(e)

            yield ctx
