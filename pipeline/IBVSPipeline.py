from pipeline.IBVSContext import IBVSContext
import logging
import numpy as np

logger = logging.getLogger(__name__)

class IBVSPipeline:
    def __init__(self, source, feature_extractor, controller=None):
        self.source = source
        self.feature_extractor = feature_extractor
        self.controller = controller


    def run(self):
        
        while True:
            ret, frame, point = self.source.read()
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

            if self.controller is not None:
                try:
                    self.controller.update_ctx(ctx)
                except Exception as e:
                    logger.error(f"Controller error processing frame: {e}", exc_info=True)
                    ctx.debug["controller_error"] = str(e)
                    ctx.debug["velocity_command"] = np.array([0.0, 0.0], dtype=np.float32)

            yield ctx
