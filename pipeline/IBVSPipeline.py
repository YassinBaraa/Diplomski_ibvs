from pipeline.IBVSContext import IBVSContext
import logging
import numpy as np

logger = logging.getLogger(__name__)


class IBVSPipeline:
    def __init__(self, source, feature_extractor, tracker=None, controller=None):
        self.source = source
        self.feature_extractor = feature_extractor
        self.tracker = tracker
        self.controller = controller

    def run(self):
        while True:
            ret, frame, point = self.source.read()
            if not ret or frame is None:
                break

            ctx = IBVSContext(frame=frame)
            if point is not None:
                ctx.point = np.asarray(point, dtype=np.float32)
            ctx.reference_frame = getattr(self.source, "reference_frame", None)
            ctx.distance_mm = getattr(self.source, "distance_mm", None)
            ctx.warmup_complete = getattr(self.source, "warmup_complete", False)

            # Step 1: extract FAST+Harris features (during warmup or on re-extraction)
            try:
                ctx.extracted_features = self.feature_extractor.extract(ctx)
            except Exception as e:
                logger.error(f"Feature extraction error: {e}", exc_info=True)
                ctx.extracted_features = np.empty((0, 2), dtype=np.float32)

            # Step 2: KLT tracking — lock on first post-warmup frame, track afterwards
            if self.tracker is not None and ctx.warmup_complete:
                try:
                    if not self.tracker._locked:
                        # Lock using the reference frame captured when final_point was set
                        ref = ctx.reference_frame if ctx.reference_frame is not None else ctx.frame
                        if ctx.extracted_features is not None and len(ctx.extracted_features) >= self.tracker.min_features:
                            self.tracker.lock(ref, ctx.extracted_features)

                    if self.tracker._locked:
                        tracked = self.tracker.update(ctx.frame, ctx.point)
                        ctx.extracted_features = tracked
                        ctx.estimated_point = tracked.mean(axis=0).astype(np.float32) if len(tracked) > 0 else None
                except Exception as e:
                    logger.error(f"Tracker error: {e}", exc_info=True)

            # Step 3: controller computes error and velocity from estimated_point
            if self.controller is not None:
                try:
                    self.controller.update_ctx(ctx)
                except Exception as e:
                    logger.error(f"Controller error: {e}", exc_info=True)
                    ctx.debug["velocity_command"] = np.zeros(2, dtype=np.float32)

            yield ctx
