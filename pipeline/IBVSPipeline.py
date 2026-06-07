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
            ctx.reference_frame = getattr(self.source, "reference_frame", None)
            ctx.distance_mm = getattr(self.source, "distance_mm", None)
            ctx.warmup_complete = getattr(self.source, "warmup_complete", False)

            locked = self.tracker is not None and self.tracker._locked

            if not locked:
                # Pre-lock: use detection point and extract features for visualization/locking
                if point is not None:
                    ctx.point = np.asarray(point, dtype=np.float32)

                try:
                    ctx.extracted_features = self.feature_extractor.extract(ctx)
                except Exception as e:
                    logger.error(f"Feature extraction error: {e}", exc_info=True)
                    ctx.extracted_features = np.empty((0, 2), dtype=np.float32)

                # Lock on the first post-warmup frame.
                # Extract features from the reference frame so _prev_gray and _tracked_pts
                # are consistent — KLT will then correctly track ref→current frame.
                if self.tracker is not None and ctx.warmup_complete and ctx.point is not None:
                    ref = ctx.reference_frame if ctx.reference_frame is not None else ctx.frame
                    lock_ctx = IBVSContext(frame=ref)
                    lock_ctx.point = ctx.point
                    lock_ctx.warmup_complete = True
                    try:
                        lock_features = self.feature_extractor.extract(lock_ctx)
                    except Exception as e:
                        logger.error(f"Lock feature extraction error: {e}", exc_info=True)
                        lock_features = np.empty((0, 2), dtype=np.float32)
                    if lock_features is not None and len(lock_features) >= self.tracker.min_features:
                        self.tracker.lock(ref, lock_features, ctx.point)
                        logger.info("KLT tracker locked — switching to feature-only tracking")
            else:
                # Post-lock: tracker is sole authority, detection output is ignored entirely
                try:
                    tracked, ctx.estimated_point = self.tracker.update(ctx.frame)
                    ctx.extracted_features = tracked
                except Exception as e:
                    logger.error(f"Tracker error: {e}", exc_info=True)

            # Controller computes error from estimated_point only (no detection fallback post-lock)
            if self.controller is not None:
                try:
                    self.controller.update_ctx(ctx)
                except Exception as e:
                    logger.error(f"Controller error: {e}", exc_info=True)
                    ctx.debug["velocity_command"] = np.zeros(2, dtype=np.float32)

            yield ctx
