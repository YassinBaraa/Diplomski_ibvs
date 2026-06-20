from pipeline.IBVSContext import IBVSContext
import logging
import numpy as np

logger = logging.getLogger(__name__)

_LOST_FRAMES_BEFORE_UNLOCK = 5  # consecutive frames with no estimated_point before forcing re-lock


class IBVSPipeline:
    def __init__(self, source, feature_extractor, tracker=None, controller=None):
        self.source = source
        self.feature_extractor = feature_extractor
        self.tracker = tracker
        self.controller = controller
        self._lost_streak = 0

    def run(self):
        frame_n = 0
        while True:
            ret, frame, point = self.source.read()
            if not ret or frame is None:
                print("[IBVSPipeline] Source exhausted — stopping")
                break

            frame_n += 1
            ctx = IBVSContext(frame=frame)
            ctx.reference_frame = getattr(self.source, "reference_frame", None)
            ctx.distance_mm = getattr(self.source, "distance_mm", None)
            ctx.warmup_complete = getattr(self.source, "warmup_complete", False)

            locked = self.tracker is not None and self.tracker._locked

            if not locked:
                print(f"[IBVSPipeline] Frame {frame_n}: DETECTION mode — "
                      f"warmup={ctx.warmup_complete}, detection_point={point}")

                if point is not None:
                    ctx.point = np.asarray(point, dtype=np.float32)

                try:
                    ctx.extracted_features = self.feature_extractor.extract(ctx)
                    print(f"[IBVSPipeline] Frame {frame_n}: extracted {len(ctx.extracted_features)} features")
                except Exception as e:
                    logger.error(f"Feature extraction error: {e}", exc_info=True)
                    ctx.extracted_features = np.empty((0, 2), dtype=np.float32)

                if self.tracker is not None and ctx.warmup_complete and ctx.point is not None:
                    ref = ctx.reference_frame if ctx.reference_frame is not None else ctx.frame
                    lock_ctx = IBVSContext(frame=ref)
                    lock_ctx.point = ctx.point
                    lock_ctx.warmup_complete = True
                    try:
                        lock_features = self.feature_extractor.extract(lock_ctx)
                        print(f"[IBVSPipeline] Frame {frame_n}: lock candidate — "
                              f"{len(lock_features)} features around point {ctx.point}")
                    except Exception as e:
                        logger.error(f"Lock feature extraction error: {e}", exc_info=True)
                        lock_features = np.empty((0, 2), dtype=np.float32)
                    if lock_features is not None and len(lock_features) >= self.tracker.min_features:
                        self.tracker.lock(ref, lock_features, ctx.point)
                        self._lost_streak = 0
                        print(f"[IBVSPipeline] Frame {frame_n}: LOCKED — switching to KLT tracking")
                    else:
                        print(f"[IBVSPipeline] Frame {frame_n}: not enough features to lock "
                              f"({len(lock_features)} < {self.tracker.min_features}), waiting...")
            else:
                try:
                    tracked, ctx.estimated_point = self.tracker.update(ctx.frame)
                    ctx.extracted_features = tracked
                    n_tracked = len(tracked) if tracked is not None else 0

                    if ctx.estimated_point is not None:
                        self._lost_streak = 0
                        print(f"[IBVSPipeline] Frame {frame_n}: KLT tracking — "
                              f"{n_tracked} features, estimated_point={ctx.estimated_point}")
                    else:
                        self._lost_streak += 1
                        print(f"[IBVSPipeline] Frame {frame_n}: KLT lost point "
                              f"(streak={self._lost_streak}/{_LOST_FRAMES_BEFORE_UNLOCK}), "
                              f"tracked={n_tracked}")
                        if self._lost_streak >= _LOST_FRAMES_BEFORE_UNLOCK:
                            print(f"[IBVSPipeline] Frame {frame_n}: LOST STREAK reached — "
                                  f"unlocking tracker, scanning for new final point")
                            self.tracker.unlock()
                            self._lost_streak = 0

                except Exception as e:
                    logger.error(f"Tracker error: {e}", exc_info=True)
                    self._lost_streak += 1
                    if self._lost_streak >= _LOST_FRAMES_BEFORE_UNLOCK:
                        print(f"[IBVSPipeline] Frame {frame_n}: tracker error streak — unlocking")
                        self.tracker.unlock()
                        self._lost_streak = 0

            if self.controller is not None:
                try:
                    self.controller.update_ctx(ctx)
                    ctrl = ctx.debug.get("controller", {})
                    print(f"[IBVSPipeline] Frame {frame_n}: controller — "
                          f"source={ctrl.get('point_source','none')}, "
                          f"error={ctx.debug.get('control_error_px')}, "
                          f"vel={ctx.debug.get('velocity_command')}")
                except Exception as e:
                    logger.error(f"Controller error: {e}", exc_info=True)
                    ctx.debug["velocity_command"] = np.zeros(2, dtype=np.float32)

            yield ctx
