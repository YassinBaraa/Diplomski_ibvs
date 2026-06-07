import sys
import cv2
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DETECTION_PIPELINE = str(_HERE.parent / "detection_pipeline")

sys.path.insert(0, str(_HERE))

from feature_extraction.FASTHarrisExtractor import FASTHarrisExtractor
from trackers.KLTTracker import KLTTracker
from controller.PointController import PointController
from pipeline.IBVSPipeline import IBVSPipeline
from pipeline.IBVSContext import IBVSContext

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUT_DIR   = _HERE / "detection_test"
VIS_W     = 640
VIS_H     = 384
FPS          = 12
NICLA_FPS    = 10

FAST_THRESHOLD   = 20
MAX_FEATURES     = 80
POINT_FOCUS_RAD  = 80
CONTROLLER_GAIN  = 0.5
MIN_FEATURES     = 8


def make_output_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUT_DIR}")


def add_label(frame, text):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    cv2.putText(frame, text, (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, ts, (frame.shape[1] - 145, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return frame


def create_writer(path, w, h, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(str(path), fourcc, fps, (w, h))


# --- Per-stage visualizers ---

def vis_feature_extraction(ctx):
    h = ctx.frame.shape[0] if ctx.frame is not None else VIS_H
    w = ctx.frame.shape[1] if ctx.frame is not None else VIS_W
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    if ctx.extracted_features is not None and len(ctx.extracted_features) > 0:
        for (x, y) in ctx.extracted_features:
            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
    return vis


def vis_klt_tracking(ctx):
    vis = ctx.frame.copy() if ctx.frame is not None else np.zeros((VIS_H, VIS_W, 3), dtype=np.uint8)
    if ctx.extracted_features is not None and len(ctx.extracted_features) > 0:
        for (x, y) in ctx.extracted_features:
            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
    if ctx.estimated_point is not None:
        ep = (int(ctx.estimated_point[0]), int(ctx.estimated_point[1]))
        cv2.circle(vis, ep, 8, (0, 0, 255), 2)
    if ctx.point is not None:
        ap = (int(ctx.point[0]), int(ctx.point[1]))
        cv2.circle(vis, ap, 7, (255, 0, 0), 2)
    return vis


def vis_controller(ctx):
    h = ctx.frame.shape[0] if ctx.frame is not None else VIS_H
    w = ctx.frame.shape[1] if ctx.frame is not None else VIS_W
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    center = (w // 2, h // 2)

    cv2.drawMarker(vis, center, (200, 200, 200), cv2.MARKER_CROSS, 30, 1)

    error = ctx.debug.get("control_error_px")

    if error is not None:
        ex, ey = float(error[0]), float(error[1])
        target = (int(center[0] + ex), int(center[1] + ey))
        cv2.arrowedLine(vis, center, target, (0, 255, 255), 2, tipLength=0.2)
        cv2.putText(vis, f"err: ({ex:.1f}, {ey:.1f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return vis


def vis_full_pipeline(ctx):
    vis = ctx.frame.copy() if ctx.frame is not None else np.zeros((VIS_H, VIS_W, 3), dtype=np.uint8)
    h, w = vis.shape[:2]
    center = (w // 2, h // 2)

    if ctx.extracted_features is not None:
        for (x, y) in ctx.extracted_features:
            cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)

    if ctx.point is not None:
        cv2.circle(vis, (int(ctx.point[0]), int(ctx.point[1])), 7, (255, 0, 0), 2)

    if ctx.estimated_point is not None:
        cv2.circle(vis, (int(ctx.estimated_point[0]), int(ctx.estimated_point[1])), 7, (0, 0, 255), 2)

    cv2.drawMarker(vis, center, (200, 200, 200), cv2.MARKER_CROSS, 20, 1)

    velocity = ctx.debug.get("velocity_command")
    if velocity is not None and np.linalg.norm(velocity) > 0.5:
        tip = (
            int(center[0] + velocity[0] * 2.0),
            int(center[1] + velocity[1] * 2.0),
        )
        cv2.arrowedLine(vis, center, tip, (0, 165, 255), 2, tipLength=0.3)

    return vis


# --- Stage runners ---

def _make_components():
    extractor = FASTHarrisExtractor(
        max_features=MAX_FEATURES,
        fast_threshold=FAST_THRESHOLD,
        point_focus_radius=POINT_FOCUS_RAD,
    )
    tracker = KLTTracker(feature_extractor=extractor, min_features=MIN_FEATURES)
    controller = PointController(gain=CONTROLLER_GAIN)
    return extractor, tracker, controller


def run_mp4_stage(idx, name, vis_fn, pipeline):
    out_path = OUT_DIR / f"ibvs_stage_{idx}_{name}.mp4"
    writer = create_writer(out_path, VIS_W, VIS_H, FPS)
    frame_count = 0
    try:
        for ctx in pipeline.run():
            vis = vis_fn(ctx)
            vis = cv2.resize(vis, (VIS_W, VIS_H))
            add_label(vis, f"Stage {idx}: {name}")
            writer.write(vis)
            frame_count += 1

    finally:
        writer.release()
    logger.info(f"Stage {idx} ({name}): {frame_count} frames → {out_path}")


def _make_detection_source():
    """Build a fresh DetectionPipelineSource backed by YOLO + MP4."""
    # Import ibvs wrapper class while ibvs path is still active.
    from sources.DetectionPipelineSource import DetectionPipelineSource

    # Remove ibvs path so detection pipeline's 'pipeline', 'sources', etc. take priority.
    while str(_HERE) in sys.path:
        sys.path.remove(str(_HERE))

    for mod in list(sys.modules.keys()):
        if any(mod == n or mod.startswith(n + ".") for n in
               ("pipeline", "sources", "config", "postprocessing", "main", "detectors",
                "trackers", "feature_extraction", "controller")):
            del sys.modules[mod]

    if _DETECTION_PIPELINE not in sys.path:
        sys.path.insert(0, _DETECTION_PIPELINE)

    try:
        import main as dp_main
        detection_iterator = dp_main.main()
    finally:
        if _DETECTION_PIPELINE in sys.path:
            sys.path.remove(_DETECTION_PIPELINE)
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))

    return DetectionPipelineSource(detection_iterator)


def run_stage_1():
    extractor, tracker, controller = _make_components()
    source = _make_detection_source()
    pipeline = IBVSPipeline(source=source, feature_extractor=extractor, tracker=None, controller=None)
    run_mp4_stage(1, "FeatureExtraction", vis_feature_extraction, pipeline)
    source.release()


def run_stage_2():
    extractor, tracker, controller = _make_components()
    source = _make_detection_source()
    pipeline = IBVSPipeline(source=source, feature_extractor=extractor, tracker=tracker, controller=None)
    run_mp4_stage(2, "KLTTracking", vis_klt_tracking, pipeline)
    source.release()


def run_stage_3():
    extractor, tracker, controller = _make_components()
    source = _make_detection_source()
    pipeline = IBVSPipeline(source=source, feature_extractor=extractor, tracker=tracker, controller=controller)
    run_mp4_stage(3, "Controller", vis_controller, pipeline)
    source.release()


def run_stage_4():
    extractor, tracker, controller = _make_components()
    source = _make_detection_source()
    pipeline = IBVSPipeline(source=source, feature_extractor=extractor, tracker=tracker, controller=controller)
    run_mp4_stage(4, "FullPipeline", vis_full_pipeline, pipeline)
    source.release()


def run_detection_pipeline_phase():
    out_path = OUT_DIR / "ibvs_detection_pipeline_full.mp4"
    extractor, tracker, controller = _make_components()
    source = _make_detection_source()
    pipeline = IBVSPipeline(source=source, feature_extractor=extractor, tracker=tracker, controller=controller)

    writer = None
    frame_count = 0
    try:
        for ctx in pipeline.run():
            vis = vis_full_pipeline(ctx)
            add_label(vis, "IBVS + Detection Pipeline (YOLO)")
            if writer is None:
                h, w = vis.shape[:2]
                writer = create_writer(out_path, w, h, FPS)
            writer.write(vis)
            frame_count += 1
    finally:
        if writer is not None:
            writer.release()
        source.release()
    logger.info(f"Detection pipeline phase: {frame_count} frames → {out_path}")


def main():
    make_output_dir()

    logger.info("--- Stage 1: Feature Extraction ---")
    run_stage_1()

    logger.info("--- Stage 2: KLT Tracking ---")
    run_stage_2()

    logger.info("--- Stage 3: Controller ---")
    run_stage_3()

    logger.info("--- Stage 4: Full Pipeline ---")
    run_stage_4()

    logger.info("--- Phase 2: Full IBVS + Detection Pipeline on Nicla (Ctrl+C to stop) ---")
    run_detection_pipeline_phase()

    logger.info("All recordings complete.")


if __name__ == "__main__":
    main()
