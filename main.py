import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
project_root_str = str(project_root)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from sources.MP4Source import MP4Source
from sources.DetectionPipelineSource import DetectionPipelineSource
from feature_extraction.FASTHarrisExtractor import FASTHarrisExtractor
from trackers.KLTTracker import KLTTracker
from controller.PointController import PointController
from pipeline.IBVSPipeline import IBVSPipeline
from config import Config

import cv2
import numpy as np


def get_detection_pipeline_iterator():
    detection_pipeline_path = str(Path(__file__).parent.parent / "detection_pipeline")

    # Remove ibvs root so detection pipeline's own packages (pipeline, sources, config) resolve correctly
    if project_root_str in sys.path:
        sys.path.remove(project_root_str)

    if detection_pipeline_path not in sys.path:
        sys.path.insert(0, detection_pipeline_path)

    # Clear shadowed ibvs modules so detection pipeline re-imports from its own path
    for module_name in list(sys.modules.keys()):
        if (
            module_name == "pipeline" or module_name.startswith("pipeline.")
            or module_name == "sources" or module_name.startswith("sources.")
            or module_name == "config" or module_name.startswith("config.")
            or module_name == "postprocessing" or module_name.startswith("postprocessing.")
            or module_name == "main"
        ):
            del sys.modules[module_name]

    try:
        import main as dp_main
        iterator = dp_main.main()
    finally:
        # Restore ibvs root so the rest of ibvs imports still work
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)

    return iterator


def main():
    
    try:
        config = Config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Choose source based on config
    source_type = config.get("source.type", "mp4")
    
    if source_type == "detection":
        detection_iterator = get_detection_pipeline_iterator()
        source = DetectionPipelineSource(detection_iterator)
    else:
        source = MP4Source(config.get("source.video_path"))

    feature_extractor = FASTHarrisExtractor(
        max_features=config.get("feature_extraction.max_features"),
        fast_threshold=config.get("feature_extraction.fast_threshold"),
        harris_block_size=config.get("feature_extraction.harris_block_size"),
        harris_ksize=config.get("feature_extraction.harris_ksize"),
        harris_k=config.get("feature_extraction.harris_k"),
        point_focus_radius=config.get("feature_extraction.point_focus_radius"),
    )

    tracker = KLTTracker(
        feature_extractor=feature_extractor,
        min_features=config.get("controller.min_features", 8),
    )

    controller = PointController(
        gain=config.get("controller.main_gain", 0.5),
    )

    pipeline = IBVSPipeline(
        source=source,
        feature_extractor=feature_extractor,
        tracker=tracker,
        controller=controller,
    )

    for frame_count, ctx in enumerate(pipeline.run(), 1):
        ctrl = ctx.debug.get("controller", {})
        velocity = ctx.debug.get("velocity_command")
        error = ctx.debug.get("control_error_px")
        print(
            f"Frame {frame_count}: tracked={ctrl.get('n_tracked', 0)}, "
            f"source={ctrl.get('point_source', 'none')}, error={error}, vel={velocity}"
        )

        vis = ctx.frame.copy()
        h, w = vis.shape[:2]
        center = (w // 2, h // 2)

        # Draw tracked features as green dots
        if ctx.extracted_features is not None:
            for (x, y) in ctx.extracted_features:
                cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)

        # Draw final_point anchor (blue) and KLT-estimated branch position (red)
        if ctx.point is not None:
            cv2.circle(vis, (int(ctx.point[0]), int(ctx.point[1])), 7, (255, 0, 0), 2)
        if ctx.estimated_point is not None:
            cv2.circle(vis, (int(ctx.estimated_point[0]), int(ctx.estimated_point[1])), 7, (0, 0, 255), 2)

        # Draw frame center crosshair
        cv2.drawMarker(vis, center, (200, 200, 200), cv2.MARKER_CROSS, 20, 1)

        # Draw velocity arrow from frame center — direction and length show where/how hard to move
        if velocity is not None and np.linalg.norm(velocity) > 0.5:
            arrow_scale = 2.0
            tip = (
                int(center[0] + velocity[0] * arrow_scale),
                int(center[1] + velocity[1] * arrow_scale),
            )
            cv2.arrowedLine(vis, center, tip, (0, 165, 255), 2, tipLength=0.3)

        cv2.imshow('IBVS', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        yield {
            "error_x": float(error[0]) if error is not None else None,
            "error_y": float(error[1]) if error is not None else None,
            "distance_mm": ctx.distance_mm,
        }

    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
