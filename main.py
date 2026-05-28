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
from controller.PointController import PointController
from pipeline.IBVSPipeline import IBVSPipeline
from config import Config

import cv2


def get_detection_pipeline_iterator():

    detection_pipeline_path = Path(__file__).parent.parent / "detection_pipeline"
    detection_pipeline_path_str = str(detection_pipeline_path)
    if detection_pipeline_path_str in sys.path:
        sys.path.remove(detection_pipeline_path_str)
    sys.path.insert(0, detection_pipeline_path_str)

    # Temporarily remove ibvs root to avoid package shadowing during detection imports.
    removed_project_root = False
    if project_root_str in sys.path:
        sys.path.remove(project_root_str)
        removed_project_root = True

    for module_name in list(sys.modules.keys()):
        if (
            module_name == "pipeline"
            or module_name.startswith("pipeline.")
            or module_name == "sources"
            or module_name.startswith("sources.")
            or module_name == "config"
            or module_name.startswith("config.")
        ):
            del sys.modules[module_name]
    
    try:
        from sources.MP4Source import MP4Source as DP_MP4Source
        from detectors.YOLOBranchSeg import YOLOBranchSeg
        from trackers.ByteTrack import ByteTrack
        from postprocessing.PostProcessor import PostProcessor
        from postprocessing.masks.MaskExtraction import MaskExtraction
        from postprocessing.geometry.DistanceHeatmap import DistanceHeatmap
        from postprocessing.geometry.BitmaskSkeleton import BitmaskSkeleton
        from postprocessing.scoring.CandidateScoring import CandidateScoring
        from postprocessing.scoring.CandidateVisualizer import CandidateVisualizer
        from pipeline.DetectionPipeline import DetectionPipeline
        from config import ConfigManager
    finally:
        if removed_project_root:
            sys.path.insert(0, project_root_str)
    
    # Load detection pipeline config
    config_dp = ConfigManager()
    
    # Initialize detection pipeline components
    source_dp = DP_MP4Source(config_dp.get("source.video_path"))
    detector = YOLOBranchSeg(
        model_path=config_dp.get("detector.model_path"),
        conf=config_dp.get("detector.confidence_threshold")
    )
    tracker = ByteTrack(detector)
    postprocessor = PostProcessor([
        MaskExtraction(),
        DistanceHeatmap(),
        BitmaskSkeleton(),
        CandidateScoring(),
        CandidateVisualizer(),
    ])
    
    # Create and run detection pipeline
    pipeline_dp = DetectionPipeline(
        source=source_dp,
        detector=detector,
        tracker=tracker,
        postprocessor=postprocessor
    )
    
    return pipeline_dp.run()


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

    controller = PointController(
        gain=config.get("controller.main_gain", 0.5),
        validation_enabled=config.get("controller.validation_enabled", True),
        max_feature_motion_px=config.get("controller.max_feature_motion_px", 50.0),
        fallback_enabled=config.get("controller.fallback_enabled", True),
        fallback_gain_scale=config.get("controller.fallback_gain_scale", 0.5),
    )

    pipeline = IBVSPipeline(
        source=source,
        feature_extractor=feature_extractor,
        controller=controller
    
    )

    frame_count = 0
    for ctx in pipeline.run():
        frame_count += 1
        velocity = ctx.debug.get("velocity_command")
        error = ctx.debug.get("control_error_px")
        point_source = ctx.debug.get("controller", {}).get("point_source", "none")
        print(
            f"Frame {frame_count}: Extracted {len(ctx.extracted_features)} features, "
            f"Point: {ctx.point}, Source: {point_source}, Error: {error}, cmd_vel: {velocity}"
        )

        # Visualization: draw extracted feature centers on the frame and display
        try:
            for (x, y) in ctx.extracted_features:
                cv2.circle(ctx.frame, (int(x), int(y)), 4, (0, 255, 0), -1)
            
            # Draw point if available
            if ctx.point is not None:
                x, y = ctx.point
                cv2.circle(ctx.frame, (int(x), int(y)), 6, (0, 0, 255), 2)
            
            cv2.imshow('IBVS - Features & Target Point', ctx.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception:
            pass

    # Cleanup
    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
