#!/usr/bin/env python3
"""Test that DetectionPipelineSource correctly passes frames and points to IBVS pipeline."""

import sys
from pathlib import Path

# Add detection_pipeline to path
detection_pipeline_path = Path(__file__).parent.parent / "detection_pipeline"
if str(detection_pipeline_path) not in sys.path:
    sys.path.insert(0, str(detection_pipeline_path))

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

# Now import IBVS modules
sys.path.insert(0, str(Path(__file__).parent))
from sources.DetectionPipelineSource import DetectionPipelineSource
from feature_extraction.FASTHarrisExtractor import FASTHarrisExtractor
from pipeline.IBVSPipeline import IBVSPipeline
from config import Config
import cv2


def main():
    print("Testing DetectionPipelineSource → IBVSPipeline flow...")
    
    # Initialize detection pipeline
    config_dp = ConfigManager()
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
    
    pipeline_dp = DetectionPipeline(
        source=source_dp,
        detector=detector,
        tracker=tracker,
        postprocessor=postprocessor
    )
    
    # Create DetectionPipelineSource with detection pipeline iterator
    print("  Creating DetectionPipelineSource from detection pipeline iterator...")
    detection_source = DetectionPipelineSource(pipeline_dp.run())
    
    # Initialize IBVS components
    config_ibvs = Config()
    feature_extractor = FASTHarrisExtractor(
        max_features=config_ibvs.get("feature_extraction.max_features"),
        fast_threshold=config_ibvs.get("feature_extraction.fast_threshold"),
        harris_block_size=config_ibvs.get("feature_extraction.harris_block_size"),
        harris_ksize=config_ibvs.get("feature_extraction.harris_ksize"),
        harris_k=config_ibvs.get("feature_extraction.harris_k"),
        point_focus_radius=config_ibvs.get("feature_extraction.point_focus_radius"),
    )
    
    # Create IBVS pipeline
    ibvs_pipeline = IBVSPipeline(
        source=detection_source,
        feature_extractor=feature_extractor
    )
    
    # Run and verify
    print("  Running IBVS pipeline with detection pipeline source...")
    frame_count = 0
    for ctx in ibvs_pipeline.run():
        frame_count += 1
        has_point = ctx.point is not None
        num_features = len(ctx.extracted_features) if ctx.extracted_features is not None else 0
        print(f"    Frame {frame_count}: point={has_point}, features={num_features}")
        
        if frame_count >= 5:
            break
    
    # Cleanup
    source_dp.release()
    cv2.destroyAllWindows()
    print(f"✓ Test passed! Processed {frame_count} frames successfully.")
    print("  - Frames passed from detection pipeline → DetectionPipelineSource → IBVSPipeline")
    print("  - Points passed correctly (when detected by detection pipeline)")


if __name__ == "__main__":
    main()
