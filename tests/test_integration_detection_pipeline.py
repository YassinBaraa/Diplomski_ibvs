import os
import sys
import unittest

from feature_extraction.FASTHarrisExtractor import FASTHarrisExtractor
from pipeline.IBVSPipeline import IBVSPipeline
from sources.DetectionPipelineSource import DetectionPipelineSource


class TestDetectionIntegration(unittest.TestCase):
    def _build_detection_pipeline(self, detection_repo: str, video_path: str, model_path: str):
        detection_repo_abs = os.path.abspath(detection_repo)
        if detection_repo_abs not in sys.path:
            sys.path.insert(0, detection_repo_abs)

        from detection_pipeline_ros.detectors.YOLOBranchSeg import YOLOBranchSeg
        from detection_pipeline_ros.pipeline.DetectionPipeline import DetectionPipeline
        from detection_pipeline_ros.postprocessing.PostProcessor import PostProcessor
        from detection_pipeline_ros.postprocessing.geometry.BitmaskSkeleton import BitmaskSkeleton
        from detection_pipeline_ros.postprocessing.geometry.DistanceHeatmap import DistanceHeatmap
        from detection_pipeline_ros.postprocessing.masks.MaskExtraction import MaskExtraction
        from detection_pipeline_ros.postprocessing.scoring.CandidateScoring import CandidateScoring
        from detection_pipeline_ros.postprocessing.scoring.CandidateVisualizer import CandidateVisualizer
        from detection_pipeline_ros.sources.MP4Source import MP4Source as DetectionMP4Source
        from detection_pipeline_ros.trackers.ByteTrack import ByteTrack

        source = DetectionMP4Source(video_path)
        detector = YOLOBranchSeg(model_path=model_path, conf=0.5)
        tracker = ByteTrack(detector)
        postprocessor = PostProcessor(
            [
                MaskExtraction(),
                DistanceHeatmap(),
                BitmaskSkeleton(),
                CandidateScoring(),
                CandidateVisualizer(),
            ]
        )
        return DetectionPipeline(
            source=source,
            detector=detector,
            tracker=tracker,
            postprocessor=postprocessor,
        )

    def test_detection_pipeline_to_ibvs_flow(self):
        if os.getenv("IBVS_RUN_DETECTION_INTEGRATION") != "1":
            self.skipTest("Set IBVS_RUN_DETECTION_INTEGRATION=1 to enable full integration test")

        detection_repo = os.getenv("IBVS_DETECTION_REPO", "../detection_pipeline")
        video_path = os.getenv("IBVS_DETECTION_VIDEO", "")
        model_path = os.getenv("IBVS_DETECTION_MODEL", "")

        if not video_path or not model_path:
            self.skipTest(
                "Set IBVS_DETECTION_VIDEO and IBVS_DETECTION_MODEL for integration test"
            )
        if not os.path.exists(video_path) or not os.path.exists(model_path):
            self.skipTest("IBVS_DETECTION_VIDEO/IBVS_DETECTION_MODEL path does not exist")

        try:
            detection_pipeline = self._build_detection_pipeline(
                detection_repo=detection_repo,
                video_path=video_path,
                model_path=model_path,
            )
        except Exception as exc:
            self.skipTest(f"Could not initialize detection pipeline: {exc}")

        source = DetectionPipelineSource(detection_pipeline.run())
        extractor = FASTHarrisExtractor(max_features=25)
        ibvs_pipeline = IBVSPipeline(source=source, feature_extractor=extractor)

        processed = 0
        for ctx in ibvs_pipeline.run():
            processed += 1
            self.assertIsNotNone(ctx.frame)
            self.assertIsNotNone(ctx.extracted_features)
            if processed >= 2:
                break

        self.assertGreater(processed, 0)


if __name__ == "__main__":
    unittest.main()
