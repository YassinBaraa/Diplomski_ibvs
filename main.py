#!/usr/bin/env python3
from sources.MP4Source import MP4Source
from sources.DetectionPipelineSource import DetectionPipelineSource
from feature_extraction.FASTHarrisExtractor import FASTHarrisExtractor
from pipeline.IBVSPipeline import IBVSPipeline
from config import ConfigManager

import cv2
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Load configuration
    try:
        config = ConfigManager()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    logger.info("Starting IBVS Pipeline")

    # Initialize source
    source_type = config.get("source.type", "mp4")
    if source_type == "detection":
        # Placeholder: detection pipeline context iterator
        # In practice, this will be passed from outside or built here
        logger.warning("Detection pipeline source requires external context iterator")
        return
    else:
        source = MP4Source(config.get("source.video_path"))

    # Initialize feature extractor
    extractor_config = config.get_section("feature_extraction")
    feature_extractor = FASTHarrisExtractor(
        max_features=extractor_config.get("max_features", 80),
        fast_threshold=extractor_config.get("fast_threshold", 20),
        harris_block_size=extractor_config.get("harris_block_size", 2),
        harris_ksize=extractor_config.get("harris_ksize", 3),
        harris_k=extractor_config.get("harris_k", 0.04),
        point_focus_radius=extractor_config.get("point_focus_radius", None),
    )

    # Create pipeline
    pipeline = IBVSPipeline(
        source=source,
        feature_extractor=feature_extractor
    )

    # Process frames
    frame_count = 0
    visualization_enabled = config.get("visualization.enabled", True)

    try:
        for ctx in pipeline.run():
            frame_count += 1

            # Display visualization if enabled
            if visualization_enabled and ctx.frame is not None:
                display_frame = ctx.frame.copy()
                
                # Draw target point if available
                if ctx.point is not None:
                    pt = tuple(map(int, ctx.point[:2]))
                    cv2.circle(display_frame, pt, 5, (0, 255, 0), -1)

                # Draw extracted features
                if ctx.extracted_features is not None and len(ctx.extracted_features) > 0:
                    for feat in ctx.extracted_features:
                        pt = tuple(map(int, feat[:2]))
                        cv2.circle(display_frame, pt, 3, (255, 0, 0), 1)

                cv2.imshow(
                    config.get("visualization.display_window_name", "IBVS"),
                    display_frame
                )
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User quit")
                    break

            if frame_count % 30 == 0:
                logger.debug(f"Processed {frame_count} frames")

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
    finally:
        source.release()
        cv2.destroyAllWindows()
        logger.info(f"Pipeline complete. Processed {frame_count} frames total")


if __name__ == "__main__":
    main()
