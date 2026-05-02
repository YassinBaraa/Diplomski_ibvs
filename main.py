from sources.MP4Source import MP4Source
#from sources.DetectionPipelineSource import DetectionPipelineSource
from feature_extraction.AprilTagExtractor import AprilTagExtractor
#from feature_extraction import FASTHarrisExtractor
from pipeline.IBVSPipeline import IBVSPipeline
from config import Config


import cv2


def main():
    
    try:
        config = Config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    

    source = MP4Source(config.get("source.video_path"))
    #source = DetectionPipelineSource(config.get("sources.detection_pipeline_path")) # find a better way torun the pipeline

    feature_extractor = AprilTagExtractor()
    #feature_extractor = FASTHarrisExtractor(config.get("feature_extraction.fastharris")) 

    pipeline = IBVSPipeline(
        source=source,
        feature_extractor=feature_extractor
    
    )

    frame_count = 0
    for ctx in pipeline.run():
        frame_count += 1
        print(f"Frame {frame_count}: Extracted {len(ctx.extracted_features)} features")

        # Visualization: draw extracted feature centers on the frame and display
        try:
            for (x, y) in ctx.extracted_features:
                cv2.circle(ctx.frame, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.waitKey(100)
            cv2.imshow('IBVS - AprilTag features', ctx.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception:
            # If visualization fails for any reason, continue without stopping pipeline
            pass

    # Cleanup
    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
