#!/usr/bin/env python3
"""
IBVS ROS Node - Image-Based Visual Servoing for UAV control
Subscribes to detection pipeline topics and computes control velocity
"""

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge

from geometry_msgs.msg import PointStamped, Twist, Vector3Stamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from .pipeline.IBVSContext import IBVSContext
from .feature_extraction.AprilTagExtractor import AprilTagExtractor
from .controller.PointController import PointController


class IBVSNode:
    def __init__(self):
        """Initialize ROS node and subscriptions."""
        rospy.init_node('ibvs_node', anonymous=False)
        
        # Parameters
        self.controller_gain = rospy.get_param('~controller_gain', 0.5)
        self.max_feature_motion = rospy.get_param('~max_feature_motion_px', 50.0)
        self.fallback_enabled = rospy.get_param('~fallback_enabled', True)
        self.fallback_gain_scale = rospy.get_param('~fallback_gain_scale', 0.5)
        self.enable_visualization = rospy.get_param('~enable_visualization', False)
        self.target_x = rospy.get_param('~target_x', None)  # None = image center
        self.target_y = rospy.get_param('~target_y', None)
        
        # Initialize extractor and controller
        self.extractor = AprilTagExtractor()
        self.controller = PointController(
            gain=self.controller_gain,
            validation_enabled=True,
            max_feature_motion_px=self.max_feature_motion,
            fallback_enabled=self.fallback_enabled,
            fallback_gain_scale=self.fallback_gain_scale
        )
        
        self.bridge = CvBridge()
        
        # State
        self.latest_frame = None
        self.latest_point = None
        self.frame_count = 0
        
        # Subscribers (detection pipeline topics)
        rospy.Subscriber('/detection/frame', Image, self.on_frame)
        rospy.Subscriber('/detection/best_candidate', PointStamped, self.on_detection_point)
        
        # Publishers (IBVS output topics)
        self.pub_cmd_vel = rospy.Publisher('/ibvs/cmd_vel', Twist, queue_size=10)
        self.pub_error = rospy.Publisher('/ibvs/error', Vector3Stamped, queue_size=10)
        self.pub_extracted_points = rospy.Publisher('/ibvs/extracted_points', Image, queue_size=10)
        self.pub_diagnostics = rospy.Publisher('/ibvs/diagnostics', Float32MultiArray, queue_size=10)
        self.pub_debug_frame = rospy.Publisher('/ibvs/debug_frame', Image, queue_size=10)
        
        rospy.loginfo("IBVS Node initialized")
        rospy.loginfo(f"  Gain: {self.controller_gain}")
        rospy.loginfo(f"  Fallback enabled: {self.fallback_enabled}")
        rospy.loginfo(f"  Visualization: {self.enable_visualization}")
    
    def on_frame(self, msg):
        """Callback for incoming frame from detection pipeline."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn(f"Frame conversion error: {e}")
    
    def on_detection_point(self, msg):
        """Callback for detection pipeline point (PointStamped)."""
        # Extract pixel coordinates from PointStamped
        point = np.array([msg.point.x, msg.point.y], dtype=np.float32)
        self.latest_point = point
        
        # Process frame if available
        if self.latest_frame is not None:
            self.process_frame()
    
    def process_frame(self):
        """Main IBVS processing pipeline."""
        frame = self.latest_frame.copy()
        point = self.latest_point
        
        # Create context
        ctx = IBVSContext(frame=frame, point=point)
        
        # Set custom target if provided
        if self.target_x is not None and self.target_y is not None:
            self.controller.desired_point = np.array([self.target_x, self.target_y], dtype=np.float32)
        
        # Extract features
        ctx.extracted_features = self.extractor.extract(ctx)
        
        # Update controller (compute velocity command)
        self.controller.update_ctx(ctx)
        
        # Publish results
        self.publish_outputs(ctx)
        
        # Visualization
        if self.enable_visualization:
            self.visualize_frame(ctx)
        
        self.frame_count += 1
        if self.frame_count % 20 == 0:
            rospy.loginfo(f"Processed {self.frame_count} frames")
    
    def publish_outputs(self, ctx):
        """Publish IBVS outputs to ROS topics."""
        header_stamp = rospy.Time.now()
        
        # 1. Publish velocity command as Twist
        vel_cmd = ctx.debug.get('velocity_command')
        if vel_cmd is not None:
            twist = Twist()
            twist.linear.x = float(vel_cmd[0])  # or normalize to m/s
            twist.linear.y = float(vel_cmd[1])
            twist.linear.z = 0.0
            self.pub_cmd_vel.publish(twist)
        
        # 2. Publish control error as Vector3Stamped
        error = ctx.debug.get('control_error_px')
        if error is not None:
            error_msg = Vector3Stamped()
            error_msg.header.stamp = header_stamp
            error_msg.header.frame_id = 'camera'
            error_msg.vector.x = float(error[0])
            error_msg.vector.y = float(error[1])
            error_msg.vector.z = 0.0
            self.pub_error.publish(error_msg)
        
        # 3. Publish extracted points as Image
        if ctx.extracted_features is not None and len(ctx.extracted_features) > 0:
            points_img = ctx.frame.copy()
            for pt in ctx.extracted_features:
                cv2.circle(points_img, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
            try:
                points_msg = self.bridge.cv2_to_imgmsg(points_img, encoding='bgr8')
                points_msg.header.stamp = header_stamp
                points_msg.header.frame_id = 'camera'
                self.pub_extracted_points.publish(points_msg)
            except Exception as e:
                rospy.logwarn(f"Extract points publish error: {e}")
        
        # 4. Publish diagnostics as Float32MultiArray
        diagnostics = Float32MultiArray()
        diagnostics.data = [
            float(ctx.debug.get('control_error_px', [0, 0])[0]),
            float(ctx.debug.get('control_error_px', [0, 0])[1]),
            float(ctx.debug.get('velocity_command', [0, 0])[0]),
            float(ctx.debug.get('velocity_command', [0, 0])[1]),
            float(len(ctx.extracted_features) if ctx.extracted_features is not None else 0),
            float(1.0 if ctx.debug.get('controller', {}).get('point_source') == 'detection' else 0.0),
        ]
        self.pub_diagnostics.publish(diagnostics)
        
        # 5. Publish debug frame with point and features
        debug_frame = ctx.frame.copy()
        if ctx.extracted_features is not None and len(ctx.extracted_features) > 0:
            for pt in ctx.extracted_features:
                cv2.circle(debug_frame, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
        if ctx.point is not None:
            cv2.circle(debug_frame, tuple(ctx.point.astype(int)), 8, (0, 0, 255), -1)
        
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            debug_msg.header.stamp = header_stamp
            debug_msg.header.frame_id = 'camera'
            self.pub_debug_frame.publish(debug_msg)
        except Exception as e:
            rospy.logwarn(f"Debug frame publish error: {e}")
    
    def visualize_frame(self, ctx):
        """Display frame with visualization (for debugging)."""
        display_frame = ctx.frame.copy()
        if ctx.extracted_features is not None and len(ctx.extracted_features) > 0:
            for pt in ctx.extracted_features:
                cv2.circle(display_frame, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
        if ctx.point is not None:
            cv2.circle(display_frame, tuple(ctx.point.astype(int)), 8, (0, 0, 255), -1)
        
        cv2.imshow('IBVS', display_frame)
        cv2.waitKey(1)
    
    def run(self):
        """Main loop (ROS handles callbacks)."""
        rospy.loginfo("IBVS Node running. Waiting for detection pipeline...")
        rospy.spin()


if __name__ == '__main__':
    try:
        node = IBVSNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("IBVS Node shutting down")
        cv2.destroyAllWindows()
