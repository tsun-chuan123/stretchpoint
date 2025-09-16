#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import cv2

class D435iCameraNode(Node):
    def __init__(self):
        super().__init__('d435i_camera_node')
        
        # Initialize publishers
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.color_info_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)
        self.depth_info_pub = self.create_publisher(CameraInfo, '/camera/depth/camera_info', 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Start streaming
        try:
            self.profile = self.pipeline.start(self.config)
            self.get_logger().info('RealSense D435i camera started successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to start camera: {str(e)}')
            return
        
        # Get camera intrinsics
        self.color_intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        
        # Depth scale (for converting depth units to meters)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.get_logger().info(f'Depth scale: {self.depth_scale}')
        
        # Create timer for publishing frames
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)  # 30 FPS
        
    def pixel_to_3d(self, pixel_x, pixel_y, depth_value):
        """Convert 2D pixel coordinates to 3D camera coordinates"""
        if depth_value <= 0:
            return None
            
        # Convert depth to meters
        depth_m = depth_value * self.depth_scale
        
        # Use color camera intrinsics for transformation
        fx = self.color_intrinsics.fx
        fy = self.color_intrinsics.fy
        cx = self.color_intrinsics.ppx
        cy = self.color_intrinsics.ppy
        
        # Calculate 3D coordinates in camera frame
        x = (pixel_x - cx) * depth_m / fx
        y = (pixel_y - cy) * depth_m / fy
        z = depth_m
        
        return np.array([x, y, z])
    
    def get_depth_at_pixel(self, depth_image, pixel_x, pixel_y, window_size=5):
        """Get robust depth value at pixel location using median filtering"""
        h, w = depth_image.shape
        
        # Ensure pixel is within image bounds
        pixel_x = max(0, min(w-1, int(pixel_x)))
        pixel_y = max(0, min(h-1, int(pixel_y)))
        
        # Extract window around pixel
        half_window = window_size // 2
        x_start = max(0, pixel_x - half_window)
        x_end = min(w, pixel_x + half_window + 1)
        y_start = max(0, pixel_y - half_window)
        y_end = min(h, pixel_y + half_window + 1)
        
        depth_window = depth_image[y_start:y_end, x_start:x_end]
        valid_depths = depth_window[depth_window > 0]
        
        if len(valid_depths) == 0:
            return 0
        
        return np.median(valid_depths)
        
        # Create timer for publishing frames
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)  # 30 FPS
        
    def timer_callback(self):
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Create ROS messages
            color_msg = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, "16UC1")
            
            # Set timestamps
            timestamp = self.get_clock().now().to_msg()
            color_msg.header.stamp = timestamp
            depth_msg.header.stamp = timestamp
            color_msg.header.frame_id = "camera_color_optical_frame"
            depth_msg.header.frame_id = "camera_depth_optical_frame"
            
            # Publish images
            self.color_pub.publish(color_msg)
            self.depth_pub.publish(depth_msg)
            
            # Publish camera info
            self.publish_camera_info(timestamp)
            
        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {str(e)}')
    
    def publish_camera_info(self, timestamp):
        # Color camera info
        color_info = CameraInfo()
        color_info.header.stamp = timestamp
        color_info.header.frame_id = "camera_color_optical_frame"
        color_info.width = self.color_intrinsics.width
        color_info.height = self.color_intrinsics.height
        color_info.k = [self.color_intrinsics.fx, 0.0, self.color_intrinsics.ppx,
                       0.0, self.color_intrinsics.fy, self.color_intrinsics.ppy,
                       0.0, 0.0, 1.0]
        color_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # Assuming no distortion
        color_info.p = [self.color_intrinsics.fx, 0.0, self.color_intrinsics.ppx, 0.0,
                       0.0, self.color_intrinsics.fy, self.color_intrinsics.ppy, 0.0,
                       0.0, 0.0, 1.0, 0.0]
        
        # Depth camera info
        depth_info = CameraInfo()
        depth_info.header.stamp = timestamp
        depth_info.header.frame_id = "camera_depth_optical_frame"
        depth_info.width = self.depth_intrinsics.width
        depth_info.height = self.depth_intrinsics.height
        depth_info.k = [self.depth_intrinsics.fx, 0.0, self.depth_intrinsics.ppx,
                       0.0, self.depth_intrinsics.fy, self.depth_intrinsics.ppy,
                       0.0, 0.0, 1.0]
        depth_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # Assuming no distortion
        depth_info.p = [self.depth_intrinsics.fx, 0.0, self.depth_intrinsics.ppx, 0.0,
                       0.0, self.depth_intrinsics.fy, self.depth_intrinsics.ppy, 0.0,
                       0.0, 0.0, 1.0, 0.0]
        
        self.color_info_pub.publish(color_info)
        self.depth_info_pub.publish(depth_info)
    
    def destroy_node(self):
        try:
            self.pipeline.stop()
        except:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = D435iCameraNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
