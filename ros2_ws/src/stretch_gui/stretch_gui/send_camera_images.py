#!/usr/bin/env python3
"""
Send camera images from Stretch robot to VLA server
Similar to send_d405_images.py from visual servoing
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import zmq
import argparse
import time
import sys
try:
    from stretch_gui import vla_networking as vn
except ImportError:
    # Fallback - create minimal networking config
    import os
    class NetworkingConfig:
        robot_ip = os.environ.get('VLA_ROBOT_IP', '10.0.0.3')
        remote_computer_ip = os.environ.get('VLA_SERVER_HOST', '10.0.0.1')
        d405_port = int(os.environ.get('D405_PORT', 4405))
        nl_command_port = int(os.environ.get('NL_COMMAND_PORT', 4030))
        nl_target_port = int(os.environ.get('NL_TARGET_PORT', 4040))
    vn = NetworkingConfig()


class CameraImageSender(Node):
    def __init__(self, use_remote_computer=True):
        super().__init__('camera_image_sender')
        
        self.use_remote_computer = use_remote_computer
        self.bridge = CvBridge()
        
        # Setup ZMQ publisher
        self.setup_zmq_publisher()
        
        # Subscribe to camera topics
        self.color_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        
        # Store latest images
        self.latest_color = None
        self.latest_depth = None
        self.color_camera_info = None
        self.depth_camera_info = None
        
        # Stats
        self.iteration_count = 0
        self.start_time = time.time()
        
        self.get_logger().info('Camera Image Sender initialized')
        
    def setup_zmq_publisher(self):
        """Setup ZMQ publisher similar to send_d405_images.py"""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        
        if self.use_remote_computer:
            address = 'tcp://*:' + str(vn.d405_port)
        else:
            address = 'tcp://127.0.0.1:' + str(vn.d405_port)
        
        self.socket.bind(address)
        self.get_logger().info(f'Camera publisher bound to {address}')
        
    def color_callback(self, msg):
        """Callback for color image"""
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.color_camera_info is None:
                # Extract camera info from message (simplified)
                self.color_camera_info = {
                    'width': msg.width,
                    'height': msg.height,
                    'encoding': msg.encoding
                }
            self.maybe_send_data()
        except Exception as e:
            self.get_logger().error(f'Error in color callback: {e}')
            
    def depth_callback(self, msg):
        """Callback for depth image"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            if self.depth_camera_info is None:
                # Extract camera info from message (simplified)
                self.depth_camera_info = {
                    'width': msg.width,
                    'height': msg.height,
                    'encoding': msg.encoding
                }
            self.maybe_send_data()
        except Exception as e:
            self.get_logger().error(f'Error in depth callback: {e}')
            
    def maybe_send_data(self):
        """Send data if both color and depth are available"""
        if self.latest_color is not None and self.latest_depth is not None:
            try:
                # Create output dict similar to d405 format
                d405_output = {
                    'color_image': self.latest_color,
                    'depth_image': self.latest_depth,
                    'color_camera_info': self.color_camera_info,
                    'depth_camera_info': self.depth_camera_info,
                    'depth_scale': 0.001,  # Standard depth scale for most cameras
                    'timestamp': time.time()
                }
                
                # Send via ZMQ
                self.socket.send_pyobj(d405_output)
                
                # Update stats
                self.iteration_count += 1
                if self.iteration_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.iteration_count / elapsed
                    self.get_logger().info(f'Sent {self.iteration_count} frames, avg fps: {fps:.1f}')
                
            except Exception as e:
                self.get_logger().error(f'Error sending data: {e}')
                
    def destroy_node(self):
        """Clean up resources"""
        try:
            self.socket.close(0)
        except Exception:
            pass
        self.context.term()
        super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(
        prog='Send Camera Images',
        description='Send camera images from Stretch robot to VLA server.'
    )
    parser.add_argument('-r', '--remote', action='store_true', 
                       help='Use this argument when allowing a remote computer to receive camera images. '
                            'Configure the network with vla_networking.py on both robot and remote computer.')
    
    parsed_args = parser.parse_args()
    
    rclpy.init(args=args)
    node = CameraImageSender(use_remote_computer=parsed_args.remote)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
