#!/usr/bin/env python3

import sys
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from stretch_interfaces.msg import VLACommand, RobotStatus, AnnotatedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import zmq
import threading
import json
import argparse
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                            QHBoxLayout, QWidget, QLabel, QPushButton, 
                            QTextEdit, QLineEdit, QStatusBar, QScrollArea)
from PyQt5.QtCore import QTimer, pyqtSignal, QThread, Qt
from PyQt5.QtGui import QPixmap, QImage, QFont

# Networking config (類似 yolo_networking)
from . import vla_networking as vn


class VLAResultThread(QThread):
    """ZMQ subscriber thread for VLA results, similar to visual servoing pattern"""
    response_received = pyqtSignal(str)
    
    def __init__(self, use_remote_computer=True):
        super().__init__()
        self.use_remote_computer = use_remote_computer
        self.context = zmq.Context()
        self.running = True
        
        # Subscriber for VLA results
        self.sub = self.context.socket(zmq.SUB)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.setsockopt(zmq.SNDHWM, 1)
        self.sub.setsockopt(zmq.RCVHWM, 1)
        self.sub.setsockopt(zmq.CONFLATE, 1)
        
        if self.use_remote_computer:
            vla_address = 'tcp://' + vn.remote_computer_ip + ':' + str(vn.nl_target_port)
        else:
            vla_address = 'tcp://127.0.0.1:' + str(vn.nl_target_port)
        
        self.sub.connect(vla_address)
        print(f"VLA result subscriber connected to {vla_address}")
        
    def run(self):
        while self.running:
            try:
                # Poll with timeout
                if self.sub.poll(100):  # 100ms timeout
                    result_data = self.sub.recv_pyobj(zmq.NOBLOCK)
                    self.response_received.emit(json.dumps(result_data))
            except zmq.Again:
                pass
            except Exception as e:
                self.response_received.emit(f"Error: {str(e)}")
                
    def stop(self):
        self.running = False
        try:
            self.sub.close(0)
        except Exception:
            pass
        self.context.term()


class CommandPublisherThread(QThread):
    """ZMQ publisher thread for natural language commands"""
    
    def __init__(self, use_remote_computer=True):
        super().__init__()
        self.use_remote_computer = use_remote_computer
        self.context = zmq.Context()
        
        # Publisher for commands
        self.pub = self.context.socket(zmq.PUB)
        self.pub.setsockopt(zmq.SNDHWM, 1)
        self.pub.setsockopt(zmq.RCVHWM, 1)
        
        # Always bind locally for command publishing
        command_address = 'tcp://*:' + str(vn.nl_command_port)
        self.pub.bind(command_address)
        print(f"Command publisher bound to {command_address}")
        
        # Give time for subscribers to connect
        time.sleep(0.1)
        
    def publish_command(self, command, image_data=None):
        try:
            message = {
                "command": command,
                "timestamp": int(time.time() * 1000)
            }
            
            if image_data is not None:
                message["image"] = image_data.tolist() if isinstance(image_data, np.ndarray) else image_data
            
            self.pub.send_string(json.dumps(message))
            
        except Exception as e:
            print(f"Error publishing command: {e}")
            
    def stop(self):
        try:
            self.pub.close(0)
        except Exception:
            pass
        self.context.term()

    def stop(self):
        self.running = False
        try:
            self.pub.close(0)
        except Exception:
            pass
        try:
            self.sub.close(0)
        except Exception:
            pass
        self.context.term()


class StretchGUINode(Node):
    def __init__(self, use_remote_computer=True):
        super().__init__('stretch_gui_node')
        
        # Store remote computer flag
        self.use_remote_computer = use_remote_computer
        
        # Initialize publishers and subscribers
        self.vla_command_pub = self.create_publisher(VLACommand, '/vla_command', 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
        self.annotated_image_sub = self.create_subscription(
            AnnotatedImage, '/annotated_image', self.annotated_image_callback, 10)
        self.robot_status_sub = self.create_subscription(
            RobotStatus, '/robot_status', self.status_callback, 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Current image
        self.current_image = None
        self.current_cv_image = None
        self.current_annotated_image = None
        self.current_targets = []
        
        # Display mode: 'original' or 'annotated'
        self.display_mode = 'original'
        
        # Robot status
        self.robot_status = "idle"
        
        # Setup ZMQ communication similar to visual servoing
        self.setup_zmq_communication()
        
        self.get_logger().info('Stretch GUI Node initialized')
    
    def setup_zmq_communication(self):
        """Setup ZMQ communication threads"""
        # Command publisher thread
        self.command_publisher = CommandPublisherThread(self.use_remote_computer)
        
        # VLA result subscriber thread
        self.result_subscriber = VLAResultThread(self.use_remote_computer)
    
    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV format
            self.current_cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = msg
        except Exception as e:
            self.get_logger().error(f'Error converting image: {str(e)}')
    
    def annotated_image_callback(self, msg):
        """Handle annotated images with detected targets"""
        try:
            # Convert ROS image to OpenCV format
            self.current_annotated_image = self.bridge.imgmsg_to_cv2(msg.image, "bgr8")
            self.current_targets = msg.detected_targets
            
            # Switch to annotated display mode
            self.display_mode = 'annotated'
            
            self.get_logger().info(f'Received annotated image with {len(self.current_targets)} targets')
            
        except Exception as e:
            self.get_logger().error(f'Error converting annotated image: {str(e)}')
    
    def status_callback(self, msg):
        self.robot_status = msg.current_state
    
    def publish_vla_command(self, command_text):
        if self.current_image is None:
            self.get_logger().warning('No image available to send with command')
            return False
        
        # Publish ROS command
        msg = VLACommand()
        msg.command_text = command_text
        msg.image = self.current_image
        msg.timestamp = self.get_clock().now().nanoseconds
        
        self.vla_command_pub.publish(msg)
        
        # Also publish via ZMQ for direct server communication
        if self.current_cv_image is not None:
            self.command_publisher.publish_command(command_text, self.current_cv_image)
        
        self.get_logger().info(f'Published VLA command: {command_text}')
        return True


class StretchGUI(QMainWindow):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        
        # Initialize display mode
        self.display_mode = 'original'
        
        # Setup ZMQ result subscriber for GUI
        self.result_subscriber = self.ros_node.result_subscriber
        self.result_subscriber.response_received.connect(self.handle_vla_response)
        self.result_subscriber.start()
        
        self.init_ui()
        
        # Timer for updating image display
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(33)  # ~30 FPS
        
    def init_ui(self):
        self.setWindowTitle('Stretch3 VLA Control Interface - Object Detection & Tracking')
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Image display
        left_panel = QVBoxLayout()
        
        # Image display with mode toggle
        image_controls = QHBoxLayout()
        self.original_button = QPushButton("Original")
        self.annotated_button = QPushButton("Annotated")
        self.original_button.clicked.connect(lambda: self.set_display_mode('original'))
        self.annotated_button.clicked.connect(lambda: self.set_display_mode('annotated'))
        image_controls.addWidget(self.original_button)
        image_controls.addWidget(self.annotated_button)
        image_controls.addStretch()
        left_panel.addLayout(image_controls)
        
        self.image_label = QLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("border: 1px solid black")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Waiting for camera image...")
        left_panel.addWidget(self.image_label)
        
        # Target information display
        targets_layout = QVBoxLayout()
        targets_layout.addWidget(QLabel("Detected Targets:"))
        
        self.targets_scroll = QScrollArea()
        self.targets_widget = QWidget()
        self.targets_layout = QVBoxLayout(self.targets_widget)
        self.targets_scroll.setWidget(self.targets_widget)
        self.targets_scroll.setMaximumHeight(150)
        targets_layout.addWidget(self.targets_scroll)
        left_panel.addLayout(targets_layout)
        
        # Right panel - Controls
        right_panel = QVBoxLayout()
        
        # Command input
        right_panel.addWidget(QLabel("Natural Language Command:"))
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("e.g., Pick up the red ball")
        self.command_input.returnPressed.connect(self.send_command)
        right_panel.addWidget(self.command_input)
        
        # Send button
        self.send_button = QPushButton("Send Command")
        self.send_button.clicked.connect(self.send_command)
        right_panel.addWidget(self.send_button)
        
        # Response area
        right_panel.addWidget(QLabel("VLA Response:"))
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.response_text.setMaximumHeight(200)
        right_panel.addWidget(self.response_text)
        
        # Robot status
        right_panel.addWidget(QLabel("Robot Status:"))
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-weight: bold; color: green")
        right_panel.addWidget(self.status_label)
        
        # Target tracking info
        right_panel.addWidget(QLabel("Current Target:"))
        self.target_info_label = QLabel("No target selected")
        self.target_info_label.setStyleSheet("font-family: monospace; background-color: #f0f0f0; padding: 10px")
        self.target_info_label.setWordWrap(True)
        right_panel.addWidget(self.target_info_label)
        
        # Emergency stop
        self.stop_button = QPushButton("EMERGENCY STOP")
        self.stop_button.setStyleSheet("background-color: red; color: white; font-weight: bold")
        self.stop_button.clicked.connect(self.emergency_stop)
        right_panel.addWidget(self.stop_button)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        
        # Status bar
        self.statusBar().showMessage('Ready')
    
    def update_display(self):
        # Update image display based on mode
        if self.display_mode == 'annotated' and self.ros_node.current_annotated_image is not None:
            self.display_image(self.ros_node.current_annotated_image)
            self.update_targets_display()
            self.update_target_info()
        elif self.ros_node.current_cv_image is not None:
            self.display_image(self.ros_node.current_cv_image)
        
        # Update robot status
        status_text = f"Status: {self.ros_node.robot_status.title()}"
        self.status_label.setText(status_text)
        
        # Update status color
        if self.ros_node.robot_status == "idle":
            self.status_label.setStyleSheet("font-weight: bold; color: green")
        elif self.ros_node.robot_status == "moving":
            self.status_label.setStyleSheet("font-weight: bold; color: blue")
        elif self.ros_node.robot_status == "grasping":
            self.status_label.setStyleSheet("font-weight: bold; color: orange")
        elif self.ros_node.robot_status == "error":
            self.status_label.setStyleSheet("font-weight: bold; color: red")
    
    def set_display_mode(self, mode):
        """Set display mode to 'original' or 'annotated'"""
        self.display_mode = mode
        self.ros_node.display_mode = mode
        
        # Update button styles
        if mode == 'original':
            self.original_button.setStyleSheet("background-color: #4CAF50; color: white")
            self.annotated_button.setStyleSheet("")
        else:
            self.annotated_button.setStyleSheet("background-color: #4CAF50; color: white")
            self.original_button.setStyleSheet("")
    
    def update_targets_display(self):
        """Update the targets display list"""
        # Clear existing targets
        for i in reversed(range(self.targets_layout.count())): 
            self.targets_layout.itemAt(i).widget().setParent(None)
        
        # Add current targets
        for i, target in enumerate(self.ros_node.current_targets):
            target_label = QLabel(
                f"{i+1}. {target.object_description} "
                f"({target.target_position_2d.x:.0f}, {target.target_position_2d.y:.0f}) "
                f"- {target.confidence:.2f}"
            )
            target_label.setStyleSheet("padding: 5px; border: 1px solid gray; margin: 2px")
            if i == 0:  # Highlight best target
                target_label.setStyleSheet("padding: 5px; border: 2px solid green; margin: 2px; background-color: #e8f5e8")
            self.targets_layout.addWidget(target_label)
    
    def update_target_info(self):
        """Update current target information"""
        if self.ros_node.current_targets:
            best_target = self.ros_node.current_targets[0]
            info_text = (
                f"Object: {best_target.object_description}\n"
                f"2D Position: ({best_target.target_position_2d.x:.1f}, {best_target.target_position_2d.y:.1f})\n"
                f"3D Position: ({best_target.target_position_3d.x:.3f}, {best_target.target_position_3d.y:.3f}, {best_target.target_position_3d.z:.3f})\n"
                f"Confidence: {best_target.confidence:.2f}"
            )
            self.target_info_label.setText(info_text)
        else:
            self.target_info_label.setText("No target selected")
    
    def display_image(self, cv_image):
        """Display OpenCV image in QLabel"""
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Scale image to fit label
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
    
    def send_command(self):
        command_text = self.command_input.text().strip()
        if not command_text:
            self.statusBar().showMessage('Please enter a command')
            return
        
        # Publish command via ROS and ZMQ
        success = self.ros_node.publish_vla_command(command_text)
        if not success:
            self.statusBar().showMessage('No image available - please wait for camera')
            return
        
        self.statusBar().showMessage('Command sent, waiting for response...')
        self.send_button.setEnabled(False)
        
        # Clear input
        self.command_input.clear()
    
    def handle_vla_response(self, response):
        """Handle response from VLA server"""
        self.response_text.append(f"Response: {response}")
        self.statusBar().showMessage('Response received')
        self.send_button.setEnabled(True)
    
    def emergency_stop(self):
        """Emergency stop function"""
        self.ros_node.get_logger().warn('Emergency stop triggered!')
        # Here you would implement actual emergency stop logic
        self.statusBar().showMessage('EMERGENCY STOP ACTIVATED')
    
    def closeEvent(self, event):
        """Clean up when closing"""
        if hasattr(self, 'result_subscriber') and self.result_subscriber is not None:
            self.result_subscriber.stop()
        if hasattr(self.ros_node, 'command_publisher') and self.ros_node.command_publisher is not None:
            self.ros_node.command_publisher.stop()
        event.accept()


def main(args=None):
    parser = argparse.ArgumentParser(
        prog='Stretch GUI Node',
        description='GUI for Stretch robot VLA control.'
    )
    parser.add_argument('-r', '--remote', action='store_true', 
                       help='Use this argument when using a remote VLA server. '
                            'By default, assumes local operation.')
    
    parsed_args = parser.parse_args()
    
    # Initialize ROS
    rclpy.init(args=args)
    
    # Create ROS node
    ros_node = StretchGUINode(use_remote_computer=parsed_args.remote)
    
    # Check if we have a display available
    display_available = os.environ.get('DISPLAY') not in (None, '')
    
    # Run headless only if no display is available
    if not display_available:
        ros_node.get_logger().warn('No display available - starting in headless mode')
        
        # Run in headless mode - just ROS node without GUI
        def ros_spin():
            rclpy.spin(ros_node)
        
        try:
            ros_node.get_logger().info('Stretch GUI running in headless mode - ROS topics available')
            ros_spin()
        except KeyboardInterrupt:
            ros_node.get_logger().info('Shutting down headless mode')
        finally:
            ros_node.destroy_node()
            rclpy.shutdown()
        return
    
    try:
        # Try to create Qt application
        app = QApplication(sys.argv)
        
        # Create GUI
        gui = StretchGUI(ros_node)
        gui.show()
        
        # Run ROS in separate thread
        def ros_spin():
            rclpy.spin(ros_node)
        
        ros_thread = threading.Thread(target=ros_spin, daemon=True)
        ros_thread.start()
        
        # Run Qt event loop
        ros_node.get_logger().info('Starting Stretch GUI with display')
        sys.exit(app.exec_())
        
    except Exception as e:
        ros_node.get_logger().error(f'Failed to start GUI: {str(e)}')
        ros_node.get_logger().info('Falling back to headless mode')
        
        # Fallback to headless mode
        try:
            rclpy.spin(ros_node)
        except KeyboardInterrupt:
            pass
        finally:
            ros_node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
