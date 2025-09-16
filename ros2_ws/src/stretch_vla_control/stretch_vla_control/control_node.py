#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from stretch_interfaces.msg import GraspTarget, RobotStatus, AnnotatedImage
from stretch_interfaces.srv import ExecuteGrasp
from geometry_msgs.msg import Point, Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import threading
import os
import logging
import logging.config
import copy
import sys
import types

# Ensure Stretch SDK logging doesn't attempt file writes on read-only FS
_orig_dictConfig = logging.config.dictConfig
def _safe_dictConfig(cfg):
    try:
        new_cfg = copy.deepcopy(cfg)
        handlers = new_cfg.get('handlers', {})
        for name, h in list(handlers.items()):
            if isinstance(h, dict) and ('filename' in h or h.get('class', '').endswith('FileHandler')):
                h.pop('filename', None)
                h.pop('mode', None)
                h.pop('delay', None)
                h['class'] = 'logging.StreamHandler'
                h.setdefault('stream', 'ext://sys.stderr')
        return _orig_dictConfig(new_cfg)
    except Exception:
        return _orig_dictConfig(cfg)

logging.config.dictConfig = _safe_dictConfig

# Optionally stub collision manager to avoid heavy SciPy/URDF deps
def _maybe_stub_collision_manager():
    use_collision = os.environ.get('USE_STRETCH_COLLISION', '0').lower() in ('1', 'true', 'yes')
    if not use_collision:
        mod = types.ModuleType('stretch_body.robot_collision')
        class RobotCollisionMgmt:  # minimal no-op stub compatible with SDK
            def __init__(self, *args, **kwargs):
                # Some SDK versions check this flag in monitor threads
                self.running = False
            def startup(self, *args, **kwargs):
                # Keep disabled by default unless explicitly enabled
                self.running = False
            def enable(self, *args, **kwargs):
                self.running = True
            def disable(self, *args, **kwargs):
                self.running = False
            def step(self, *args, **kwargs):
                # No-op when stubbed
                return
        mod.RobotCollisionMgmt = RobotCollisionMgmt
        sys.modules['stretch_body.robot_collision'] = mod

_maybe_stub_collision_manager()

# Use real hardware: import Stretch SDK at module load (with safe logging)
import stretch_body.robot as rb
from stretch_body import robot_params
from stretch_body import hello_utils as hu

# Restore standard logging configuration function
logging.config.dictConfig = _orig_dictConfig

class CoordinateTransformer:
    """Handle coordinate transformations between 2D pixels and 3D world coordinates"""
    
    def __init__(self):
        self.camera_info = None
        self.depth_scale = 0.001  # Default depth scale for RealSense
        
    def set_camera_info(self, camera_info_msg):
        """Set camera intrinsic parameters from ROS CameraInfo message"""
        self.camera_info = {
            'camera_matrix': np.array(camera_info_msg.k).reshape(3, 3),
            'width': camera_info_msg.width,
            'height': camera_info_msg.height
        }
    
    def pixel_to_3d(self, xy_pix, depth_value):
        """Convert 2D pixel coordinates to 3D world coordinates"""
        if self.camera_info is None:
            raise ValueError("Camera info not set")
        
        x_pix, y_pix = xy_pix
        z_in = depth_value * self.depth_scale  # Convert depth to meters
        
        camera_matrix = self.camera_info['camera_matrix']
        f_x = camera_matrix[0, 0]
        c_x = camera_matrix[0, 2]
        f_y = camera_matrix[1, 1]
        c_y = camera_matrix[1, 2]
        
        x_out = ((x_pix - c_x) * z_in) / f_x
        y_out = ((y_pix - c_y) * z_in) / f_y
        
        return np.array([x_out, y_out, z_in])
    
    def pixel_from_3d(self, xyz):
        """Convert 3D world coordinates to 2D pixel coordinates"""
        if self.camera_info is None:
            raise ValueError("Camera info not set")
        
        x_in, y_in, z_in = xyz
        camera_matrix = self.camera_info['camera_matrix']
        f_x = camera_matrix[0, 0]
        c_x = camera_matrix[0, 2]
        f_y = camera_matrix[1, 1]
        c_y = camera_matrix[1, 2]
        
        x_pix = ((f_x * x_in) / z_in) + c_x
        y_pix = ((f_y * y_in) / z_in) + c_y
        
        return np.array([x_pix, y_pix])


class StretchVLAController:
    """Main controller for Stretch robot using VLA (Visual Language Acting)"""
    
    def __init__(self, node):
        self.node = node
        self.robot = None
        self.coordinate_transformer = CoordinateTransformer()
        
        # Robot state
        self.current_state = "idle"  # idle, moving, grasping, error
        self.last_error = ""
        
        # Control parameters (load from ROS parameters)
        self.approach_distance = node.get_parameter('robot.approach_distance').get_parameter_value().double_value
        self.grasp_height_offset = node.get_parameter('robot.grasp_height_offset').get_parameter_value().double_value
        self.max_reach_distance = node.get_parameter('robot.max_reach_distance').get_parameter_value().double_value
        
        # Target tracking
        self.current_target = None
        self.target_tracking_enabled = False
        self.last_target_time = 0
        
        # Movement parameters (load from ROS parameters)
        try:
            self.base_translate_speed = node.get_parameter('robot.base.max_linear_velocity').get_parameter_value().double_value
            self.base_rotate_speed = node.get_parameter('robot.base.max_angular_velocity').get_parameter_value().double_value
            self.arm_extend_speed = node.get_parameter('robot.arm.default_speed').get_parameter_value().double_value
            self.lift_speed = node.get_parameter('robot.lift.default_speed').get_parameter_value().double_value
            
            # Set depth scale
            depth_scale = node.get_parameter('camera.depth_scale').get_parameter_value().double_value
            self.coordinate_transformer.depth_scale = depth_scale
        except Exception as e:
            node.get_logger().warning(f'Could not load all parameters, using defaults: {e}')
            # Fallback to default values
            self.base_translate_speed = 0.1  # m/s
            self.base_rotate_speed = 0.3  # rad/s
            self.arm_extend_speed = 0.1  # m/s
            self.lift_speed = 0.1  # m/s
            self.coordinate_transformer.depth_scale = 0.001
        
        # Initialize robot connection
        self.connect_to_robot()
    
    def connect_to_robot(self):
        """Connect to Stretch robot (no simulation fallback)."""
        # Detect Docker and require explicit opt-in
        in_docker = os.path.exists('/.dockerenv')
        allow_hw_in_docker = os.environ.get('USE_STRETCH_HARDWARE', '0').lower() in ('1', 'true', 'yes')
        if in_docker and not allow_hw_in_docker:
            raise RuntimeError('Running in Docker. Set USE_STRETCH_HARDWARE=1 and mount calibration to use hardware.')

        # Resolve calibration directory (HELLO_FLEET_PATH/ID, ~/stretch_user, /root/stretch_user)
        fleet_id = os.environ.get('HELLO_FLEET_ID', '')
        def with_id(p):
            return os.path.join(p, fleet_id) if fleet_id else p

        env_fp = os.environ.get('HELLO_FLEET_PATH')
        candidates = []
        if env_fp:
            candidates.append(with_id(env_fp))
        candidates.append(with_id(os.path.expanduser('~/stretch_user')))
        candidates.append(with_id('/root/stretch_user'))

        def has_calibration(d):
            return os.path.isdir(d) and (
                os.path.exists(os.path.join(d, 'stretch_user_params.yaml')) or
                os.path.exists(os.path.join(d, 'stretch_configuration_params.yaml'))
            )

        calib_dir = next((d for d in candidates if has_calibration(d)), None)
        if calib_dir is None:
            details = '\n'.join(f'- {c}' for c in candidates)
            raise FileNotFoundError(f'Calibration not found. Checked:\n{details}\nSet HELLO_FLEET_PATH/HELLO_FLEET_ID or mount the directory into the container.')

        # Ensure env consistent for downstream tools
        found_fp = os.path.dirname(calib_dir) if fleet_id else calib_dir
        os.environ.setdefault('HELLO_FLEET_PATH', found_fp)
        self.node.get_logger().info(f'Using calibration at: {calib_dir}')

        # Connect to robot
        self.robot = rb.Robot()
        self.robot.startup()
        self.node.get_logger().info('Connected to Stretch robot successfully')
        # Move to home position
        self.home_position()
    
    def home_position(self):
        """Move robot to home/ready position"""
        if self.robot is None:
            # Simulation mode
            self.node.get_logger().info('Simulation: Moving robot to home position')
            self.current_state = "moving"
            time.sleep(1.0)  # Simulate movement time
            self.current_state = "idle"
            self.node.get_logger().info('Simulation: Robot moved to home position')
            return True
        
        try:
            self.current_state = "moving"
            
            # Move to a ready position
            self.robot.stow()
            self.robot.push_command()
            
            # Wait for completion
            time.sleep(2.0)
            
            # Lift the arm to a ready height
            self.robot.lift.move_to(0.6)  # 60cm height
            self.robot.arm.move_to(0.1)   # Extend arm slightly
            self.robot.push_command()
            
            time.sleep(2.0)
            
            self.current_state = "idle"
            self.node.get_logger().info('Robot moved to home position')
            return True
            
        except Exception as e:
            self.node.get_logger().error(f'Error moving to home position: {str(e)}')
            self.current_state = "error"
            self.last_error = str(e)
            return False
    
    def update_target_tracking(self, grasp_target):
        """Update current target for tracking"""
        self.current_target = grasp_target
        self.last_target_time = time.time()
        self.target_tracking_enabled = True
        self.node.get_logger().info(
            f'Target updated: {grasp_target.object_description} at '
            f'({grasp_target.target_position_2d.x:.1f}, {grasp_target.target_position_2d.y:.1f})'
        )
    
    def move_to_target(self, grasp_target, depth_image=None):
        """Move robot towards the detected target point"""
        if self.robot is None:
            self.node.get_logger().warning('Robot not connected - cannot move to target')
            return False, "Robot not connected"
        
        try:
            self.current_state = "moving"
            
            # Get 3D coordinates from target
            target_3d = self.get_3d_coordinates(grasp_target, depth_image)
            if target_3d is None:
                return False, "Could not determine 3D coordinates"
            
            # Transform to robot base frame
            robot_target = self.transform_to_robot_frame(target_3d)
            
            # Check if target is reachable
            if not self.is_target_reachable(robot_target):
                return False, "Target is not reachable"
            
            # Calculate required movements
            base_x, base_y, base_theta = self.calculate_base_movement(robot_target)
            arm_extension, lift_height = self.calculate_arm_movement(robot_target)
            
            # Execute movements
            success = self.execute_movement_sequence(base_x, base_y, base_theta, arm_extension, lift_height)
            
            if success:
                self.current_state = "idle"
                return True, "Moved to target successfully"
            else:
                self.current_state = "error"
                return False, "Movement failed"
                
        except Exception as e:
            self.node.get_logger().error(f'Error moving to target: {str(e)}')
            self.current_state = "error"
            self.last_error = str(e)
            return False, str(e)
    
    def calculate_base_movement(self, robot_target):
        """Calculate required base movement to position robot optimally for reaching target"""
        target_x, target_y, target_z = robot_target
        
        # Calculate distance and angle to target
        distance_to_target = np.sqrt(target_x**2 + target_y**2)
        angle_to_target = np.arctan2(target_y, target_x)
        
        # Determine optimal base position
        # We want to be close enough to reach, but not too close
        optimal_distance = self.max_reach_distance * 0.7  # Use 70% of max reach
        
        if distance_to_target > optimal_distance:
            # Move closer to target
            move_distance = distance_to_target - optimal_distance
            base_x = move_distance * np.cos(angle_to_target)
            base_y = move_distance * np.sin(angle_to_target)
        else:
            # Already close enough
            base_x = 0
            base_y = 0
        
        # Calculate rotation to face target
        base_theta = angle_to_target
        
        return base_x, base_y, base_theta
    
    def calculate_arm_movement(self, robot_target):
        """Calculate required arm extension and lift height"""
        target_x, target_y, target_z = robot_target
        
        # Calculate horizontal distance (after base movement)
        horizontal_distance = np.sqrt(target_x**2 + target_y**2)
        
        # Arm extension (accounting for some safety margin)
        arm_extension = max(0.1, min(self.max_reach_distance, horizontal_distance - 0.1))
        
        # Lift height (target height plus offset)
        lift_height = max(0.2, min(1.1, target_z + self.grasp_height_offset))
        
        return arm_extension, lift_height
    
    def execute_movement_sequence(self, base_x, base_y, base_theta, arm_extension, lift_height):
        """Execute the calculated movement sequence"""
        try:
            # Rotate base to face target
            if abs(base_theta) > 0.05:  # 3 degree threshold
                self.robot.base.rotate_by(base_theta)
                self.robot.push_command()
                time.sleep(abs(base_theta) / self.base_rotate_speed + 0.5)
            
            # Move base closer if needed
            if base_x != 0 or base_y != 0:
                move_distance = np.sqrt(base_x**2 + base_y**2)
                self.robot.base.translate_by(move_distance)
                self.robot.push_command()
                time.sleep(move_distance / self.base_translate_speed + 0.5)
            
            # Extend arm and adjust lift height
            self.robot.arm.move_to(arm_extension)
            self.robot.lift.move_to(lift_height)
            self.robot.push_command()
            
            # Wait for arm movement to complete
            time.sleep(max(arm_extension / self.arm_extend_speed, lift_height / self.lift_speed) + 1.0)
            
            self.node.get_logger().info(
                f'Movement completed: base({base_x:.2f}, {base_y:.2f}, {base_theta:.2f}), '
                f'arm({arm_extension:.2f}), lift({lift_height:.2f})'
            )
            
            return True
            
        except Exception as e:
            self.node.get_logger().error(f'Error in movement sequence: {str(e)}')
            return False
    
    def execute_grasp(self, grasp_target, depth_image=None):
        """Execute grasp based on the target information"""
        if self.robot is None:
            return False, "Robot not connected"
        
        try:
            self.current_state = "moving"
            
            # Get 3D coordinates from 2D pixel coordinates
            target_3d = self.get_3d_coordinates(grasp_target, depth_image)
            if target_3d is None:
                return False, "Could not determine 3D coordinates"
            
            # Transform coordinates to robot base frame
            robot_target = self.transform_to_robot_frame(target_3d)
            
            # Check if target is reachable
            if not self.is_target_reachable(robot_target):
                return False, "Target is not reachable"
            
            # Execute grasp sequence
            success = self.execute_grasp_sequence(robot_target)
            
            if success:
                self.current_state = "idle"
                return True, "Grasp executed successfully"
            else:
                self.current_state = "error"
                return False, "Grasp execution failed"
                
        except Exception as e:
            self.node.get_logger().error(f'Error executing grasp: {str(e)}')
            self.current_state = "error"
            self.last_error = str(e)
            return False, str(e)
    
    def get_3d_coordinates(self, grasp_target, depth_image):
        """Get 3D coordinates from 2D pixel coordinates and depth"""
        try:
            # Get pixel coordinates
            x_pix = int(grasp_target.target_position_2d.x)
            y_pix = int(grasp_target.target_position_2d.y)
            
            if depth_image is not None:
                # Get depth value at the target pixel
                height, width = depth_image.shape
                
                # Ensure coordinates are within image bounds
                x_pix = max(0, min(x_pix, width - 1))
                y_pix = max(0, min(y_pix, height - 1))
                
                depth_value = depth_image[y_pix, x_pix]
                
                # If depth is invalid, use surrounding pixels
                if depth_value == 0:
                    depth_value = self.get_valid_depth_around_pixel(
                        depth_image, x_pix, y_pix, radius=5)
                
                if depth_value > 0:
                    # Convert to 3D coordinates
                    target_3d = self.coordinate_transformer.pixel_to_3d(
                        [x_pix, y_pix], depth_value)
                    return target_3d
                else:
                    self.node.get_logger().warning('Invalid depth value, using default distance')
            
            # If no depth available, use default distance
            default_distance = 0.5  # 50cm default
            target_3d = self.coordinate_transformer.pixel_to_3d(
                [x_pix, y_pix], int(default_distance / self.coordinate_transformer.depth_scale))
            
            return target_3d
            
        except Exception as e:
            self.node.get_logger().error(f'Error getting 3D coordinates: {str(e)}')
            return None
    
    def get_valid_depth_around_pixel(self, depth_image, x, y, radius=5):
        """Get a valid depth value around a pixel if the center pixel has no depth"""
        height, width = depth_image.shape
        
        for r in range(1, radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        depth_val = depth_image[ny, nx]
                        if depth_val > 0:
                            return depth_val
        return 0
    
    def transform_to_robot_frame(self, camera_coords):
        """Transform coordinates from camera frame to robot base frame"""
        # This is a simplified transformation
        # In a real implementation, you would use proper TF transformations
        
        # Assuming camera is mounted on the robot head with known transformation
        # These values should be calibrated for your specific setup
        camera_to_base_transform = np.array([
            [0, 0, 1, 0.13],   # Camera X -> Robot Y
            [-1, 0, 0, 0.0],   # Camera Y -> Robot -X  
            [0, -1, 0, 1.2],   # Camera Z -> Robot -Z (height adjustment)
            [0, 0, 0, 1]
        ])
        
        # Convert to homogeneous coordinates
        camera_homo = np.append(camera_coords, 1)
        
        # Transform to robot frame
        robot_homo = camera_to_base_transform @ camera_homo
        robot_coords = robot_homo[:3]
        
        return robot_coords
    
    def is_target_reachable(self, target_coords):
        """Check if the target coordinates are within robot's reach"""
        x, y, z = target_coords
        
        # Check distance from robot base
        distance = np.sqrt(x**2 + y**2)
        
        if distance > self.max_reach_distance:
            self.node.get_logger().warning(f'Target too far: {distance:.2f}m > {self.max_reach_distance}m')
            return False
        
        # Check height constraints
        if z < 0.1 or z > 1.5:  # Robot can reach between 10cm and 150cm height
            self.node.get_logger().warning(f'Target height out of range: {z:.2f}m')
            return False
        
        return True
    
    def execute_grasp_sequence(self, target_coords):
        """Execute the actual grasp sequence"""
        try:
            self.current_state = "grasping"
            
            x, y, z = target_coords
            
            if self.robot is None:
                # Simulation mode
                self.node.get_logger().info(f'Simulation: Executing grasp sequence at coordinates [{x:.3f}, {y:.3f}, {z:.3f}]')
                self.node.get_logger().info('Simulation: Opening gripper')
                time.sleep(0.5)
                self.node.get_logger().info('Simulation: Moving to approach position')
                time.sleep(1.0)
                self.node.get_logger().info('Simulation: Extending arm towards target')
                time.sleep(1.0)
                self.node.get_logger().info('Simulation: Fine positioning')
                time.sleep(0.5)
                self.node.get_logger().info('Simulation: Closing gripper')
                time.sleep(0.5)
                self.node.get_logger().info('Simulation: Lifting object')
                time.sleep(0.5)
                self.node.get_logger().info('Simulation: Retracting arm')
                time.sleep(0.5)
                self.node.get_logger().info('Simulation: Grasp sequence completed successfully')
                return True
            
            # Real robot operation
            # Open gripper first
            self.robot.end_of_arm.move_to('stretch_gripper', -50)
            self.robot.push_command()
            time.sleep(1.0)
            
            # Move to approach position (slightly behind target)
            approach_x = x - self.approach_distance
            approach_z = z + self.grasp_height_offset
            
            # Move lift to target height
            self.robot.lift.move_to(approach_z)
            self.robot.push_command()
            time.sleep(2.0)
            
            # Extend arm towards target
            arm_extension = max(0.0, min(approach_x, 0.5))  # Limit arm extension
            self.robot.arm.move_to(arm_extension)
            self.robot.push_command()
            time.sleep(2.0)
            
            # Fine approach - move closer to target
            final_arm_extension = max(0.0, min(x, 0.5))
            self.robot.arm.move_to(final_arm_extension)
            self.robot.push_command()
            time.sleep(1.5)
            
            # Close gripper to grasp
            self.robot.end_of_arm.move_to('stretch_gripper', 20)
            self.robot.push_command()
            time.sleep(2.0)
            
            # Lift object slightly
            self.robot.lift.move_to(approach_z + 0.1)
            self.robot.push_command()
            time.sleep(1.0)
            
            # Retract arm
            self.robot.arm.move_to(0.1)
            self.robot.push_command()
            time.sleep(2.0)
            
            self.node.get_logger().info('Grasp sequence completed successfully')
            return True
            
        except Exception as e:
            self.node.get_logger().error(f'Error in grasp sequence: {str(e)}')
            return False
    
    def get_robot_pose(self):
        """Get current robot pose"""
        pose = Pose()
        
        if self.robot is None:
            # Simulation mode - return default pose
            pose.position.x = 0.0
            pose.position.y = 0.0
            pose.position.z = 0.0
            return pose
        
        pose.position.x = self.robot.base.status['x']
        pose.position.y = self.robot.base.status['y']
        pose.position.z = 0.0
        
        # Convert theta to quaternion (simplified - only yaw rotation)
        theta = self.robot.base.status['theta']
        pose.orientation.z = np.sin(theta / 2)
        pose.orientation.w = np.cos(theta / 2)
        
        return pose
    
    def is_gripper_open(self):
        """Check if gripper is open"""
        if self.robot is None:
            return False
        
        gripper_pos = self.robot.end_of_arm.status['stretch_gripper']['pos']
        return gripper_pos < 0  # Negative values indicate open gripper
    
    def emergency_stop(self):
        """Emergency stop the robot"""
        if self.robot is not None:
            self.robot.stop()
            self.current_state = "error"
            self.last_error = "Emergency stop activated"
            self.node.get_logger().warning('Emergency stop activated')
    
    def shutdown(self):
        """Shutdown the robot connection"""
        if self.robot is not None:
            try:
                self.robot.stop()
                self.robot.close()
            except:
                pass


class StretchVLAControlNode(Node):
    def __init__(self):
        super().__init__('stretch_vla_control_node')
        
        # Declare parameters with default values
        self.declare_parameter('robot.max_reach_distance', 0.8)
        self.declare_parameter('robot.approach_distance', 0.1)
        self.declare_parameter('robot.grasp_height_offset', 0.05)
        self.declare_parameter('robot.arm.max_extension', 0.5)
        self.declare_parameter('robot.arm.default_speed', 0.1)
        self.declare_parameter('robot.lift.min_height', 0.1)
        self.declare_parameter('robot.lift.max_height', 1.5)
        self.declare_parameter('robot.lift.default_speed', 0.1)
        self.declare_parameter('robot.lift.home_height', 0.6)
        self.declare_parameter('robot.gripper.open_position', -50)
        self.declare_parameter('robot.gripper.close_position', 20)
        self.declare_parameter('robot.gripper.grasp_force', 20)
        self.declare_parameter('robot.base.max_linear_velocity', 0.5)
        self.declare_parameter('robot.base.max_angular_velocity', 1.0)
        self.declare_parameter('camera.depth_scale', 0.001)
        
        # Initialize controller
        self.controller = StretchVLAController(self)
        
        # Publishers
        self.robot_status_pub = self.create_publisher(
            RobotStatus, '/robot_status', 10)
        
        # Subscribers
        self.grasp_target_sub = self.create_subscription(
            GraspTarget, '/grasp_target', self.grasp_target_callback, 10)
        
        self.annotated_image_sub = self.create_subscription(
            AnnotatedImage, '/annotated_image', self.annotated_image_callback, 10)
        
        self.depth_image_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_image_callback, 10)
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/depth/camera_info', self.camera_info_callback, 10)
        
        # Services
        self.execute_grasp_service = self.create_service(
            ExecuteGrasp, 'execute_grasp', self.execute_grasp_callback)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Current data
        self.current_depth_image = None
        
        # Status publishing timer
        self.status_timer = self.create_timer(0.5, self.publish_status)  # 2 Hz
        
        self.get_logger().info('Stretch VLA Control Node initialized')
    
    def grasp_target_callback(self, msg):
        """Handle grasp target messages"""
        self.get_logger().info(
            f'Received grasp target: ({msg.target_position_2d.x:.1f}, {msg.target_position_2d.y:.1f}), '
            f'confidence: {msg.confidence:.2f}, object: {msg.object_description}')
        
        # Update target tracking
        self.controller.update_target_tracking(msg)
        
        # Move to target (instead of immediately grasping)
        success, message = self.controller.move_to_target(msg, self.current_depth_image)
        
        if success:
            self.get_logger().info(f'Movement to target successful: {message}')
        else:
            self.get_logger().warning(f'Movement to target failed: {message}')
    
    def annotated_image_callback(self, msg):
        """Handle annotated image messages with multiple targets"""
        self.get_logger().info(f'Received annotated image with {len(msg.detected_targets)} targets')
        
        # If we have targets, use the most confident one
        if msg.detected_targets:
            # Sort targets by confidence
            sorted_targets = sorted(msg.detected_targets, key=lambda t: t.confidence, reverse=True)
            best_target = sorted_targets[0]
            
            self.get_logger().info(
                f'Selected best target: {best_target.object_description} '
                f'(confidence: {best_target.confidence:.2f})'
            )
            
            # Update target tracking
            self.controller.update_target_tracking(best_target)
            
            # Move to the best target
            success, message = self.controller.move_to_target(best_target, self.current_depth_image)
            
            if success:
                self.get_logger().info(f'Movement to annotated target successful: {message}')
            else:
                self.get_logger().warning(f'Movement to annotated target failed: {message}')
    
    def depth_image_callback(self, msg):
        """Handle depth image messages"""
        try:
            self.current_depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {str(e)}')
    
    def camera_info_callback(self, msg):
        """Handle camera info messages"""
        self.controller.coordinate_transformer.set_camera_info(msg)
    
    def execute_grasp_callback(self, request, response):
        """Handle execute grasp service requests"""
        success, message = self.controller.execute_grasp(
            request.target, self.current_depth_image)
        
        response.success = success
        response.result_message = message
        response.final_pose = self.controller.get_robot_pose()
        
        return response
    
    def publish_status(self):
        """Publish robot status"""
        msg = RobotStatus()
        msg.current_state = self.controller.current_state
        msg.current_pose = self.controller.get_robot_pose()
        msg.is_gripper_open = self.controller.is_gripper_open()
        msg.last_error = self.controller.last_error
        msg.timestamp = self.get_clock().now().nanoseconds
        
        self.robot_status_pub.publish(msg)
    
    def destroy_node(self):
        """Clean up when node is destroyed"""
        self.controller.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StretchVLAControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
