#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get package directory
    pkg_share = get_package_share_directory('stretch_vla_control')
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(pkg_share, 'config', 'robot_params.yaml'),
        description='Path to robot configuration file'
    )
    
    # Control node
    control_node = Node(
        package='stretch_vla_control',
        executable='control_node',
        name='stretch_vla_control_node',
        parameters=[LaunchConfiguration('config_file')],
        output='screen',
        emulate_tty=True
    )
    
    return LaunchDescription([
        config_file_arg,
        control_node
    ])
