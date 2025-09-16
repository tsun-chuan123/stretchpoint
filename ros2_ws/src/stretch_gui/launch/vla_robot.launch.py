#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    use_remote_arg = DeclareLaunchArgument(
        'use_remote',
        default_value='true',
        description='Whether to use remote VLA server'
    )
    
    # GUI node
    gui_node = Node(
        package='stretch_gui',
        executable='gui_node',
        name='stretch_gui_node',
        parameters=[{
            'use_remote_computer': LaunchConfiguration('use_remote')
        }],
        arguments=['--remote'] if LaunchConfiguration('use_remote') == 'true' else [],
        output='screen'
    )
    
    # Camera image sender
    camera_sender = Node(
        package='stretch_gui',
        executable='send_camera_images',
        name='camera_image_sender',
        arguments=['--remote'] if LaunchConfiguration('use_remote') == 'true' else [],
        output='screen'
    )
    
    return LaunchDescription([
        use_remote_arg,
        gui_node,
        camera_sender,
    ])
