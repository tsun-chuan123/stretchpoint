#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='wentao-yuan/robopoint-v1-vicuna-v1.5-13b',
        description='Path or name of the model to load'
    )
    
    model_type_arg = DeclareLaunchArgument(
        'model_type',
        default_value='transformers',
        description='Model backend type: transformers or ollama'
    )
    
    remote_arg = DeclareLaunchArgument(
        'remote',
        default_value='false',
        description='Use remote computer for processing'
    )
    
    # LLaVA Server Node
    llava_server_node = Node(
        package='llava_server',
        executable='server_node',
        name='llava_server_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'model_type': LaunchConfiguration('model_type'),
            'use_remote_computer': LaunchConfiguration('remote'),
        }],
        arguments=[
            '--model-type', LaunchConfiguration('model_type'),
            '--model-path', LaunchConfiguration('model_path'),
        ]
    )
    
    return LaunchDescription([
        model_path_arg,
        model_type_arg,
        remote_arg,
        llava_server_node
    ])
