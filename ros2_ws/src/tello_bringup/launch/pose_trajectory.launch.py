#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_bringup = get_package_share_directory('tello_bringup')
    shared_param_file = os.path.join(pkg_bringup, 'config', 'pose_trajectory_config.yaml')

    tello_driver_node = Node(
        package='tello_driver',
        executable='tello_driver_node',
        name='tello_driver_node',
        parameters=[shared_param_file],
        output='screen'
    )
    
    pose_estimator_node = Node(
        package='tello_vision',
        executable='pose_estimator',
        name='pose_estimator',
        parameters=[shared_param_file],
        output='screen'
    )
    
    waypoint_service_node = Node(
        package='tello_control',
        executable='waypoint_service',
        name='waypoint_service',
        parameters=[shared_param_file],
        output='screen'
    )
    
    pose_controller_node = Node(
        package='tello_control',
        executable='pose_controller',
        name='pose_controller',
        parameters=[shared_param_file],
        output='screen'
    )
    
    return LaunchDescription([
        tello_driver_node,
        pose_estimator_node,
        waypoint_service_node,
        pose_controller_node
    ])