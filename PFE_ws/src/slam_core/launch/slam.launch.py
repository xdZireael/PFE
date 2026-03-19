from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock (true when running in Gazebo)'
    )

    use_lidar_arg = DeclareLaunchArgument(
        'use_lidar',
        default_value='true',
        description='Enable the LiDAR frontend'
    )

    use_camera_arg = DeclareLaunchArgument(
        'use_camera',
        default_value='true',
        description='Enable the camera frontend'
    )

    vocab_path_arg = DeclareLaunchArgument(
        'vocab_path',
        default_value='',
        description='Path to ORBvoc.txt for the camera frontend BoW'
    )

    lidar_frontend_node = Node(
        package='lidar_frontend',
        executable='lidar_frontend_node',
        name='lidar_frontend',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_lidar')),
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }]
    )

    camera_frontend_node = Node(
        package='camera_frontend',
        executable='camera_frontend_node',
        name='camera_frontend',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_camera')),
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'vocab_path':   LaunchConfiguration('vocab_path'),
        }]
    )

    slam_core_node = Node(
        package='slam_core',
        executable='slam_core_node',
        name='slam_core',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }]
    )

    return LaunchDescription([
        use_sim_time_arg,
        use_lidar_arg,
        use_camera_arg,
        vocab_path_arg,
        lidar_frontend_node,
        camera_frontend_node,
        slam_core_node,
    ])